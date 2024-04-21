""" Main script for training a video classification model on HMDB51 dataset. """

import os
import json
import argparse
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics
from utils import visualization
from utils.early_stopping import EarlyStopping
from collections import defaultdict

def train(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        description: str = ""
    ) -> (float, float):
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        acc_mean (float): The mean accuracy of the model on the training dataset.
        final_loss_mean (float): The mean loss of the model on the training dataset.
    """
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        batch_clips, labels = batch['clips'], batch['labels'].to(device)
        # Compute loss
        clips_per_video = 1
        if batch_clips.dim() == 6:
            clips_per_video = batch_clips.size(1)
            batch_clips = batch_clips.view(-1, *batch_clips.size()[2:])
        outputs = model(batch_clips.to(device))
        if clips_per_video > 1:
            outputs = outputs.view(-1, clips_per_video, outputs.size(1)).mean(dim=1)
        loss = loss_fn(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()
        hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )
    acc_mean = float(hits) / count
    final_loss_mean = loss_train_mean.get_mean()
    return acc_mean, final_loss_mean



def evaluate(
        model: nn.Module,
        valid_loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        description: str = "",
        compute_per_class_acc: bool = False
    ) -> (float, float):
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".
        compute_per_class_acc (bool, optional): Whether to compute accuracy per class. Defaults to False.

    Returns:
        acc_mean (float): The mean accuracy of the model on the validation dataset.
        final_loss_mean (float): The mean loss of the model on the validation dataset.
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    class_hits = defaultdict(int)
    class_totals = defaultdict(int)

    for batch in pbar:
        # Gather batch and move to device
        # batched clips is (num_videos, num_clips_per_video, C, T, H, W)
        # Labels is (num_videos * num_clips_per_video)
        batch_clips, labels = batch['clips'], batch['labels'].to(device)
        
        # Forward pass
        with torch.no_grad():
            # outputs = torch.stack(outputs, dim=0)
            clips_per_video = 1
            if batch_clips.dim() == 6:
                clips_per_video = batch_clips.size(1)
                batch_clips = batch_clips.view(-1, *batch_clips.size()[2:])
            outputs = model(batch_clips.to(device))
            if clips_per_video > 1:
                outputs = outputs.view(-1, clips_per_video, outputs.size(1)).mean(dim=1)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels)
            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )

        if compute_per_class_acc:
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_hits[label.item()] += 1
                class_totals[label.item()] += 1
    
    acc_mean = float(hits) / count
    final_loss_mean = loss_valid_mean.get_mean()
    if not compute_per_class_acc:
        return acc_mean, final_loss_mean

    class_accuracies = {cls: class_hits[cls] / class_totals[cls] for cls in class_totals}
    class_names = valid_loader.dataset.CLASS_NAMES
    class_accuracies_with_names = {class_names[cls]: acc for cls, acc in class_accuracies.items()}

    return acc_mean, final_loss_mean, class_accuracies_with_names

def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        clips_per_video: int,
        crops_per_clip: int,
        tsn_k: int,
        deterministic: bool
) -> Dict[str, HMDB51Dataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        clips_per_video (int): Number of clips to sample from each video.
        crops_per_clip (int): Number of crops to sample from each clip.
        tsn_k  (int): Number of clips to sample per video for TSN aggregation

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
            clips_per_video,
            crops_per_clip,
            tsn_k,
            deterministic
        )

    return datasets


def create_dataloaders(
        datasets: Dict[str, HMDB51Dataset],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return dataloaders


def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given parameters.
    
    Args:
        optimizer_name (str): Name of the optimizer (supported: "adam" and "sgd" for now).
        parameters (Iterator[nn.Parameter]): Iterator over model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        torch.optim.Optimizer: The optimizer for the model parameters.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def print_model_summary(
        model: nn.Module,
        clip_length: int,
        crop_size: int,
        print_model: bool = True,
        print_params: bool = True,
        print_FLOPs: bool = True
    ) -> (float, float):
    """
    Prints a summary of the given model.

    Args:
        model (nn.Module): The model for which to print the summary.
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        print_model (bool, optional): Whether to print the model architecture. Defaults to True.
        print_params (bool, optional): Whether to print the number of parameters. Defaults to True.
        print_FLOPs (bool, optional): Whether to print the number of FLOPs. Defaults to True.

    Returns:
        None
    """
    if print_model:
        print(model)
    num_params = sum(p.numel() for p in model.parameters())
    # num_params = model_analysis.calculate_parameters(model) # should be equivalent
    num_params = round(num_params / 10e6, 2)
    if print_params:
        print(f"Number of parameters (M): {num_params}")

    num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
    num_FLOPs = round(num_FLOPs / 10e9, 2)
    if print_FLOPs:
        print(f"Number of FLOPs (G): {num_FLOPs}")

    return num_params, num_FLOPs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('frames_dir', type=str,
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--clip-length', type=int, default=4,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--model-name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                    help='Load pretrained weights for the model (if available)')
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=8,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Number of epochs after which to validate the model')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--early-stopping', type=int, default=5,
                        help='Number of epochs to wait after last time validation loss improved')
    parser.add_argument('--wandb', action='store_true', default=False, 
                        help='Use Weights & Biases for logging')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load a model from a file')
    parser.add_argument('--only-inference', action='store_true', default=False,
                        help='Only perform inference on the test set (requires a model [--load-model] to load)')
    parser.add_argument('--clips-per-video', type=int, default=3,
                        help='Number of clips to sample per video')
    parser.add_argument('--crops-per-clip', type=int, default=1,
                        help='Number of spatial crops to sample per clip')
    parser.add_argument('--tsn-k', type=int, default=5,
                        help='Number of clips to sample per video for TSN aggregation')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use our deterministic method, TSN by default if this flag is not set')

    args = parser.parse_args()

    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        clips_per_video=args.clips_per_video,
        crops_per_clip=args.crops_per_clip,
        tsn_k=args.tsn_k,
        deterministic=args.deterministic
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    # Init model, optimizer, and loss function
    model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    num_params, num_FLOPs = print_model_summary(model, args.clip_length, args.crop_size)

    model = model.to(args.device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=args.device))
        model_name = args.load_model.split("/")[-1].split(".")[0]
    else:
        model_name = args.model_name + "_hmdb51"

    if args.wandb:
        config = vars(args)
        config['num_params_M'] = num_params
        config['num_GFLOPs'] = num_FLOPs
        wandb.init(project="ActionClassificationTask-W5", entity="mcv-c6-2024-team5",
                   config=config)
        wandb.watch(model)
        model_name = wandb.run.name + "_hmdb51"

    if args.only_inference:
        if not args.load_model:
            raise ValueError("Please provide a model to load for inference")
        test_acc_mean, test_loss_mean, acc_per_class = evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing", compute_per_class_acc=True)
        # Save image of accuracy per class
        save_path = f"results/{model_name}"
        plt = visualization.plot_acc_per_class(acc_per_class, save_path=save_path + "/acc_per_class.png")
        print(f"Clips_per_video =  {args.clips_per_video}, Crops_per_clip = {args.crops_per_clip}. Results: Test accuracy: {test_acc_mean}, Test loss: {test_loss_mean}")
        print(acc_per_class)
        exit()

    # Initialize the early stopper with desired patience
    early_stopper = EarlyStopping(patience=args.early_stopping, verbose=True)

    for epoch in range(args.epochs):
        # Training
        description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
        train_acc_mean, train_loss_mean = train(model, loaders['training'], optimizer, loss_fn, args.device, description=description)
        if args.wandb:
            wandb.log({"train_acc": train_acc_mean, "train_loss": train_loss_mean}, step=epoch)
        # Validation
        if epoch % args.validate_every == 0:
            description = f"Validation [Epoch: {epoch + 1}/{args.epochs}]"
            val_acc_mean, val_loss_mean = evaluate(model, loaders['validation'], loss_fn, args.device, description=description)
            if args.wandb:
                wandb.log({"val_acc": val_acc_mean, "val_loss": val_loss_mean}, step=epoch)
            early_stopper(val_loss_mean)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Testing
    test_acc_mean, test_loss_mean, acc_per_class = evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing", compute_per_class_acc=True)

    # Check if results folder exists
    if not os.path.exists("results"):
        os.makedirs("results")
    save_path = f"results/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.wandb:
        wandb.log({"test_acc": test_acc_mean, "test_loss": test_loss_mean})
        plt = visualization.plot_acc_per_class(acc_per_class, save_path=save_path + "/acc_per_class.png")
        wandb.log({"acc_per_class": wandb.Image(plt)})
    else:
        plt = visualization.plot_acc_per_class(acc_per_class, save_path=save_path + "/acc_per_class.png")
        plt.show()
        # Save the acc_per_class dictionary to a json file
    with open(save_path + "/acc_per_class.json", "w") as f:
        json.dump(acc_per_class, f)

    # Save model
    if not os.path.exists("weights"):
        os.makedirs("weights")
    model_save_path = f"weights/{model_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    if args.wandb:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_save_path)
        artifact.add_file(save_path + "/acc_per_class.json")
        wandb.log_artifact(artifact, type="model")
        wandb.finish()

    exit()
