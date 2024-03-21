import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam

# Custom Transform
class ToFloatScale(object):
    """Convert a tensor image to float and scale it to [0, 1]."""
    def __call__(self, tensor):
        return tensor.float() / 255

    def __repr__(self):
        return self.__class__.__name__ + '()'

# Dataset Class
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.triplets = self._load_triplets(root_dir)
        self.transform = transform

    def _load_triplets(self, root_dir):
        triplets = []
        if os.path.isdir(root_dir):
            for filename in os.listdir(root_dir):
                if "anchor" in filename:
                    base_name = filename.split("_anchor")[0]
                    triplets.append({
                        "anchor": os.path.join(root_dir, f"{base_name}_anchor.jpg"),
                        "positive": os.path.join(root_dir, f"{base_name}_positive.jpg"),
                        "negative": os.path.join(root_dir, f"{base_name}_negative.jpg"),
                    })
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor = read_image(triplet["anchor"])
        positive = read_image(triplet["positive"])
        negative = read_image(triplet["negative"])

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

# Model Definition
class TripletEfficientNet(nn.Module):
    def __init__(self):
        super(TripletEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, 256)

    def forward(self, x):
        return self.base_model(x)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Triplet network training and testing script.")
    parser.add_argument("--combination", type=int, default=1, choices=[0, 1, 2], help="Index for selecting sequence. 0: S0103 data for S04, 1: S0104 for S03, 2: S0304 for S01.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Operating mode: train or test.")
    parser.add_argument("--data_path", type=str, default="data/triplets_data", help="Base directory for the dataset.")
    parser.add_argument("--models_path", type=str, default="models", help="Directory to save models.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for the triplet loss.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    SEQUENCES = ["S0103", "S0104", "S0304"]
    choosen_sequence = SEQUENCES[args.combination]
    DATA_PATH = os.path.join(args.data_path, choosen_sequence)
    MODELS_PATH = args.models_path
    TEST_SEQUENCES = ["S04", "S03", "S01"]
    test_sequence = TEST_SEQUENCES[args.combination]
    TEST_PATH = f'data/aic19-track1-mtmc-train/train/{test_sequence}/triplets'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplet_loss = nn.TripletMarginLoss(margin=args.margin)

    # Define transformations for training and testing
    train_transform = Compose([
        Resize((224, 224)),
        ToFloatScale(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        Resize((224, 224)),
        ToFloatScale(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if args.mode == "train":
        dataset = TripletDataset(root_dir=DATA_PATH, transform=train_transform)
        train_size = int(0.75 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = TripletEfficientNet().to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        min_val_loss = float('inf')
        best_model = None

        model.train()  # Set the model to training mode
        for epoch in range(args.epochs):  # Number of epochs
            running_loss = 0.0
            for anchors, positives, negatives in train_dataloader:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                optimizer.zero_grad()

                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_dataloader)
            print(f"Epoch {epoch}, Avg Train Loss: {avg_train_loss}")

            # Validation step
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for anchors, positives, negatives in val_dataloader:
                    anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                    anchor_embeddings = model(anchors)
                    positive_embeddings = model(positives)
                    negative_embeddings = model(negatives)
                    loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch}, Avg Val Loss: {avg_val_loss}")

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_model = model

        # Save the model
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_path = os.path.join(MODELS_PATH, f'triplet_model_{choosen_sequence}.pt')
        torch.save(model.state_dict(), model_path)
        # Save the best model
        best_model_path = os.path.join(MODELS_PATH, f'triplet_model_best_{choosen_sequence}.pt')
        torch.save(best_model.state_dict(), best_model_path)

    elif args.mode == "test":
        dataset = TripletDataset(root_dir=TEST_PATH, transform=test_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        model = TripletEfficientNet().to(device)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, f'triplet_model_{choosen_sequence}.pt')))
        model.eval()  # Set the model to evaluation mode

        general_positive_distance_mean = None
        general_negative_distance_mean = None
        with torch.no_grad():
            for anchors, positives, negatives in dataloader:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)
                
                positive_distance = (anchor_embeddings - positive_embeddings).pow(2).sum(1)
                negative_distance = (anchor_embeddings - negative_embeddings).pow(2).sum(1)

                general_positive_distance_mean = positive_distance.mean().item() if general_positive_distance_mean is None else (general_positive_distance_mean + positive_distance.mean().item()) / 2
                general_negative_distance_mean = negative_distance.mean().item() if general_negative_distance_mean is None else (general_negative_distance_mean + negative_distance.mean().item()) / 2

        print(f"General positive distance mean: {general_positive_distance_mean}")
        print(f"General negative distance mean: {general_negative_distance_mean}")


    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")

if __name__ == "__main__":
    main()