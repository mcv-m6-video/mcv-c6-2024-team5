import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam


SEQUENCES = ["S0103", "S0104", "S0304"]

## CONFIG ##

combination = 0
choosen_sequence = SEQUENCES[combination]
DATA_PATH = f'data/triplets_data/{choosen_sequence}'
MODELS_PATH = 'models'
mode = "train" # "train" or "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SEQUENCES = ["S04", "S03", "S01"]
test_sequence = TEST_SEQUENCES[combination]
TEST_PATH = f'data/aic19-track1-mtmc-train/train/{test_sequence}/triplets'

## DATA LOADING ##

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the triplets.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
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

## MODEL DEFINITION ##

class TripletEfficientNet(nn.Module):
    def __init__(self):
        super(TripletEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')  # Use EfficientNet-B0
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, 256)  # Replace the classifier with a new embedding layer

    def forward(self, x):
        return self.base_model(x)

def get_dataset_split(dataset, train_split=0.75):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def validate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for anchors, positives, negatives in dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# ? ToTensor() not needed because read_image already returns a tensor
train_transform = Compose([
    # Resize((256, 256)),
    # RandomResizedCrop((224, 224)),
    Resize((224, 224)),
    RandomHorizontalFlip(),
    RandomRotation(10),  # Rotates by degrees (-10, 10)
    # Assuming read_image gives tensor in [0, 255], convert to float and scale to [0, 1]
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    lambda x: x.float() / 255,
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = Compose([
    Resize((224, 224)),
    # Assuming read_image gives tensor in [0, 255], convert to float and scale to [0, 1]
    lambda x: x.float() / 255,
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
triplet_loss = nn.TripletMarginLoss(margin=1.0)

def main():
    match mode:
        case "train":
            # Apply the split
            dataset = TripletDataset(root_dir=DATA_PATH, transform=train_transform)  # Assuming training mode
            train_dataset, val_dataset = get_dataset_split(dataset)

            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

            ## TRAINING ##
            model = TripletEfficientNet().to(device)
            model.train()  # Set the model to training mode
            optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

            min_val_loss = float('inf')
            best_model = None
            # Example training loop
            num_epochs = 20
            for epoch in range(num_epochs):
                running_loss = 0.0
                for anchors, positives, negatives in train_dataloader:  # Assuming dataloader is defined
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
                avg_val_loss = validate_model(model, val_dataloader, device)
                print(f"Epoch {epoch}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}")

                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    # Save the model in a variable
                    best_model = model

            # Save the model
            os.makedirs(MODELS_PATH, exist_ok=True)
            model_path = os.path.join(MODELS_PATH, f'triplet_model_{choosen_sequence}.pt')
            torch.save(model.state_dict(), model_path)
            # Save the best model
            best_model_path = os.path.join(MODELS_PATH, f'triplet_model_best_{choosen_sequence}.pt')
            torch.save(best_model.state_dict(), best_model_path)
        case "test":
            dataset = TripletDataset(root_dir=TEST_PATH, transform=test_transform)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            ## TESTING ##
            model = TripletEfficientNet().to(device)
            model.load_state_dict(torch.load(f'models/triplet_model_{choosen_sequence}.pt'))
            model.eval()  # Set the model to evaluation mode

            all_valid_triplets = 0
            all_triplets = 0
            distance_threshold = None
            general_positive_distance_mean = None
            general_negative_distance_mean = None
            # Example testing loop
            for anchors, positives, negatives in dataloader:  # Assuming dataloader is defined
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)

                # Compute the distances between the anchor and positive, and anchor and negative
                positive_distance = (anchor_embeddings - positive_embeddings).pow(2).sum(1)
                negative_distance = (anchor_embeddings - negative_embeddings).pow(2).sum(1)

                # The triplet is considered valid if the distance to the positive is smaller than the distance to the negative
                valid_triplets = positive_distance < negative_distance
                valid_triplets_len = len(valid_triplets)
                valid_triplets = valid_triplets.sum().item()
                print(f"Valid triplets: {valid_triplets} out of {valid_triplets_len}")

                # Mean distance to the positive and negative
                positive_distance_mean = positive_distance.mean().item()
                negative_distance_mean = negative_distance.mean().item()
                print(f"Mean distance to positive: {positive_distance_mean}")
                print(f"Mean distance to negative: {negative_distance_mean}")

                if distance_threshold is None:
                    distance_threshold = (positive_distance_mean + negative_distance_mean) / 2
                else:
                    distance_threshold = (distance_threshold + (positive_distance_mean + negative_distance_mean) / 2) / 2

                if general_positive_distance_mean is None:
                    general_positive_distance_mean = positive_distance_mean
                else:
                    general_positive_distance_mean = (general_positive_distance_mean + positive_distance_mean) / 2

                if general_negative_distance_mean is None:
                    general_negative_distance_mean = negative_distance_mean
                else:
                    general_negative_distance_mean = (general_negative_distance_mean + negative_distance_mean) / 2

                all_valid_triplets += valid_triplets
                all_triplets += valid_triplets_len            

            print(f"Total valid triplets: {all_valid_triplets} out of {all_triplets}")
            print(f"General positive distance mean: {general_positive_distance_mean}")
            print(f"General negative distance mean: {general_negative_distance_mean}")
            print(f"Proposed distance threshold: {distance_threshold}")     


SEQUENCE_METADATA = {
    "S01": {
        "model": "triplet_model_S0304.pt",
        "distance_threshold": 28.85,
    },
    "S03": {
        "model": "triplet_model_S0104.pt",
        "distance_threshold": 21.11,
    },
    "S04": {
        "model": "triplet_model_S0103.pt",
        "distance_threshold": 35.51,
    },
}

## INFERENCE ##

def get_embedding(image, sequence):
    model = TripletEfficientNet().to(device)
    model.load_state_dict(torch.load(f'models/{SEQUENCE_METADATA[sequence]["model"]}'))

    # Image needs to be a tensor
    inference_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        # Assuming read_image gives tensor in [0, 255], convert to float and scale to [0, 1]
        lambda x: x.float() / 255,
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Apply the transform
    image = inference_transform(image).to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        embedding = model(image)
    return embedding

if __name__ == "__main__":
    main()