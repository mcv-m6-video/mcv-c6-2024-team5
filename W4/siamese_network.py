import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, Lambda
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam

SEQUENCES = ["S0103", "S0104", "S0304"]

## CONFIG ##

choosen_sequence = SEQUENCES[2]
DATA_PATH = f'data/triplets_data/{choosen_sequence}'
MODELS_PATH = 'models'
mode = "test" # "train" or "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SEQUENCES = ["S01", "S03", "S04"]
test_sequence = TEST_SEQUENCES[2]
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

# Example usage:
transform = Compose([
    Resize((224, 224)),
    # ToTensor(), # ! It's not needed because read_image already returns a tensor
    # Assuming read_image gives tensor in [0, 255], convert to float and scale to [0, 1]
    lambda x: x.float() / 255,
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## MODEL DEFINITION ##

class TripletEfficientNet(nn.Module):
    def __init__(self):
        super(TripletEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')  # Use EfficientNet-B0
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, 256)  # Replace the classifier with a new embedding layer

    def forward(self, x):
        return self.base_model(x)

triplet_loss = nn.TripletMarginLoss(margin=1.0)

def main():
    match mode:
        case "train":
            dataset = TripletDataset(root_dir=DATA_PATH, transform=transform)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            ## TRAINING ##
            model = TripletEfficientNet().to(device)
            optimizer = Adam(model.parameters(), lr=0.0001)

            # Example training loop
            num_epochs = 20
            for epoch in range(num_epochs):
                for anchors, positives, negatives in dataloader:  # Assuming dataloader is defined
                    anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

                    anchor_embeddings = model(anchors)
                    positive_embeddings = model(positives)
                    negative_embeddings = model(negatives)

                    loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch}, Loss: {loss.item()}")

            # Save the model
            os.makedirs(MODELS_PATH, exist_ok=True)
            model_path = os.path.join(MODELS_PATH, f'triplet_model_{choosen_sequence}.pt')
            torch.save(model.state_dict(), model_path)
        case "test":
            dataset = TripletDataset(root_dir=TEST_PATH, transform=transform)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            ## TESTING ##
            model = TripletEfficientNet().to(device)
            model.load_state_dict(torch.load(f'models/triplet_model_{choosen_sequence}.pt'))
            model.eval()  # Set the model to evaluation mode

            all_valid_triplets = 0
            all_triplets = 0
            distance_threshold = None
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

                all_valid_triplets += valid_triplets
                all_triplets += valid_triplets_len
            
            print(f"Total valid triplets: {all_valid_triplets} out of {all_triplets}")
            print(f"Proposed distance threshold: {distance_threshold}")

            ## INFERENCE ##
                
            # def get_embedding(model, image):
            #     model.eval()  # Set the model to evaluation mode
            #     with torch.no_grad():
            #         embedding = model(image)
            #     return embedding

            # Assuming `image` is a preprocessed image tensor
            # embedding = get_embedding(model, image)

if __name__ == "__main__":
    main()