import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam

SEQUENCES = ["S0103", "S0104", "S0304"]

## CONFIG ##

choosen_sequence = SEQUENCES[0]
DATA_PATH = f'data/triplets_data/{choosen_sequence}'
MODELS_PATH = 'models'
mode = "train" # "train" or "test"

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

dataset = TripletDataset(root_dir=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

## MODEL DEFINITION ##

class TripletEfficientNet(nn.Module):
    def __init__(self):
        super(TripletEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')  # Use EfficientNet-B0
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, 256)  # Replace the classifier with a new embedding layer

    def forward(self, x):
        return self.base_model(x)

triplet_loss = nn.TripletMarginLoss(margin=1.0)

match mode:
    case "train":
        ## TRAINING ##
        model = TripletEfficientNet()
        optimizer = Adam(model.parameters(), lr=0.001)

        # Example training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            for anchors, positives, negatives in dataloader:  # Assuming dataloader is defined
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
        ## INFERENCE ##
            
        def get_embedding(model, image):
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                embedding = model(image)
            return embedding

        # Assuming `image` is a preprocessed image tensor
        # embedding = get_embedding(model, image)