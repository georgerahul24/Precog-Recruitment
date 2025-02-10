import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from pathlib import Path

from tqdm import tqdm

image_size = (64, 64)
class WordClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WordClassifier, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor before passing to the classifier
        x = self.classifier(x)
        return x

model = None

def load_model_and_labels():
    try:
        global model
        model = torch.load("../Models/LM1.pth")
        model.eval()

        train_dir = "trainLetter"
        label_map = {
            idx: folder for idx, folder in enumerate(sorted(os.listdir(train_dir)))
            if os.path.isdir(os.path.join(train_dir, folder))
        }
        return model, label_map
    except Exception as e:
        print(f"Error loading model or label map: {str(e)}")
        sys.exit(1)


def process_image(image_path, model, label_map, device):
    try:
        # Convert string path to Path object to handle spaces correctly
        image_path = Path(image_path.strip())

        if not image_path.exists():
            print(f"Error: File '{image_path}' does not exist")
            return

        # Configuration
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        tensor_image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(tensor_image)
            _, predicted = torch.max(outputs, 1)

        predicted_class = label_map.get(predicted.item(), "Unknown")

        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Image: {image_path.name}")
        print(f"Predicted Class: {predicted_class}")
        print("-" * 50)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])
class WordImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_map = {}

        # Load data and labels
        for idx, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.label_map[idx] = folder
                for file in os.listdir(folder_path):
                    if file.endswith(".png"):
                        self.data.append(os.path.join(folder_path, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
def evaluate_model(dataloader):
    global device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
device = None
def main():
    global device
    # Check if Metal backend is available (for M1 Mac)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and labels
    print("Loading model...")
    model, label_map = load_model_and_labels()
    model = model.to(device)
    test_dataset = WordImageDataset("testLetter", transform=transform)
    print("Accuracy: ",evaluate_model(DataLoader(test_dataset, batch_size=32, shuffle=False)))
    while True:
        # Get image path from user
        print("\nEnter the path to your image (or 'q' to quit):")
        image_path = input().strip()

        # Check if user wants to quit
        if image_path.lower() == 'q':
            print("Goodbye!")
            break

        # Process the image
        process_image(image_path, model, label_map, device)


if __name__ == "__main__":
    main()