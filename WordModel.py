import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
train_dir = "../train"
test_dir = "../test"
image_size = (256, 128)
batch_size = 32
epochs = 70
learning_rate = 0.001

# Custom Dataset
class WordImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load data and labels (number of letters)
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path) and folder.isdigit():  # Ensure the folder name is a number
                label = int(folder)  # Convert folder name to integer (number of letters)
                for file in os.listdir(folder_path):
                    if file.endswith(".png"):
                        self.data.append(os.path.join(folder_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]  # Number of letters in the word
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)  # Float label for regression

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Load Datasets
train_dataset = WordImageDataset(train_dir, transform=transform)
test_dataset = WordImageDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of testing images: {len(test_dataset)}")

# Neural Network
class WordLengthPredictor(nn.Module):
    def __init__(self):
        super(WordLengthPredictor, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))

        # Regression layer (1 neuron to predict word length)
        self.regressor = nn.Sequential(
            nn.Linear(128 * 4 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 1 neuron for regression
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x  # No activation, since it's a regression task

# Model, Loss, Optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = WordLengthPredictor().to(device)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
from tqdm import tqdm
import shutil

shutil.rmtree('../ModelRegression', ignore_errors=True)
os.makedirs('../ModelRegression', exist_ok=True)

train_losses = []
test_losses = []
model_number = 0

def train_and_record_model():
    global model_number
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)  # Ensure correct shape
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluate model
        test_loss = evaluate_model(test_loader)
        test_losses.append(test_loss)

        torch.save(model, f"../ModelRegression/model_{model_number}_{test_loss:.4f}.pth")
        model_number += 1
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

# Evaluate Test Loss
def evaluate_model(dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Run Training
print("Starting training...")
train_and_record_model()
print("Training complete.")

# Generate loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
plt.title("Training and Testing Loss Per Epoch")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()
