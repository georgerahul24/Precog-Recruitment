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
batch_size = 16
epochs = 20
learning_rate = 0.001


# Custom Dataset
class WordImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load data and labels (number of letters in each word)
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".png"):
                        self.data.append(os.path.join(folder_path, file))
                        self.labels.append(len(folder))  # Number of letters in the folder name (assumes folder name is a word)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


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

# Neural Network for Regression (predict number of letters)
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value (number of letters)
        )

    def forward(self, x):
        return self.network(x)


# Model, Loss, Optimizer
model = ComplexNN()
criterion = nn.MSELoss()  # Use MSELoss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if Metal backend is available (for M1 Mac)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU (Metal)
model = model.to(device)

# Store accuracy and loss for visualization
train_losses = []
test_losses = []


# Training Loop
def train_and_record_model():
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())  # Squeeze output to match the shape of labels
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluate loss after each epoch
        test_loss = evaluate_model(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")


# Evaluate Loss
def evaluate_model(dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()

    return total_loss / len(dataloader)


# Run Training and Evaluation
print("Starting training...")
train_and_record_model()
print("Training complete.")

# Save the model
torch.save(model, "model.pth")

# Generate loss graph
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, label="Training Loss")
plt.plot(epochs_range, test_losses, label="Testing Loss")
plt.title("Train and Test Loss Per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
