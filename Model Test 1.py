import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
train_dir = "train"
test_dir = "test"
image_size = (256, 128)
batch_size = 16
epochs = 6
learning_rate = 0.001


# Custom Dataset
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
num_classes = len(train_dataset.label_map)
print(f"Number of classes: {num_classes}")
print(f"Number of training images per class: {len(train_dataset) // num_classes}")
print(f"Number of testing images per class: {len(test_dataset) // num_classes}")


# Neural Network
class SimpleNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size[0] * image_size[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Model, Loss, Optimizer
model = SimpleNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Store accuracy and loss for visualization
train_accuracies = []
test_accuracies = []
epoch_losses = []


# Training Loop
def train_and_record_model():
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        # Evaluate accuracy after each epoch
        train_acc = evaluate_model(train_loader)
        test_acc = evaluate_model(test_loader)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")


# Evaluate Accuracy
def evaluate_model(dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# Run Training and Evaluation
print("Starting training...")
train_and_record_model()
print("Training complete.")

torch.save(model, "model.pth")

# Generate accuracy graph
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
plt.plot(epochs_range, test_accuracies, label="Testing Accuracy")
plt.title("Train and Test Accuracy Per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.show()
