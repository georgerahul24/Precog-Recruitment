import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Configuration
train_dir = "letter_train"
test_dir = "letter_test"
image_size = (128, 64)
batch_size = 64
epochs = 50
learning_rate = 0.0003

# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Dataset Class
class WordImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}

        # Create mapping of word to index
        for idx, word in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, word)
            if os.path.isdir(folder_path):
                self.class_to_idx[word] = idx
                for file in os.listdir(folder_path):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(os.path.join(folder_path, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Improved Model Architecture
class LetterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LetterClassifier, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# Load datasets
train_dataset = WordImageDataset(train_dir, transform=transform_train)
test_dataset = WordImageDataset(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.class_to_idx)
print(f'Number of classes: {num_classes}')
print(f'Number of training images: {len(train_dataset) // num_classes}')
print(f'Number of testing images: {len(test_dataset) // num_classes}')

# Initialize model, loss, and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = LetterClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# Training function
def train_model():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy


# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    return test_accuracy


# Training loop with early stopping
best_accuracy = 0.0
patience = 5
epochs_without_improvement = 0
train_accuracies = []
test_accuracies = []

start_time = time.time()

for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    train_loss, train_accuracy = train_model()
    test_accuracy = evaluate_model()

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Learning rate scheduling
    scheduler.step()

    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'letter_model.pth')
    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= patience:
        print("Early stopping!")
        break

    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    print('-' * 50)

total_time = time.time() - start_time
print(f"Training complete! Total time: {total_time:.2f}s")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()


# Function to predict word for a single image
def predict_word(image_path):
    model.load_state_dict(torch.load('best_word_classifier.pth'))
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)

    # Convert index back to word
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    predicted_word = idx_to_class[predicted.item()]

    return predicted_word

# Example usage
# predicted_word = predict_word("path_to_image.png")
# print(f'Predicted Word: {predicted_word}')
