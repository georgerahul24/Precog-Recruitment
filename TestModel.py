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
batch_size = 32
epochs = 20
learning_rate = 0.001

# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) took the values form image net
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


class WordClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WordClassifier, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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

#num_classes = len(train_dataset.class_to_idx)
num_classes = len(train_dataset.class_to_idx)
print(f'Number of classes: {num_classes}')
print(f'Number of training images per class: {len(train_dataset)//num_classes}')
print(f'Number of testing images per class: {len(test_dataset)//num_classes}')

# Initialize model, loss, and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = WordClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


from tqdm import tqdm
import time

def train_model():
    best_accuracy = 0.0
    train_accuracies = []
    test_accuracies = []

    total_epochs = epochs  # Total number of epochs
    start_time = time.time()  # Record the start time

    for epoch in range(total_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Initialize tqdm progress bar for the training loop
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for images, labels in pbar:
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

                # Update tqdm progress bar with relevant information
                pbar.set_postfix({
                    'Train Loss': running_loss / (pbar.n + 1),
                    'Train Accuracy': 100. * correct / total
                })

        # Calculate train accuracy
        train_accuracy = 100. * correct / total
        train_accuracies.append(train_accuracy)

        # Evaluation phase
        test_accuracy = evaluate_model()
        test_accuracies.append(test_accuracy)

        # Learning rate scheduling
        scheduler.step(100 - test_accuracy)  # Use accuracy as metric

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_word_classifier.pth')

        # Calculate time elapsed and remaining time
        elapsed_time = time.time() - start_time
        time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = time_per_epoch * (total_epochs - (epoch + 1))

        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(f'Time Elapsed: {elapsed_time:.2f}s | Time Remaining: {remaining_time:.2f}s')
        print('-' * 50)

    total_time = time.time() - start_time
    print(f"Training complete! Total time: {total_time:.2f}s")
    return train_accuracies, test_accuracies



def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


# Train the model
print("Starting training...")
train_accuracies, test_accuracies = train_model()
print("Training complete!")

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