import os
import time

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Training loop parameters
num_epochs = 1000
train_folder = 'train'
test_folder = 'test'


def collect_characters(folder):
    chars = set()
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            label = filename.split('_')[0]
            chars.update(label)
    return sorted(chars)


# Custom Dataset
class CaptchaDataset(Dataset):
    def __init__(self, folder, char_to_idx, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}

        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                self.image_paths.append(os.path.join(folder, filename))
                self.labels.append(filename.split('_')[0])

    def __len__(self):
        return len(self.image_paths)

    def print_summary(self):
        print(f"Loaded {len(self.image_paths)} images from {self.folder}")

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label characters to indices
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return image, target


def collate_fn(batch):
    """
    Expects a list of (image, target) tuples.
    Returns:
        images: (B, C, H, W) tensor,
        targets: concatenated target tensor,
        target_lengths: tensor of each target length.
    """
    images = []
    targets = []
    target_lengths = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
        target_lengths.append(len(tgt))

    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths


import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.num_chars = num_chars

        # Enhanced CNN with dropout and batch normalization
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # Maintain width for sequence
            nn.Dropout2d(0.3),

            # Additional block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.4),
        )

        # RNN for sequence modeling (replacing LSTM with vanilla RNN)
        self.rnn = nn.RNN(
            input_size=512,     # This matches the CNN output channels
            hidden_size=256,
            num_layers=4,
            dropout=0.3,
            bidirectional=True,
            batch_first=False
        )

        # Final classifier with layer normalization
        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_chars + 1)  # +1 for the blank token used in CTC
        )

    def forward(self, x):
        batch_size = x.size(0)
        # CNN Feature Extraction
        x = self.cnn(x)  # shape: (N, 512, H, W)
        # Prepare for RNN: collapse height dimension while keeping width as sequence length
        x = x.permute(0, 3, 2, 1)  # (N, W, H, C)
        b, w, h, c = x.size()
        x = x.reshape(batch_size, w * h, c)  # (N, T, C) where T = W*H
        x = x.permute(1, 0, 2)  # (T, N, C)

        # Bidirectional RNN
        # Note: The vanilla RNN returns (output, hidden_state)
        x, _ = self.rnn(x)

        # Final classifier
        x = self.fc(x)
        # Use log softmax for CTC loss (if applicable)
        x = F.log_softmax(x, dim=2)
        return x


def decode(output, idx_to_char):
    """
    Decodes the network output into text.
    Args:
        output: Tensor of shape (T, N, C) (logits)
        idx_to_char: Dictionary mapping indices to characters.
    Returns:
        List of decoded strings.
    """
    output = output.permute(1, 0, 2)  # (N, T, C)
    _, max_indices = torch.max(output, dim=2)  # (N, T)
    decoded = []
    for indices in max_indices:
        chars = []
        prev = None
        for idx in indices:
            idx = idx.item()
            # Skip duplicate predictions and the blank (0)
            if idx != prev and idx != 0:
                chars.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars))
    return decoded


def evaluate_model(model, dataloader, idx_to_char, device, desc='Evaluating'):
    """
    Evaluate model on the given dataloader.
    Returns:
        predictions: List of predicted strings.
        actuals: List of actual strings.
        accuracy: Overall accuracy in percentage.
    """
    model.eval()
    all_predictions = []
    all_actuals = []

    # Loop over the dataloader with tqdm
    for images, targets, target_lengths in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)

        predictions = decode(outputs, idx_to_char)

        # Reconstruct the ground truth labels from the concatenated targets.
        actuals = []
        start = 0
        for length in target_lengths:
            length = length.item()
            target_seq = targets[start: start + length]
            text = ''.join([idx_to_char.get(i.item(), '') for i in target_seq])
            actuals.append(text)
            start += length

        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

    # Calculate accuracy
    correct = sum([p == a for p, a in zip(all_predictions, all_actuals)])
    accuracy = (correct / len(all_actuals)) * 100 if all_actuals else 0
    return all_predictions, all_actuals, accuracy


def save_model(model, epoch, test_accuracy, base_path="GenerationModelRNN"):
    """
    Save model with metrics in filename.
    """
    model_path = f"{base_path}/model_epoch_{epoch + 1}_test_acc_{test_accuracy:.5f}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


# --------------------------
# Main Script Starts Here
# --------------------------

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

# Collect characters and prepare mappings
chars = collect_characters(train_folder)
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # Reserve 0 for blank
idx_to_char = {i + 1: c for i, c in enumerate(chars)}
num_chars = len(chars)

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((50, 150)),
    transforms.ToTensor(),
])

# Prepare datasets and dataloaders
train_dataset = CaptchaDataset(train_folder, char_to_idx, transform)
train_dataset.print_summary()
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

test_dataset = CaptchaDataset(test_folder, char_to_idx, transform)
test_dataset.print_summary()
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize the model, optimizer, and loss criterion (CTC Loss)
model = CRNN(num_chars).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss(blank=0)

# Create directory for saving models
if os.path.exists('GenerationModelRNN'):
    os.system('rm -rf GenerationModelRNN')
os.makedirs('GenerationModelRNN', exist_ok=True)

# --------------------------
# Training and Evaluation Loop
# --------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Training loop with progress bar
    train_loop = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for images, targets, target_lengths in train_loop:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        output = model(images)  # output shape: (T, N, C)

        T, N = output.size(0), output.size(1)
        # Create input lengths: all sequences have length T
        input_lengths = torch.full((N,), T, dtype=torch.long)

        # For CTC loss, move tensors to CPU (helps with MPS backend)
        loss = criterion(
            output.cpu().float(),  # cast to float if necessary
            targets.cpu(),
            input_lengths,
            target_lengths.cpu()
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

    # --------------------------
    # Evaluation on Test Data
    # --------------------------
    test_predictions, test_actuals, test_accuracy = evaluate_model(
        model, test_dataloader, idx_to_char, device, desc='Testing'
    )
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Print a couple of sample predictions for debugging
    print("Sample Predictions:")
    for i in range(min(2, len(test_predictions))):
        print(f"  Prediction: {test_predictions[i]} | Actual: {test_actuals[i]}")
    with open('log1.txt', 'a') as f:
        f.write(f" {time.time()} Epoch [{epoch + 1}/{num_epochs}] - Test Accuracy: {test_accuracy:.5f}%\n")
        f.write("Sample Predictions:\n")
        for i in range(min(2, len(test_predictions))):
            f.write(f"  Prediction: {test_predictions[i]} | Actual: {test_actuals[i]}\n")
        f.close()

    # Save model checkpoint
    model_path = save_model(model, epoch, test_accuracy)
    print(f"\nModel saved to {model_path}\n")
    print()
