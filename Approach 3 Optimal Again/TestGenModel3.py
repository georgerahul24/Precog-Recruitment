#!/usr/bin/env python3
""" For Bonus Generation 2.py model i.e. the non optimised one"""
import os
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

input_folder = 'test'  # Folder containing test images
model_path = '../Models/GM6.pth'  # Change to your desired model checkpoint file


# ---------------------------
# Helper Functions & Classes
# ---------------------------
def collect_characters(folder):
    chars = set()
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            label = filename.split('_')[0]
            chars.update(label)
    return sorted(chars)


class CaptchaDataset(Dataset):
    def __init__(self, folder, char_to_idx, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.backgrounds = []  # Store background color (red/green)
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}

        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                self.image_paths.append(os.path.join(folder, filename))
                label, _, background = filename.split('_')  # Extract label and background
                self.labels.append(label)
                self.backgrounds.append(background.split('.')[0])  # Remove .png

    def __len__(self):
        return len(self.image_paths)

    def print_summary(self):
        print(f"Loaded {len(self.image_paths)} images from {self.folder}")

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        background = self.backgrounds[idx]  # Get background color

        if self.transform:
            image = self.transform(image)

        # Convert label characters to indices
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        # Convert background to binary: 1 for red, 0 for green
        background_label = 1 if background == 'red' else 0
        return image, target, background_label


def collate_fn(batch):
    """
    Expects a list of (image, target, background_label) tuples.
    Returns:
        images: (B, C, H, W) tensor,
        targets: concatenated target tensor,
        target_lengths: tensor of each target length,
        background_labels: tensor of background labels.
    """
    images = []
    targets = []
    target_lengths = []
    background_labels = []

    for img, tgt, bg in batch:
        images.append(img)
        targets.append(tgt)
        target_lengths.append(len(tgt))
        background_labels.append(bg)

    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    background_labels = torch.tensor(background_labels, dtype=torch.float32)

    return images, targets, target_lengths, background_labels


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class GenLSTM(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.num_chars = num_chars

        self.cnn = nn.Sequential(
            SeparableConv2d(3, 32, kernel_size=3, padding=1),
            SeparableConv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            SeparableConv2d(32, 64, kernel_size=3, padding=1),
            SeparableConv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            SeparableConv2d(64, 128, kernel_size=3, padding=1),
            SeparableConv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.3),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            dropout=0.3,
            batch_first=False
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_chars + 1)  # +1 for CTC blank token
        )

        self.bg_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)

        bg_pred = self.bg_classifier(x)  # Probability of red background

        x = x.permute(0, 3, 2, 1)  # (N, W, H, C)
        b, w, h, c = x.size()
        x = x.reshape(batch_size, w * h, c)
        x = x.permute(1, 0, 2)

        x, _ = self.lstm(x)

        # Reverse sequences if background is red
        for i in range(batch_size):
            if bg_pred[i] > 0.5:  # Red background
                x[:, i, :] = x.flip(0)[:, i, :]

        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x, bg_pred


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


def char_level_accuracy(actual, predicted):
    """
    Computes the character-level accuracy between two strings.
    """
    if max(len(actual), len(predicted)) == 0:
        return 0.0
    match_count = sum(1 for ac, pr in zip(actual, predicted) if ac == pr)
    return match_count / max(len(actual), len(predicted))


def evaluate_model(model, dataloader, idx_to_char, device):
    """
    Evaluates the model on the provided dataloader.
    Returns:
        A tuple of:
          - predictions: List of predicted strings.
          - actuals: List of ground-truth strings.
          - bg_predictions: List of predicted background labels.
          - bg_actuals: List of ground-truth background labels.
          - overall_accuracy: Overall text accuracy (%).
          - acc_90: Percentage of samples with >= 90% char-level accuracy.
          - acc_95: Percentage of samples with >= 95% char-level accuracy.
    """
    model.eval()
    all_predictions = []
    all_actuals = []
    all_bg_predictions = []
    all_bg_actuals = []

    with torch.no_grad():
        for images, targets, target_lengths, bg_labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs, bg_preds = model(images)
            predictions = decode(outputs, idx_to_char)
            # Reconstruct ground truth texts from targets
            actuals = []
            start = 0
            for length in target_lengths:
                length = length.item()
                target_seq = targets[start: start + length]
                text = ''.join([idx_to_char.get(i.item(), '') for i in target_seq])
                actuals.append(text)
                start += length
            # Convert background predictions to binary values
            bg_predictions = (bg_preds > 0.5).int().squeeze().tolist()
            if isinstance(bg_predictions, int):
                bg_predictions = [bg_predictions]
            bg_actuals = bg_labels.int().tolist()

            all_predictions.extend(predictions)
            all_actuals.extend(actuals)
            all_bg_predictions.extend(bg_predictions)
            all_bg_actuals.extend(bg_actuals)

    # Overall text accuracy (exact match)
    correct = sum([p == a for p, a in zip(all_predictions, all_actuals)])
    overall_accuracy = (correct / len(all_actuals)) * 100 if all_actuals else 0

    # Compute character-level accuracy for each sample
    acc_90_count = sum(1 for a, p in zip(all_actuals, all_predictions) if char_level_accuracy(a, p) >= 0.90)
    acc_95_count = sum(1 for a, p in zip(all_actuals, all_predictions) if char_level_accuracy(a, p) >= 0.95)
    acc_60_count = sum(1 for a, p in zip(all_actuals, all_predictions) if char_level_accuracy(a, p) >= 0.60)
    acc_90 = (acc_90_count / len(all_actuals)) * 100 if all_actuals else 0
    acc_95 = (acc_95_count / len(all_actuals)) * 100 if all_actuals else 0
    acc_60 = (acc_60_count / len(all_actuals)) * 100 if all_actuals else 0
    return all_predictions, all_actuals, all_bg_predictions, all_bg_actuals, overall_accuracy, acc_90, acc_95, acc_60


# ---------------------------
# Main Testing Script
# ---------------------------
if __name__ == '__main__':
    # Explicit variables for input folder and model checkpoint path

    print(f"Loading model checkpoint from: {model_path}")

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using device:", device)

    # For character collection, we assume training characters are the same as test characters.
    # Adjust the folder if needed.
    train_folder = 'train'
    chars = collect_characters(train_folder)
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # Reserve index 0 for CTC blank token
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    num_chars = len(chars)

    # Transformations for the images
    transform = transforms.Compose([
        transforms.Resize((50, 150)),
        transforms.ToTensor(),
    ])

    # Prepare the test dataset and dataloader
    test_dataset = CaptchaDataset(input_folder, char_to_idx, transform)
    test_dataset.print_summary()
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize the model and load the checkpoint
    model = GenLSTM(num_chars).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Evaluate the model on the test dataset
    predictions, actuals, bg_predictions, bg_actuals, overall_accuracy, acc_90, acc_95, acc_60 = evaluate_model(
        model, test_dataloader, idx_to_char, device
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Overall Text Accuracy: {overall_accuracy:.2f}%")
    print(f"60%+ Character-Level Accuracy: {acc_60:.2f}%\n")
    print(f"90%+ Character-Level Accuracy: {acc_90:.2f}%")
    print(f"95%+ Character-Level Accuracy: {acc_95:.2f}%\n")

    # Print a few sample predictions
    print("Sample Predictions:")
    for i in range(min(6, len(predictions))):
        pred_text = predictions[i]
        true_text = actuals[i]
        pred_bg = "Red" if bg_predictions[i] == 1 else "Green"
        true_bg = "Red" if bg_actuals[i] == 1 else "Green"
        print(f"  Prediction: {pred_text} | Actual: {true_text} | Predicted BG: {pred_bg} | Actual BG: {true_bg}")
