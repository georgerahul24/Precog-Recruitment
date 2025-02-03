import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from collections import defaultdict

input_folder = 'test'


# Collect all unique characters from training data
def collect_characters(train_folder):
    chars = set()
    for filename in os.listdir(train_folder):
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
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}

        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                self.image_paths.append(os.path.join(folder, filename))
                self.labels.append(filename.split('_')[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return image, target


def collate_fn(batch):
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

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=512,  # This matches the CNN output channels
            hidden_size=256,
            bidirectional=True,
            num_layers=4,
            dropout=0.3,
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
        # Debug: print input shape
        # print("Input shape:", x.shape)

        # CNN Feature Extraction
        x = self.cnn(x)  # shape: (N, 512, H, W)
        # Debug: print CNN output shape
        # print("After CNN:", x.shape)

        # Prepare for LSTM: collapse height dimension while keeping width as sequence length
        x = x.permute(0, 3, 2, 1)  # (N, W, H, C)
        b, w, h, c = x.size()
        x = x.reshape(batch_size, w * h, c)  # (N, T, C) where T = W*H
        x = x.permute(1, 0, 2)  # (T, N, C)
        # Debug: print reshaped feature shape
        # print("After reshaping for LSTM:", x.shape)

        # Bidirectional LSTM
        x, _ = self.lstm(x)
        # Debug: print LSTM output shape
        # print("After LSTM:", x.shape)

        # Attention mechanism
        # Debug: print attention output shape
        # print("After Attention:", x.shape)

        # Final classifier
        x = self.fc(x)
        # Use log softmax for CTC loss
        x = nn.functional.log_softmax(x, dim=2)
        # Debug: print classifier output shape
        # print("After classifier:", x.shape)
        return x

# Decode predictions
def decode(output, idx_to_char):
    output = output.permute(1, 0, 2)  # (N, T, C)
    _, max_indices = torch.max(output, dim=2)

    decoded = []
    for batch in max_indices:
        chars = []
        prev = None
        for idx in batch:
            idx = idx.item()
            if idx != prev and idx != 0:
                chars.append(idx_to_char[idx])
            prev = idx
        decoded.append(''.join(chars))
    return decoded

# Configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

chars = collect_characters(input_folder)
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 is blank
idx_to_char = {i + 1: c for i, c in enumerate(chars)}
num_chars = len(chars)
# Transformations
transform = transforms.Compose([
    transforms.Resize((50, 150)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CaptchaDataset(input_folder, char_to_idx, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
model = CRNN(num_chars).to(device)
model.load_state_dict(torch.load('Models/GM1.pth', map_location=device))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss(blank=0)

model.eval()
with torch.no_grad():
    indices = np.random.choice(len(dataset), 2)
    samples = [dataset[i] for i in indices]
    images = torch.stack([s[0] for s in samples]).to(device)
    actuals = [''.join([idx_to_char[i.item()] for i in s[1]]) for s in samples]

    outputs = model(images)
    predictions = decode(outputs, idx_to_char)
    accuracy = (sum([1 for a, p in zip(actuals, predictions) if a == p]) / len(actuals)) * 100
    print("Model Accuracy: "+str(accuracy))
    while True:
        file_path = input("Enter the file path: ")
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        prediction = decode(output, idx_to_char)
        print(f"Predicted: {prediction[0]}")




