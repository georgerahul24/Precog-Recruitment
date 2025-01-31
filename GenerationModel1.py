import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# Training loop
num_epochs = 1000
# Collect all unique characters from training data
def collect_characters(train_folder):
    chars = set()
    for filename in os.listdir(train_folder):
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

# CNN + LSTM Model
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.num_chars = num_chars

        # Enhanced CNN with residual connections
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

        # Enhanced LSTM with depth-wise processing
        self.lstm = nn.LSTM(
            input_size=512,  # Increased to match CNN output
            hidden_size=256,
            bidirectional=True,
            num_layers=3,  # Increased from 2
            dropout=0.3,
            batch_first=False
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # 256*2 for bidirectional
            num_heads=8,
            dropout=0.2
        )

        # Final classifier with layer normalization
        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_chars + 1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN Feature Extraction
        x = self.cnn(x)  # (N, 512, H, W)

        # Prepare for LSTM
        x = x.permute(0, 3, 2, 1)  # (N, W, H, C)
        x = x.reshape(batch_size, -1, 512)  # (N, W*H, C)
        x = x.permute(1, 0, 2)  # (T, N, C)

        # Bidirectional LSTM
        x, _ = self.lstm(x)

        # Attention
        x, _ = self.attention(x, x, x)

        # Classifier
        x = self.fc(x)
        x = nn.functional.log_softmax(x, dim=2)

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
train_folder = 'train'
chars = collect_characters(train_folder)
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 is blank
idx_to_char = {i + 1: c for i, c in enumerate(chars)}
num_chars = len(chars)

# Transformations
transform = transforms.Compose([
    transforms.Resize((50, 150)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CaptchaDataset(train_folder, char_to_idx, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Model setup
model = CRNN(num_chars).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss(blank=0)

if os.path.exists('GenerationModel'):
    os.system('rm -rf GenerationModel')
os.makedirs('GenerationModel', exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

    for images, targets, target_lengths in loop:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        output = model(images)

        T, N = output.size(0), output.size(1)
        input_lengths = torch.full((N,), T, dtype=torch.long, device='cpu')  # Create on CPU

        # Move tensors to CPU for CTC loss
        loss = criterion(
            output.cpu().float(),  # Add float() for MPS->CPU compatibility
            targets.cpu(),
            input_lengths,
            target_lengths.cpu()
        )

        loss.backward()
        optimizer.step()

        # Print sample predictions after each epoch
    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(dataset), 2)
        samples = [dataset[i] for i in indices]
        images = torch.stack([s[0] for s in samples]).to(device)
        actuals = [''.join([idx_to_char[i.item()] for i in s[1]]) for s in samples]

        outputs = model(images)
        predictions = decode(outputs, idx_to_char)

        print(f"\nSample Predictions after Epoch {epoch + 1}:")
        print(f"Actual: {actuals[0]} → Predicted: {predictions[0]}")
        print(f"Actual: {actuals[1]} → Predicted: {predictions[1]}\n")

# Save the model
accuracy = (sum([1 for a, p in zip(actuals, predictions) if a == p]) / len(actuals)) * 100
model_path = f"GenerationModel/model_epoch_{epoch + 1}_acc_{accuracy:.2f}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")