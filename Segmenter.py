import os
import glob
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # progress bar

trainFolder = "trainSeg"
testFolder = "testSeg"
OUTPUT_FOLDER = "SegmentationModel"
VIZ_Folder = "epoch_vis"
MAX_EPOCHS = 200

# -------------------------------
# Global constants and char mapping
# -------------------------------

IMG_HEIGHT = 128
IMG_WIDTH = 256
IMG_CHANNELS = 3
MAX_LETTERS = 10
# We reserve index 0 as "blank". Then digits (0-9) and letters (A-Z):
NUM_CLASSES = 1 + 10 + 26  # = 37

# Define our character set:
blank_token = ''
digits = list("0123456789")
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
characters = [blank_token] + digits + letters
char_to_idx = {c: i for i, c in enumerate(characters)}


# -------------------------------
# Dataset Definition
# -------------------------------

class CaptchaDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = glob.glob(os.path.join(folder, "*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        basename = os.path.basename(file_path)
        label_text = basename.split("_")[0].upper()
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Image {file_path} could not be loaded.")

        # Resize and convert image
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb.transpose(2, 0, 1) / 255.0, dtype=torch.float32)

        # OpenCV-based segmentation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_centers = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 10:  # Filter small noise
                x_centers.append(x + w // 2)

        x_centers.sort()
        L = min(len(label_text), MAX_LETTERS)

        # Handle mismatch between detected characters and label length
        if len(x_centers) > L:
            x_centers = x_centers[:L]
        elif len(x_centers) < L:
            x_centers += [0] * (L - len(x_centers))  # Pad with zeros

        # Create segmentation ground truth
        seg_gt = np.full(MAX_LETTERS, -1.0, dtype=np.float32)
        for i in range(L):
            seg_gt[i] = x_centers[i] / IMG_WIDTH  # Normalized position
        seg_gt = torch.tensor(seg_gt, dtype=torch.float32)

        # Create classification ground truth
        class_gt = np.zeros(MAX_LETTERS, dtype=np.int64)
        for i in range(MAX_LETTERS):
            if i < len(label_text):
                class_gt[i] = char_to_idx.get(label_text[i], 0)
            else:
                class_gt[i] = 0
        class_gt = torch.tensor(class_gt, dtype=torch.long)

        return img_tensor, seg_gt, class_gt


def masked_mse_loss(pred, target):
    loss = (pred - target) ** 2
    loss = loss
    return loss.sum()


def visualize_segmentation(model, dataset, epoch, device):
    model.eval()
    num_samples = min(5, len(dataset))

    for i in range(num_samples):
        img, seg_gt, _ = dataset[i]
        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            seg_pred, _ = model(img_input)

        seg_pred = seg_pred.cpu().numpy()[0]
        img_np = img.numpy().transpose(1, 2, 0)

        # Get ground truth boundaries
        gt_mask = seg_gt.numpy() != -1
        gt_x = seg_gt.numpy()[gt_mask] * IMG_WIDTH

        # Get predicted boundaries
        pred_x = seg_pred * IMG_WIDTH

        # Create visualization
        plt.figure(figsize=(15, 6))

        # Ground truth plot
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        for x in gt_x:
            plt.axvline(x, color='lime', linestyle='--', linewidth=2, alpha=0.8, label='OpenCV GT')
        plt.title(f"OpenCV Segmentation ({len(gt_x)} chars)")
        plt.legend()

        # Model prediction plot
        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        for x in pred_x:
            plt.axvline(x, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Model Prediction')
        plt.title(f"Model Prediction ({len(pred_x)} positions)")
        plt.legend()

        plt.suptitle(f"Epoch {epoch + 1} - Sample {i + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_Folder, f"epoch_{epoch + 1}_sample_{i}.png"))
        plt.close()

    model.train()


# -------------------------------
# Model Definition
# -------------------------------

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        feat_h = IMG_HEIGHT // 8
        feat_w = IMG_WIDTH // 8
        self.flatten_dim = 128 * feat_h * feat_w

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU()
        )
        self.fc_seg = nn.Sequential(
            nn.Linear(512, MAX_LETTERS),
            nn.Sigmoid()
        )
        self.fc_class = nn.Linear(512, MAX_LETTERS * NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        seg_out = self.fc_seg(x)
        class_out = self.fc_class(x)
        class_out = class_out.view(-1, MAX_LETTERS, NUM_CLASSES)
        return seg_out, class_out


criterion_class = nn.CrossEntropyLoss(ignore_index=0)


# -------------------------------
# Training and Testing Loops
# -------------------------------


def train_model():
    num_epochs = MAX_EPOCHS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CaptchaDataset(trainFolder)
    test_dataset = CaptchaDataset(testFolder)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = CaptchaModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, seg_gt, class_gt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"):
            imgs = imgs.to(device)
            seg_gt = seg_gt.to(device)
            class_gt = class_gt.to(device)

            optimizer.zero_grad()
            seg_pred, class_pred = model(imgs)

            loss_seg = masked_mse_loss(seg_pred, seg_gt)
            class_pred_flat = class_pred.view(-1, NUM_CLASSES)
            class_gt_flat = class_gt.view(-1)
            loss_class = criterion_class(class_pred_flat, class_gt_flat)
            loss = loss_seg + loss_class

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Compute training accuracy
        train_correct = 0
        train_total = 0
        model.eval()
        with torch.no_grad():
            for imgs, _, class_gt in train_loader:
                imgs = imgs.to(device)
                class_gt = class_gt.to(device)
                _, class_pred = model(imgs)
                class_pred_flat = class_pred.view(-1, NUM_CLASSES)
                class_gt_flat = class_gt.view(-1)
                preds = torch.argmax(class_pred_flat, dim=1)
                mask = (class_gt_flat != 0)
                train_correct += (preds[mask] == class_gt_flat[mask]).sum().item()
                train_total += mask.sum().item()
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss} , Training Accuracy: {train_accuracy*100} %")

        # Compute testing loss and accuracy
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, seg_gt, class_gt in tqdm(test_loader, desc="Epoch Testing", leave=False):
                imgs = imgs.to(device)
                seg_gt = seg_gt.to(device)
                class_gt = class_gt.to(device)
                seg_pred, class_pred = model(imgs)

                loss_seg = masked_mse_loss(seg_pred, seg_gt)
                class_pred_flat = class_pred.view(-1, NUM_CLASSES)
                class_gt_flat = class_gt.view(-1)
                loss_class = criterion_class(class_pred_flat, class_gt_flat)
                loss = loss_seg + loss_class
                test_running_loss += loss.item() * imgs.size(0)

                preds = torch.argmax(class_pred_flat, dim=1)
                mask = (class_gt_flat != 0)
                test_correct += (preds[mask] == class_gt_flat[mask]).sum().item()
                test_total += mask.sum().item()

        test_loss = test_running_loss / len(test_dataset)
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100} %")

        visualize_segmentation(model, test_dataset, epoch, device)

        # Save model checkpoint for this epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, f"segmentation_model_{epoch}_{test_accuracy}.pth"))


if __name__ == "__main__":

    if (os.path.exists(OUTPUT_FOLDER)):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if (os.path.exists(VIZ_Folder)):
        shutil.rmtree(VIZ_Folder)
    os.makedirs(VIZ_Folder, exist_ok=True)
    train_model()