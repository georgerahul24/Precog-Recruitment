import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import torch.nn as nn
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

# Load the trained model
model = torch.load("model.pth")

# Automatically generate the label map from the training directory structure
train_dir = "train"  # Replace with the path to your training data
label_map = {idx: folder for idx, folder in enumerate(sorted(os.listdir(train_dir))) if os.path.isdir(os.path.join(train_dir, folder))}

# Configuration for image size
image_size = (256, 128)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# GUI Application class
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("400x400")

        self.label = tk.Label(root, text="Drag and Drop an Image to Classify", font=("Arial", 16))
        self.label.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        # Canvas to display image
        self.canvas = tk.Canvas(root, width=256, height=128)
        self.canvas.pack()

        # Open file dialog to select image
        self.canvas.bind("<ButtonRelease-1>", self.open_image)

    def open_image(self, event):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        # Open image and apply transformations
        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict using the model
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Get class name from label_map
        predicted_class = label_map.get(predicted.item(), "Unknown")
        self.display_image(file_path, predicted_class)

    def display_image(self, file_path, predicted_class):
        # Display image on canvas
        image = Image.open(file_path)
        image.thumbnail((256, 128))
        img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(128, 64, image=img_tk)

        # Update result label
        self.result_label.config(text=f"Predicted: {predicted_class}")


# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
