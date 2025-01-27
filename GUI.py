import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import ttk

# Configuration
image_size = (256, 128)

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Neural Network
class SimpleNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size[0] * image_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Load trained model
model_path = "model.pth"
num_classes = 10  # Set this to match your model's number of classes
model = SimpleNN(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Label map (update according to your classes)
label_map = {0: "Class A", 1: "Class B", 2: "Class C", 3: "Class D", 4: "Class E",
             5: "Class F", 6: "Class G", 7: "Class H", 8: "Class I", 9: "Class J"}

def predict_image(image_path):
    """Load an image, preprocess it, and return the model's prediction."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return label_map[predicted.item()]

def browse_image():
    """Open file dialog to select an image and display the prediction."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        prediction = predict_image(file_path)
        result_label.config(text=f"Prediction: {prediction}")

# GUI Setup
root = tk.Tk()
root.title("Image Classification")
root.geometry("400x200")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Widgets
browse_button = ttk.Button(frame, text="Browse Image", command=browse_image)
browse_button.grid(row=0, column=0, pady=10)

result_label = Label(frame, text="Prediction: ", font=("Arial", 14))
result_label.grid(row=1, column=0, pady=20)

# Start GUI
root.mainloop()
