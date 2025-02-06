import os
import random
import shutil
import string
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
output_dir = "test"
font_path = "Fonts/Roboto.ttf"
image_size = (256, 128)
font_size = 32
num_samples_per_class = 100
min_word_length = 3
max_word_length = 11

# Create the output directory if it doesn't exist
# Remove the output directory if it exists and create a new one
if(os.path.exists(output_dir)):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

def create_image(word, font, size):
    """Create an image with the specified word and font."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, word, font=font, fill="black")

    return img

def generate_random_word(length):
    """Generate a random word of the given length."""
    return ''.join(random.choices(string.ascii_lowercase+string.digits+string.ascii_uppercase, k=length))

# Load the font
font = ImageFont.truetype(font_path, font_size)

# Generate dataset
for length in range(min_word_length, max_word_length + 1):
    for _ in tqdm(range(num_samples_per_class), desc=f"Generating {length}-letter words"):
        word = generate_random_word(length)
        # Create a folder for the word length if it doesn't exist
        length_dir = os.path.join(output_dir, str(length))
        os.makedirs(length_dir, exist_ok=True)

        # Create and save the image
        img = create_image(word, font, image_size)
        img_filename = os.path.join(length_dir, f"{word}.png")
        img.save(img_filename)

print(f"Dataset generated in '{output_dir}'")