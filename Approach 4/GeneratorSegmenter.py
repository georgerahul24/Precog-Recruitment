import os
import random
import shutil
import string
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
output_dir = "trainSeg"  # Flat folder with all images
font_path = "../Fonts/Roboto.ttf"  # Path to a TTF font file
image_size = (256, 128)
font_size = 32
num_samples_per_length = 6000
min_word_length = 3
max_word_length = 10  # Maximum is 10 to match the model's MAX_LETTERS

# If the output directory exists, remove it; then create a new one
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

def create_image(word, font, size):
    """Create an image with the specified word and font centered on a white background."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    # Get the bounding box of the text to center it
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, word, font=font, fill="black")
    return img

def generate_random_word(length):
    """Generate a random word of the given length using lowercase, uppercase, and digits."""
    all_chars = string.ascii_lowercase + string.digits + string.ascii_uppercase
    return ''.join(random.choices(all_chars, k=length))

# Load the font
font = ImageFont.truetype(font_path, font_size)

# Generate dataset with a flat folder structure
for length in range(min_word_length, max_word_length + 1):
    for _ in tqdm(range(num_samples_per_length), desc=f"Generating {length}-letter words"):
        word = generate_random_word(length)
        # Append a random 4-digit number for uniqueness in filename
        rand_suffix = random.randint(1000, 9999)
        filename = f"{word}_{rand_suffix}.png"
        filepath = os.path.join(output_dir, filename)
        img = create_image(word, font, image_size)
        img.save(filepath)

print(f"Dataset generated in '{output_dir}'")
