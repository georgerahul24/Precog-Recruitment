import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
output_dir = "test"
font_path = "./Fonts/roboto.ttf"
image_size = (256, 128)
font_size = 32
num_samples_per_class = 100
words_list = "Rudra Choudhary Aditya Peketi Vardhan George Rahul Sri Rama Venkata Dinakar Venapati Aashrith Reddy Hrishikesh Milind Gawas Akshath Puneeth Gupta".split()
# words_list = []
# # Load additional words from the dictionary file
# with open("Dictionary.txt", "r") as file:
#     words_list.extend(eval(file.read()))

print("Number of words found in the Dictionary:", len(words_list))

# Create the output directory if it doesn't exist
# Remove the output directory if it exists and create a new one
os.system(f"rm -rf {output_dir}")
os.makedirs(output_dir, exist_ok=True)

def create_image(word, font, size):
    """Create an image with the specified word and font."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, word.capitalize(), font=font, fill="black")

    return img

# Load the font
font = ImageFont.truetype(font_path, font_size)

# Generate dataset
for word in tqdm(words_list, desc="Generating Dataset"):
    # Create a folder for the word if it doesn't exist
    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)

    for i in range(num_samples_per_class):
        # Create and save the image
        img = create_image(word, font, image_size)
        img_filename = os.path.join(word_dir, f"img_{i:04d}.png")
        img.save(img_filename)

print(f"Dataset generated in '{output_dir}'")
