import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
output_dir = "output"
font_path = "./Fonts/roboto.ttf"
image_size = (256, 128)
font_size = 32
num_samples = 10
words_list= []
with open("Dictionary.txt", "r") as file:
    words_list.extend(eval(file.read()))
print("Number of words found in the Dictionary: ",len(words_list))
# Source for the basic logic https://cloudinary.com/guides/image-effects/a-guide-to-adding-text-to-images-with-python
os.makedirs(output_dir, exist_ok=True)

def create_image(word, font, size):
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, word.capitalize(), font=font, fill="black")

    return img

font = ImageFont.truetype(font_path, font_size)

labels = []
for i in tqdm(range(num_samples), desc="Generating Dataset"):
    word = words_list[i % len(words_list)]
    img = create_image(word, font, image_size)
    img_filename = os.path.join(output_dir, f"img_{i:04d}.png")
    img.save(img_filename)
    labels.append(f"{img_filename},{word}\n")

labels_path = os.path.join(output_dir, "labels.csv")
with open(labels_path, "w") as f:
    f.writelines(labels)

print(f"Dataset generated in '{output_dir}'")
