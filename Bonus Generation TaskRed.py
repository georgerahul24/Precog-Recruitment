import os
import random
import shutil
import string
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
output_dir = "train"
fonts_dir = "Fonts"
image_size = (256, 128)
font_size = 40
num_samples = 10000  # Total number of images to generate
max_length = 8  # Maximum length of the random string
noise_probability = 0.1
line_probability = 0.1
jitter_range = 2
num_required_fonts = 29

# Ensure output and fonts directories exist
shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fonts_dir, exist_ok=True)

# Expanded list of font families with cursive and variety
font_families = [
    "Roboto", "Smooch+Sans", "Lexend+Giga", "Inter", 'Lora', 'Quicksand', 'Fira+Sans',
    'Source+Code+Pro', 'Fjalla+One', 'Asap', 'Zilla+Slab', 'Cabin', 'Cormorant+Garamond', 'Crimson+Text',
    'Merriweather', 'Nunito', 'Open+Sans', 'Oswald', 'Poppins', 'Raleway', 'Roboto', 'Rubik', 'Rubik+Gemstones',
    'Ubuntu', 'Varela+Round', 'Barrio', 'Bangers', 'Atma', 'Henny+Penny', 'Joti+One'
]


# Function to download fonts
def download_fonts():
    print("Downloading fonts from Google Fonts...")
    base_url = "https://fonts.googleapis.com/css2?family="
    for font_family in tqdm(font_families[:num_required_fonts], desc="Downloading Fonts"):
        css_url = f"{base_url}{font_family}&display=swap"
        response = requests.get(css_url)
        if response.status_code != 200:
            print(f"Failed to fetch CSS for {font_family}")
            continue

        font_urls = [line.split("url(")[-1].split(")")[0].strip('\'"') for line in response.text.splitlines() if
                     "url(" in line]
        for url in font_urls:
            try:
                font_response = requests.get(url)
                font_response.raise_for_status()
                font_filename = os.path.join(fonts_dir, f"{font_family.replace('+', '_')}.ttf")
                with open(font_filename, "wb") as font_file:
                    font_file.write(font_response.content)
                break
            except Exception as e:
                print(f"Failed to download font from {url}: {e}")


# Ensure fonts are available
font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]
if len(font_files) < num_required_fonts:
    download_fonts()
    font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]

fonts = [ImageFont.truetype(font_path, font_size) for font_path in font_files]


# Generate a random string of lowercase letters and digits
def generate_random_string():
    length = random.randint(3, max_length)
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


# Draw random lines for noise
def draw_random_lines(draw, size):
    num_lines = random.randint(1, 5)
    for _ in range(num_lines):
        start_point = (random.randint(0, size[0]), random.randint(0, size[1]))
        end_point = (random.randint(0, size[0]), random.randint(0, size[1]))
        line_color = tuple(random.randint(0, 255) for _ in range(3))
        draw.line([start_point, end_point], fill=line_color, width=random.randint(1, 3))


# Generate image with text
def create_image(word, fonts, size):
    background_color = random.choice(["red", "green"])
    img = Image.new("RGB", size, color=background_color)
    draw = ImageDraw.Draw(img)
    draw_random_lines(draw, size)

    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < noise_probability:
                noise_color = tuple(random.randint(150, 250) for _ in range(3))
                draw.point((x, y), fill=noise_color)

    central_font = random.choice(fonts)
    total_width = sum(draw.textbbox((0, 0), letter, font=central_font)[2] for letter in word)
    total_height = draw.textbbox((0, 0), word[0], font=central_font)[3] - \
                   draw.textbbox((0, 0), word[0], font=central_font)[1]

    x = (size[0] - total_width) // 2
    y = (size[1] - total_height) // 2
    if background_color == "red":
        word = word[::-1]

    for letter in word:
        font = random.choice(fonts)
        letter_color = tuple(random.randint(0, 70) for _ in range(3))
        text_bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        jitter_x = random.randint(-jitter_range, jitter_range)
        jitter_y = random.randint(-jitter_range, jitter_range)

        draw.text((x + jitter_x, y + jitter_y), letter, font=font, fill=letter_color)
        x += text_width

    return img,background_color


# Generate dataset
def process_image(i):
    word = generate_random_string()
    img,bg_color = create_image(word, fonts, image_size)
    img_filename = os.path.join(output_dir, f"{word}_{i:05d}_{bg_color}.png")
    img.save(img_filename)


with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, range(num_samples)), desc="Generating Dataset", total=num_samples))

print(f"Dataset generated in '{output_dir}'")
