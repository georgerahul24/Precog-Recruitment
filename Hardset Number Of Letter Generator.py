import os
import random
import string
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
output_dir = "train"
fonts_dir = "Fonts"
image_size = (256, 128)
font_size = 32
num_samples_per_length = 300
num_required_fonts = 42
min_length = 1
max_length = 10
noise_probability = 0.1

# Ensure output and fonts directories exist
if os.path.exists(output_dir):
    os.system(f"rm -rf {output_dir}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fonts_dir, exist_ok=True)

# Font families list
font_families = [
    "Roboto", "Open+Sans", "Lato", "Montserrat", "Poppins", "Raleway", "Oswald",
    "Merriweather", "Nunito", "Ubuntu", "Playfair+Display", "Noto+Sans", "Noto+Serif",
    "Titillium+Web", "Source+Sans+Pro", "PT+Sans", "PT+Serif", "Bitter", "Arimo",
    "Cabin", "Dosis", "Quicksand", "Work+Sans", "Rubik", "Varela+Round", "Dancing+Script",
    "Pacifico", "Great+Vibes", "Courgette", "Allura", "Handlee", "Lobster", "Satisfy",
    "Cookie", "Tangerine", "Sacramento", "Parisienne", "Yellowtail", "Kaushan+Script",
    "Shadows+Into+Light", "Amatic+SC", "Indie+Flower"
]


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


# Check and load fonts
font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]
if len(font_files) < num_required_fonts:
    download_fonts()
    font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]

print(f"Number of fonts available: {len(font_files)}")
fonts = [ImageFont.truetype(font_path, font_size) for font_path in font_files]


def generate_random_word(length):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


def randomize_case(word):
    return ''.join(random.choice([char.upper(), char.lower()]) for char in word)


def create_image(word, font, size):
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < noise_probability:
                noise_color = tuple(random.randint(0, 255) for _ in range(3))
                draw.point((x, y), fill=noise_color)

    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    random_color = tuple(random.randint(0, 255) for _ in range(3))
    draw.text(position, word, font=font, fill=random_color)

    return img


# Main execution loop - generate datasets sequentially
for length in range(min_length, max_length + 1):
    print(f"\nGenerating dataset for length {length}")

    # Create directory for this length
    length_dir = os.path.join(output_dir, str(length))
    os.makedirs(length_dir, exist_ok=True)

    # Generate samples for this length
    for i in tqdm(range(num_samples_per_length), desc=f"Generating images for length {length}"):
        # Generate a random word of current length
        word = generate_random_word(length)
        randomized_word = randomize_case(word)

        # Create and save the image
        font = random.choice(fonts)
        img = create_image(randomized_word, font, image_size)
        img_filename = os.path.join(length_dir, f"{randomized_word}_{i:03d}.png")
        img.save(img_filename)

    print(f"Completed generating {num_samples_per_length} images for length {length}")

print(f"\nComplete dataset generated in '{output_dir}'")