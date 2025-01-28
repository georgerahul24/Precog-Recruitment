import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
output_dir = "letter_train"
fonts_dir = "Fonts"
image_size = (64,128)
num_samples_per_letter = 2000  # Number of images per letter
num_required_fonts = 50  # Increased number of fonts
letters_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
noise_probability = 0.03

# Font size and jitter configuration
min_font_size = 20
max_font_size = 25
min_jitter = -6
max_jitter = 6

# Ensure output and fonts directories exist
if os.path.exists(output_dir):
    os.system(f"rm -rf {output_dir}")
os.makedirs(output_dir, exist_ok=True)

os.makedirs(fonts_dir, exist_ok=True)

# Expanded list of font families with cursive and variety
font_families = [
    "Roboto", "Open+Sans", "Lato", "Montserrat", "Poppins", "Raleway", "Oswald",
    "Merriweather", "Nunito", "Ubuntu", "Playfair+Display", "Noto+Sans", "Noto+Serif",
    "Titillium+Web", "Source+Sans+Pro", "PT+Sans", "PT+Serif", "Bitter", "Arimo",
    "Cabin", "Dosis", "Quicksand", "Work+Sans", "Rubik", "Varela+Round", "Dancing+Script",
    "Pacifico", "Great+Vibes", "Courgette", "Allura", "Handlee", "Lobster", "Satisfy",
    "Cookie", "Tangerine", "Sacramento", "Parisienne", "Yellowtail", "Kaushan+Script",
    "Shadows+Into+Light", "Amatic+SC", "Indie+Flower", "Alegreya", "Bebas+Neue", "Caveat",
    "Cinzel", "Comfortaa", "Cormorant+Garamond", "Crimson+Text", "Fira+Sans", "Inconsolata",
    "Josefin+Sans", "Libre+Baskerville", "Lobster+Two", "Merriweather+Sans", "Muli",
    "Nanum+Gothic", "Nunito+Sans", "Open+Sans+Condensed", "Overpass", "PT+Mono", "Quattrocento",
    "Quattrocento+Sans", "Raleway+Dots", "Righteous", "Roboto+Condensed", "Roboto+Mono",
    "Source+Code+Pro", "Source+Serif+Pro", "Spectral", "Zilla+Slab"
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

        # Parse CSS to find .ttf or .woff2 URLs
        font_urls = [line.split("url(")[-1].split(")")[0].strip('\'"') for line in response.text.splitlines() if "url(" in line]
        for url in font_urls:
            try:
                font_response = requests.get(url)
                font_response.raise_for_status()
                font_filename = os.path.join(fonts_dir, f"{font_family.replace('+', '_')}.ttf")
                with open(font_filename, "wb") as font_file:
                    font_file.write(font_response.content)
                break  # Download only one variant per font
            except Exception as e:
                print(f"Failed to download font from {url}: {e}")

# Ensure there are at least `num_required_fonts` fonts in the folder
font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]
if len(font_files) < num_required_fonts:
    download_fonts()
    font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]

print(f"Number of fonts available: {len(font_files)}")

# Load fonts into memory
fonts = [ImageFont.truetype(font_path, random.randint(min_font_size, max_font_size)) for font_path in font_files]

def create_image(letter, font, size):
    """Create an image with the given letter rendered in the specified font."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    # Add random pixelated noise
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < noise_probability:
                noise_color = tuple(random.randint(0, 150) for _ in range(3))
                draw.point((x, y), fill=noise_color)

    text_bbox = draw.textbbox((0, 0), letter, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    jitter_x = random.randint(min_jitter, max_jitter)
    jitter_y = random.randint(min_jitter, max_jitter)
    position = ((size[0] - text_width) // 2 + jitter_x, (size[1] - text_height) // 2 + jitter_y)
    random_color = tuple(random.randint(0, 200) for _ in range(3))
    draw.text(position, letter, font=font, fill=random_color)

    return img

def process_letter(letter):
    """Generate images for a single letter."""
    letter_dir = os.path.join(output_dir, letter)
    os.makedirs(letter_dir, exist_ok=True)
    for i in range(num_samples_per_letter):
        font = random.choice(fonts)
        img = create_image(letter, font, image_size)
        img_filename = os.path.join(letter_dir, f"{letter}_{i:03d}.png")
        img.save(img_filename)

# Generate images using multithreading
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_letter, letter) for letter in letters_list]
    for future in tqdm(as_completed(futures), desc="Generating Dataset", total=len(letters_list)):
        future.result()

print(f"Dataset generated in '{output_dir}'")