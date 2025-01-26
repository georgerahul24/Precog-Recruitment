import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
output_dir = "output"
fonts_dir = "./Fonts"
image_size = (256, 128)
font_size = 32
num_samples_per_word = 5  # Number of images per word
num_required_fonts = 42
words_list = []

# Ensure output and fonts directories exist
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
    "Shadows+Into+Light", "Amatic+SC", "Indie+Flower"
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
fonts = [ImageFont.truetype(font_path, font_size) for font_path in font_files]

# Load words from dictionary file
with open("Dictionary.txt", "r") as file:
    words_list.extend(eval(file.read()))
print("Number of words found in the Dictionary: ", len(words_list))

def randomize_case(word):
    """Randomly capitalize letters in a word."""
    return ''.join(random.choice([char.upper(), char.lower()]) for char in word)

def create_image(word, font, size):
    """Create an image with the given word rendered in the specified font."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, word, font=font, fill="black")
    return img

def process_word(word):
    """Generate images for a single word."""
    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    for i in range(num_samples_per_word):
        randomized_word = randomize_case(word)
        font = random.choice(fonts)
        img = create_image(randomized_word, font, image_size)
        img_filename = os.path.join(word_dir, f"{word}_{i:03d}.png")
        img.save(img_filename)

# Generate images using multithreading
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_word, words_list), desc="Generating Dataset", total=len(words_list)))

print(f"Dataset generated in '{output_dir}'")
