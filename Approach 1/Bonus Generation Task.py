import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
output_dir = "report1"
fonts_dir = "Fonts"
image_size = (256, 128)
font_size = 40
num_samples_per_word = 1  # Number of images per word
num_required_fonts = 5
words_list = []
noise_probability = 0.1
line_probability = 0.1
jitter_range = 2
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
        font_urls = [line.split("url(")[-1].split(")")[0].strip('\'"') for line in response.text.splitlines() if
                     "url(" in line]
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


def draw_random_lines(draw, size):
    """Draw random straight or curved lines on the image."""
    num_lines = random.randint(1, 5)  # Randomize number of lines
    for _ in range(num_lines):
        if random.random() < 0.5:  # Straight line
            start_point = (random.randint(0, size[0]), random.randint(0, size[1]))
            end_point = (random.randint(0, size[0]), random.randint(0, size[1]))
            line_color = tuple(random.randint(0, 255) for _ in range(3))
            draw.line([start_point, end_point], fill=line_color, width=random.randint(1, 3))
        else:  # Curved line (approximation with multiple points)
            points = [(random.randint(0, size[0]), random.randint(0, size[1])) for _ in range(3)]
            line_color = tuple(random.randint(0, 255) for _ in range(3))
            draw.line(points, fill=line_color, width=random.randint(1, 3))


def create_image_bonus(word, fonts, size):
    """Create an image with the given word rendered with background-based rules."""
    # Randomly choose a background color: red or green
    background_color = random.choice(["red", "green"])
    img = Image.new("RGB", size, color=background_color)
    draw = ImageDraw.Draw(img)

    # Add random lines for CAPTCHA effect
    draw_random_lines(draw, size)

    # Add random pixelated noise
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < noise_probability:
                noise_color = tuple(random.randint(0, 255) for _ in range(3))
                draw.point((x, y), fill=noise_color)

    # Render each letter with a random font

    # Calculate total width and height of the word in the selected central font
    central_font = random.choice(fonts)
    total_width = sum(draw.textbbox((0, 0), letter, font=central_font)[2] for letter in word)
    total_height = draw.textbbox((0, 0), word[0], font=central_font)[3] - \
                   draw.textbbox((0, 0), word[0], font=central_font)[1]

    # Starting position for the first letter
    x = (size[0] - total_width) // 2
    y = (size[1] - total_height) // 2
    if background_color == "red": word = word[::-1] # reversing it for red background
    # Draw each letter with random font, color, and jitter
    for letter in word:
        font = random.choice(fonts)
        letter_color = tuple(random.randint(0, 127) for _ in range(3))
        text_bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Add jitter to position
        jitter_x = random.randint(-jitter_range, jitter_range)
        jitter_y = random.randint(-jitter_range, jitter_range)

        draw.text((x + jitter_x, y + jitter_y), letter, font=font, fill=letter_color)

        # Update x position for the next letter
        x += text_width
    return img


def process_word_bonus(word):
    """Generate images for a single word with bonus conditions."""
    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    for i in range(num_samples_per_word):
        randomized_word = randomize_case(word)
        img = create_image_bonus(randomized_word, fonts, image_size)
        img_filename = os.path.join(word_dir, f"{word}_{i:03d}.png")
        img.save(img_filename)


# Generate images using multithreading
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_word_bonus, words_list), desc="Generating Bonus Dataset", total=len(words_list)))

print(f"Bonus dataset generated in '{output_dir}'")
