from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont

figs_path = Path('figs/t-sne')

# concat all the t-sne images
models = [
    "Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.1",
]

# Define fixed width and height for each image
fixed_width = 800
fixed_height = 600

# Collect all t-SNE images
images = []
for model in models:
    image_path = figs_path / f"tsne_{model}.png"
    if image_path.exists():
        images.append(Image.open(image_path).resize((fixed_width, fixed_height)))

# Create a new image with all models in one row horizontally
if images:
    new_image = Image.new('RGB', (fixed_width * len(images), fixed_height))
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Paste images and add model names
    for idx, (model, img) in enumerate(zip(models, images)):
        x_offset = idx * fixed_width
        new_image.paste(img, (x_offset, 0))
        
        # Add model name text
        draw = ImageDraw.Draw(new_image)
        draw.text((x_offset + 10, 10), model, font=font, fill="white")
    
    # Save the concatenated image
    new_image.save(figs_path / "concatenated_tsne_all_models.png")