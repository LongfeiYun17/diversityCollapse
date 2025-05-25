import os
from PIL import Image, ImageDraw, ImageFont

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "microsoft/Phi-3.5-mini-instruct"
]
mode = ['simple_steer', 'full_template']

# Define fixed width and height for each image
fixed_width = 800
fixed_height = 600

# Create a new image with 2 rows and 5 columns
images = []
for model in models:
    images.extend([Image.open(f"figs/attention_flow_{model.split('/')[-1]}_{m}.png").resize((fixed_width, fixed_height)) for m in mode])

new_image = Image.new('RGB', (fixed_width * len(models), fixed_height * 2))

# Load a font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Add model name text and paste images
for idx, model in enumerate(models):
    draw = ImageDraw.Draw(new_image)
    draw.text((fixed_width * idx + 10, 10), model, font=font, fill="black")
    
    new_image.paste(images[idx * 2], (fixed_width * idx, 0))
    new_image.paste(images[idx * 2 + 1], (fixed_width * idx, fixed_height))

new_image.save("figs/concatenated_attention_flow_all_models.png")
