# concat all vocab images
import os
from PIL import Image

# Define the models
models = [
    ("Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct"),
    ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), 
    ("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1"),
    ("Llama-3.1-Tulu-3-8B", "Llama-3.1-Tulu-3-8B-SFT")
]

# Get all similarity distribution plots
sim_images = []
for base, sft in models:
    sim_path = f"figs/{base}_{sft}_cosine_similarity_distribution.png"
    if os.path.exists(sim_path):
        sim_images.append(Image.open(sim_path))

# Get all wordcloud plots        
cloud_images = []
for base, sft in models:
    cloud_path = f"figs/{base}_{sft}_token_shift_wordcloud.png"
    if os.path.exists(cloud_path):
        cloud_images.append(Image.open(cloud_path))

# Check if we have any images before proceeding
if not sim_images or not cloud_images:
    print("No images found to concatenate")
    exit()

# Calculate dimensions for combined image
sim_width = max(img.width for img in sim_images)
sim_height = max(img.height for img in sim_images)
cloud_width = max(img.width for img in cloud_images) 
cloud_height = max(img.height for img in cloud_images)

# Create combined image
total_width = max(sim_width, cloud_width) * len(models)
total_height = max(sim_height, cloud_height)

combined = Image.new('RGB', (total_width, total_height), 'white')

# Paste similarity plots and wordcloud plots side by side
for i, (sim_img, cloud_img) in enumerate(zip(sim_images, cloud_images)):
    x = i * cloud_width
    y = 0
    combined.paste(cloud_img, (x, y))

# Save combined image
combined.save('figs/combined_vocab_analysis.png')
