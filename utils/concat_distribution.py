from PIL import Image

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "microsoft/Phi-3.5-mini-instruct"
]
mode = ['simple_steer', 'full_template']

fixed_width = 800
fixed_height = 600

images = []
for model in models:
    images.extend([Image.open(f"figs/distribution_chart_{model.split('/')[-1]}_{m}.png").resize((fixed_width, fixed_height)) for m in mode])

new_image = Image.new('RGB', (fixed_width * len(models), fixed_height * 2))

for idx, model in enumerate(models):
    new_image.paste(images[idx * 2], (fixed_width * idx, 0))
    new_image.paste(images[idx * 2 + 1], (fixed_width * idx, fixed_height))

new_image.save("figs/concatenated_distribution_chart_all_models.png")