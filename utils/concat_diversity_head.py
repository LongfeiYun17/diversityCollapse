import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

path = Path('figs/diversity_head')
if (path / 'concatenated_image.png').exists():
    os.remove(path / 'concatenated_image.png')

images = [Image.open(img_path) for img_path in path.glob('*01*.png')]
# remove tulu and phi
images = [img for img in images if 'tulu' not in img.filename and 'phi' not in img.filename]
# Get the total width and the maximum height of the concatenated image
total_width = sum(img.width for img in images)
max_height = max(img.height for img in images)

# Create a new image with the total width and maximum height
concatenated_image = Image.new('RGB', (total_width, max_height))

# Paste each image into the concatenated image
x_offset = 0
for img in images:
    concatenated_image.paste(img, (x_offset, 0))
    x_offset += img.width

# Save the concatenated image
concatenated_image.save(path / 'concatenated_image.png')

# generate legend
labels = ['[0, 0.2]', '[0.2, 0.4]', '[0.4, 0.6]', '[0.6, 0.8]', '[0.8, 1.0]']
pallete = sns.color_palette('magma')

fig, ax = plt.subplots(figsize=(6, 1))
handles = [plt.Rectangle((0, 0), 1, 1, color=pallete[i]) for i in range(len(labels))]
plt.legend(
    handles,
    labels,
    loc='center',
    ncol=len(labels),
    fontsize=12,
    frameon=False,
)
plt.axis('off')
plt.tight_layout()
plt.savefig(path / 'legend.png', bbox_inches='tight', pad_inches=0, transparent=True)