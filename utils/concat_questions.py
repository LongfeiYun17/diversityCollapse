import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='simple_steer', choices=['simple_steer', 'full_template'])
args = parser.parse_args()

domains = ["education_questions", "technology_questions", "society_questions", "healthcare_questions", "economy_questions"]
mode = args.mode
figs_dir = Path('data/visualizations')

all_figs = []
for domain in domains:
    fig_path = figs_dir / f'response_by_question_{mode}_{domain}.png'
    if not fig_path.exists():
        print(f'{fig_path} does not exist')
        continue
    all_figs.append(fig_path)


# concatenate all_figs in one row
# create a new figure
# original size is 15 x 10
fig = plt.figure(figsize=(15 * len(all_figs), 10))
gs = GridSpec(1, len(all_figs), figure=fig)

for i, fig_path in enumerate(all_figs):
    # Read the image
    img = plt.imread(fig_path)
    
    # Create subplot with tight layout
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img)
    ax.axis('off')

# Remove padding/spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.tight_layout(pad=0)

# Save with bbox_inches to remove any remaining whitespace
plt.savefig(figs_dir / f'all_figs_{mode}.png', bbox_inches='tight', pad_inches=0)
