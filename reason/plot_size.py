import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

with_base = False
topk = False
model_name = 'llama'
_1b_folder = Path(f'reason/entropies/{model_name}_3.2_1b')
_3b_folder = Path(f'reason/entropies/{model_name}')
_8b_folder = Path(f'reason/entropies/{model_name}_3.1_8b')

# get all .pkl files in the folder
_1b_pkl_files = list(_1b_folder.glob('*.pkl'))
_3b_pkl_files = list(_3b_folder.glob('*.pkl'))
_8b_pkl_files = list(_8b_folder.glob('*.pkl'))

# load all pkl files
_1b_entropies = [pickle.load(open(file, 'rb')) for file in _1b_pkl_files]
_3b_entropies = [pickle.load(open(file, 'rb')) for file in _3b_pkl_files]
_8b_entropies = [pickle.load(open(file, 'rb')) for file in _8b_pkl_files]

avg_1b_full_template_entropies = [0] * 50
avg_3b_full_template_entropies = [0] * 50
avg_8b_full_template_entropies = [0] * 50

for entropy in _1b_entropies:
    for i in range(50):
        avg_1b_full_template_entropies[i] += entropy['full_template'][i]

for entropy in _3b_entropies:
    for i in range(50):
        avg_3b_full_template_entropies[i] += entropy['full_template'][i]

for entropy in _8b_entropies:
    for i in range(50):
        avg_8b_full_template_entropies[i] += entropy['full_template'][i]

avg_1b_full_template_entropies = [e / len(_1b_entropies) for e in avg_1b_full_template_entropies]
avg_3b_full_template_entropies = [e / len(_3b_entropies) for e in avg_3b_full_template_entropies]
avg_8b_full_template_entropies = [e / len(_8b_entropies) for e in avg_8b_full_template_entropies]

print(avg_1b_full_template_entropies)
print(avg_3b_full_template_entropies)
print(avg_8b_full_template_entropies)

palette = sns.color_palette("Paired")
# plot the entropies
plt.style.use('seaborn-v0_8-darkgrid')
fontsize = 20
plt.figure(figsize=(10, 6))
plt.plot(range(50), avg_1b_full_template_entropies, 'o-', label='1B', color=palette[0], markersize=4)
plt.plot(range(50), avg_3b_full_template_entropies, 'o-', label='3B', color=palette[1], markersize=4)
plt.plot(range(50), avg_8b_full_template_entropies, 'o-', label='8B', color=palette[2], markersize=4)
plt.xlabel('Generation Step', fontsize=fontsize)
plt.ylabel('Average Entropy', fontsize=fontsize)
plt.title(f'{model_name} Entropies Comparison', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)
print(f'reason/figs/{model_name}_size_entropies.png')
plt.savefig(f'reason/figs/{model_name}_size_entropies.png')
