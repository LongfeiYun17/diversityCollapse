import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

with_base = False
topk = True
model_name = 'tulu'
if topk:
    folder = Path(f'reason/entropies_topk/{model_name}')
else:
    folder = Path(f'reason/entropies/{model_name}')
if with_base:
    base_folder = Path(f'reason/entropies/llama_base')

# get all .pkl files in the folder
pkl_files = list(folder.glob('*.pkl'))
if with_base:
    base_pkl_files = list(base_folder.glob('*.pkl'))
# load all pkl files
entropies = [pickle.load(open(file, 'rb')) for file in pkl_files]
if with_base:
    base_entropies = [pickle.load(open(file, 'rb')) for file in base_pkl_files]

avg_full_template_entropies = [0] * 50
avg_simple_steer_entropies = [0] * 50
if with_base:
    avg_base_simple_steer_entropies = [0] * 50

for entropy in entropies:
    for i in range(50):
        avg_full_template_entropies[i] += entropy['full_template'][i]
        avg_simple_steer_entropies[i] += entropy['simple_steer'][i]

if with_base:
    for entropy in base_entropies:
        for i in range(min(len(entropy['simple_steer']), 50)):
            avg_base_simple_steer_entropies[i] += entropy['simple_steer'][i]

avg_full_template_entropies = [e / len(entropies) for e in avg_full_template_entropies]
avg_simple_steer_entropies = [e / len(entropies) for e in avg_simple_steer_entropies]
if with_base:
    avg_base_simple_steer_entropies = [e / len(base_entropies) for e in avg_base_simple_steer_entropies]

print(avg_full_template_entropies)
print(avg_simple_steer_entropies)
if with_base:
    print(avg_base_simple_steer_entropies)
palette = sns.color_palette("Paired")
# plot the entropies
plt.style.use('seaborn-v0_8-darkgrid')
fontsize = 20
plt.figure(figsize=(10, 6))
plt.plot(range(50), avg_full_template_entropies, 'o-', label='Full Template', color=palette[4], markersize=4)
plt.plot(range(50), avg_simple_steer_entropies, 'o-', label='Simple Steer', color=palette[5], markersize=4)
if with_base:
    plt.plot(range(50), avg_base_simple_steer_entropies, 'o-', label='Base Simple Steer', color=palette[6], markersize=4)
plt.xlabel('Generation Step', fontsize=fontsize)
plt.ylabel('Average Entropy', fontsize=fontsize)
if topk:
    plt.title(f'{model_name} Top-K Entropies Comparison', fontsize=fontsize)
else:
    plt.title(f'{model_name} Entropies Comparison', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize - 2)
if topk:
    print(f'reason/figs_topk/{model_name}_entropies.png')
    plt.savefig(f'reason/figs_topk/{model_name}_entropies.png')
else:
    print(f'reason/figs/{model_name}_entropies.png')
    plt.savefig(f'reason/figs/{model_name}_entropies.png')
