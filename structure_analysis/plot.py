import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
# === Load JSON lines ===
path = "structure_analysis/results/results.jsonl"  # or .txt with each line as a json obj
lines = []
with open(path, "r") as f:
    for line in f:
        lines.append(json.loads(line))

# === Prepare data ===
models = sorted(set([d["id"].split("_")[0] for d in lines]))
modes = ["simple_steer", "full_template"]
metrics_std = ["token_length_std", "sentence_count_std", "content_ratio_std"]
metrics_entropy = ["token_length_entropy", "sentence_count_entropy", "content_ratio_entropy"]

def extract_metric(metric_list):
    data = {metric: {"simple": [], "full": []} for metric in metric_list}
    for model in models:
        for metric in metric_list:
            simple = next(d for d in lines if d["id"] == f"{model}_simple_steer")[metric]
            full = next(d for d in lines if d["id"] == f"{model}_full_template")[metric]
            data[metric]["simple"].append(simple)
            data[metric]["full"].append(full)
    return data

std_data = extract_metric(metrics_std)
entropy_data = extract_metric(metrics_entropy)

# === Plotting ===
def plot_bar(metric_data, title, ylabel, save_path):
    for i, (metric, values) in enumerate(metric_data.items()):
        x = np.arange(len(models))
        width = 0.35
        palette = sns.color_palette("Paired")
        plt.style.use("seaborn-v0_8-darkgrid")
        fontsize = 20
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width/2, values["simple"], width, label="Simple Steer", color=palette[8])
        ax.bar(x + width/2, values["full"], width, label="Full Template", color=palette[9])
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=fontsize)
        ax.set_title(f"{title}: {metric}", fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if i == 0:
            ax.legend(fontsize=fontsize - 4, ncol=2, loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{metric}.png", dpi=300)
        plt.close()

save_path = "structure_analysis/figs"
os.makedirs(save_path, exist_ok=True)
plot_bar(std_data, "Structural Diversity (Std)", "Standard Deviation", save_path)
plot_bar(entropy_data, "Structural Diversity (Entropy)", "Entropy", save_path)
