import matplotlib.pyplot as plt

# concatenate two images horizontally
def concatenate_images(img1, img2, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[1].imshow(img2)
    ax[1].axis('off')
    
    plt.savefig(f"{save_path}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    model_names = ['Meta-Llama-3-8B-Instruct', 
                  'Llama-3.1-Tulu-3-8B-SFT',
                  'Mistral-7B-Instruct-v0.1',
                  'Phi-3.5-mini-instruct',
                  'Qwen2.5-7B-Instruct'
                  ]
    for model_name in model_names:
        img_path1 = f'figs/distribution_chart_{model_name}_simple_steer.png'
        img_path2 = f'figs/distribution_chart_{model_name}_full_template.png'
        img1 = plt.imread(img_path1)
        img2 = plt.imread(img_path2)
        concatenate_images(img1, img2, f"figs/concatenated_{model_name}.png")
