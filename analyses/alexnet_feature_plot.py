import os
import torch
import matplotlib.pyplot as plt
import numpy as np

#params
device = "cuda" if torch.cuda.is_available() else "cpu"

layer_indices = [0, 3, 6, 8, 10]

model_name = "alexnet"

model_features_path = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/features/{model_name}/dtd"
if not os.path.exists(model_features_path):
    os.makedirs(model_features_path, exist_ok=True)

plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"
if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

mean_vecs = []
for l in sorted(layer_indices):
    fname = f"{model_name}_features_dtd_layer_{l}.pt"
    features_path = os.path.join(model_features_path, fname)
    
    feature = torch.load(features_path, map_location="cpu")
    print(l, feature.shape)  

    vec = feature.float().mean(dim=(0, 2, 3)).numpy()
    mean_vecs.append(vec)

rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
axes = axes.flatten()
    
for i, (layer, vec) in enumerate(zip(layer_indices, mean_vecs)):
    axes[i].plot(vec, alpha=0.8)
    axes[i].set_title(f"{model_name} dtd layer {l}")
    axes[i].set_xlabel("channel index")
    axes[i].set_ylabel("mean value")

out = os.path.join(plot_path, f"{model_name}_dtd_feaures.png")
plt.tight_layout()
plt.savefig(out)
plt.close()
print("alexnet feature plot done")  
