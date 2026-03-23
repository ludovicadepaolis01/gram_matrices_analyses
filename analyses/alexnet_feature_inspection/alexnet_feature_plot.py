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

layer_means = []
layer_stds = []

for l in sorted(layer_indices):
    fname = f"{model_name}_features_dtd_layer_{l}.pt"
    features_path = os.path.join(model_features_path, fname)

    feature = torch.load(features_path, map_location="cpu").float()
    print(l, feature.shape)

    layer_mean = feature.mean().item()
    layer_std = feature.std().item()

    layer_means.append(layer_mean)
    layer_stds.append(layer_std)

x = np.arange(len(layer_indices))

plt.figure(figsize=(8, 5))
plt.errorbar(x, layer_means, yerr=layer_stds, fmt='-o', capsize=5)

plt.xticks(x, layer_indices)
plt.xlabel("Layer number")
plt.ylabel("Activation mean and std")
plt.title(f"{model_name} dtd")
plt.tight_layout()

out = os.path.join(plot_path, f"{model_name}_dtd_stats.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()

print("saved alexnet plot")