import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd
import regex as re
import matplotlib.pyplot as plt

data_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/leaderboard.csv"
output_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"
subset = "all"

all_files = glob.glob(os.path.join(data_path, "*.csv"))

files_classes  = sorted([file for file in all_files if "real_classes" in os.path.basename(file)])
files_clusters = sorted([file for file in all_files if "found_clusters" in os.path.basename(file)])

#with open(os.path.join(output_path, "output.txt"), "w") as f:

data = []
for class_, cluster in zip(files_classes, files_clusters):
    
    df_classes  = pd.read_csv(class_)
    df_clusters = pd.read_csv(cluster)

    df_joint = pd.DataFrame({
        "true_classes": df_classes["true_classes"],
        "cluster_id": df_clusters["cluster_id"]
    })

    mi_nats = ndd.mutual_information(df_joint.to_numpy(dtype=int))
    #print(mi_nats)
    mi_bits = mi_nats * np.log2(np.e)
    #print(mi_bits)

    class_string = os.path.basename(class_)
    cluster_string = os.path.basename(cluster)

    model_pattern = re.compile(r"^([^_]+)")
    layer_pattern = re.compile(r"layer_(?:layer|bn\d*|conv)_\d+")
    model = re.match(model_pattern, class_string).group(1)
    layer = re.search(layer_pattern, class_string).group(0)

    data.append({"model": model, "layer": layer, "mi": mi_bits})

        #f.write(f"{os.path.basename(class_)} vs {os.path.basename(cluster)} -> MI = {mi}\n\n")

df = pd.DataFrame(data, columns=["model", "layer", "mi"]).reset_index(drop=True)
mi_csv = df.to_csv(os.path.join(output_path, f"mi_csv_subset_{subset}.csv"))
#print(df)

df_copy = df.copy()
df_copy["layer_idx"] = df_copy["layer"].str.extract(r"(\d+)$").astype(int)
df_copy = df_copy.sort_values(["model", "layer_idx"])

# 2) For each model, assign x = 1..5 (position in that model after sorting)
df_copy["pos"] = df_copy.groupby("model").cumcount() + 1

# 3) Plot: one line per model, x = 1..5, y = MI
fig, ax = plt.subplots(figsize=(15, 10))
for model, group in df_copy.groupby("model"):
    ax.plot(group["pos"], group["mi"], marker="o", label=model)
    # optional: annotate each point with the true layer index
    for x, y, idx in zip(group["pos"], group["mi"], group["layer_idx"]):
        ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)

ax.set_xlabel("Layers")
ax.set_ylabel("MI (bits)")
ax.set_title("MI per model across layers")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlim(0.5, 5.5)
ax.grid(True, alpha=0.3)
ax.legend(title="Model", loc="best")

plt.tight_layout()
plt.savefig(os.path.join(plot_path, f"mi_per_model_data_{subset}_bits.png"), bbox_inches="tight")
plt.close(fig)