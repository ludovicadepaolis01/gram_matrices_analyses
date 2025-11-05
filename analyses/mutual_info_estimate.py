import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
from natsort import natsorted
import matplotlib.cm as cm
import matplotlib.colors as mcolors

data_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs_k47"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/leaderboard.csv"
output_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots/mi_layers_models"
subset = "all"
info_metric = "bits"

all_files = glob.glob(os.path.join(data_path, "*.csv"))

files_classes  = natsorted([f for f in all_files if "real_classes"   in os.path.basename(f)])
files_clusters = natsorted([f for f in all_files if "found_clusters" in os.path.basename(f)])

#with open(os.path.join(output_path, "output.txt"), "w") as f:

data = [] 
for class_, cluster in zip(files_classes, files_clusters): 
    df_classes = pd.read_csv(class_) 
    df_clusters = pd.read_csv(cluster) 
    df_joint = pd.DataFrame({ "true_classes": df_classes["true_classes"], "cluster_id": df_clusters["cluster_id"] }) 
    
    mi_nats = ndd.mutual_information(df_joint.to_numpy(dtype=int)) 
    mi_bits = mi_nats * np.log2(np.e) 

    class_string = os.path.basename(class_) 
    cluster_string = os.path.basename(cluster) 
    
    model_pattern = re.compile(r"^([^_]+)") 
    layer_pattern = re.compile(r"layer_(?:layer|bn\d*|conv)_\d+") 
    model = re.match(model_pattern, class_string).group(1) 
    print(model) 
    layer = re.search(layer_pattern, class_string).group(0) 
    print(layer) 
    print(f"{mi_bits}\n\n") 
    print(df_joint) 
    data.append({"model": model, "layer": layer, "mi": mi_bits}) 
    
    #f.write(f"{os.path.basename(class_)} vs {os.path.basename(cluster)} -> MI = {mi}\n\n") 
    
df = pd.DataFrame(data, columns=["model", "layer", "mi"]).reset_index(drop=True) 
mi_csv = df.to_csv(os.path.join(output_path, f"mi_csv_subset_{subset}_k47.csv")) 

df_copy = df.copy() 
df_copy["layer_idx"] = df_copy["layer"].str.extract(r"(\d+)$").astype(int) 
df_copy = df_copy.sort_values(["model", "layer_idx"])
# 2) For each model, assign x = 1..5 (position in that model after sorting)
df_copy["pos"] = df_copy.groupby("model").cumcount() + 1

# 3) Plot: one line per model, x = 1..5, y = MI
fig, ax = plt.subplots(figsize=(15, 10),
                       constrained_layout=True,)

models = sorted(df_copy["model"].unique())
colors = cm.get_cmap("tab20", len(models)) 

for i, (model, group) in enumerate(df_copy.groupby("model")):
    ax.plot(group["pos"], group["mi"], marker="o", label=model, color=colors(i))
    '''
    # optional: annotate each point with the true layer index
    for x, y, idx in zip(group["pos"], group["mi"], group["layer_idx"]):
        ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9)
    '''

ax.set_xlabel("Layer indices (1-5)", fontsize=15)
ax.set_ylabel(f"MI ({info_metric})", fontsize=15)
#ax.set_title("MI per model across layers", fontsize=15)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlim(0.5, 5.5)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid(True, alpha=0.3)
ax.legend(title="Model", loc="best")
plt.savefig(os.path.join(plot_path, f"mi_per_model_data_{subset}_{info_metric}_k47.png"), bbox_inches="tight")
plt.close(fig)