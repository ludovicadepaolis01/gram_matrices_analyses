import os
import numpy as np
import ndd
import glob
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
from natsort import natsorted
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#input paths
gram_matrices_path = "/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/csvs_k47"
brainscore_table_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs/leaderboard.csv"

#output paths
out_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

#params
info_metric = "bits"

all_files = glob.glob(os.path.join(gram_matrices_path, "*.csv"))

files_classes  = natsorted([f for f in all_files if "real_classes"   in os.path.basename(f)])
files_clusters = natsorted([f for f in all_files if "found_clusters" in os.path.basename(f)])

#mask
pretty_model_names = {
    "alexnet":      "AlexNet",
    "densenet121":  "DenseNet-121",
    "densenet169":  "DenseNet-169",
    "densenet201":  "DenseNet-201",
    "inceptionv3":  "InceptionV3",
    "mobilenet":    "MobileNetV2",
    "resnet18":     "ResNet18",
    "resnet34":     "ResNet34",
    "resnet50":     "ResNet50",
    "resnet101":    "ResNet101",
    "resnet152":    "ResNet152",
    "vgg16":        "VGG-16",
    "vgg19":        "VGG-19",
}

def mi_estimate(
        classes,
        clusters,
        output_path,
        plot=True,
        plot_path=".",
        ):
    data = [] 
    for class_, cluster in zip(classes, clusters): 
        df_classes = pd.read_csv(class_) 
        df_clusters = pd.read_csv(cluster) 
        df_joint = pd.DataFrame({"true_classes": df_classes["true_classes"], "cluster_id": df_clusters["cluster_id"] }) 

        #convert MI in bits for easy display
        mi_nats = ndd.mutual_information(df_joint.to_numpy(dtype=int)) 
        mi_bits = mi_nats * np.log2(np.e) 

        class_string = os.path.basename(class_) 
        
        model_pattern = re.compile(r"^([^_]+)") 
        layer_pattern = re.compile(r"layer_(?:layer|bn\d*|conv)_\d+") 
        model = re.match(model_pattern, class_string).group(1) 
        layer = re.search(layer_pattern, class_string).group(0) 
        data.append({"model": model, "layer": layer, "mi": mi_bits}) 
            
    df = pd.DataFrame(data, columns=["model", "layer", "mi"]).reset_index(drop=True) 
    mi_csv = df.to_csv(os.path.join(output_path, f"mi_csv_k47.csv")) 

    df_copy = df.copy() 
    df_copy["layer_idx"] = df_copy["layer"].str.extract(r"(\d+)$").astype(int) 
    df_copy = df_copy.sort_values(["model", "layer_idx"])
    # 2) For each model, assign x = 1..5 (position in that model after sorting)
    df_copy["pos"] = df_copy.groupby("model").cumcount() + 1

    if plot:
        # 3) Plot: one line per model, x = 1..5, y = MI
        fig, ax = plt.subplots(figsize=(15, 10),
                            constrained_layout=True,)

        models = sorted(df_copy["model"].unique())
        colors = cm.get_cmap("tab20", len(models)) 

        for idx, (model, group) in enumerate(df_copy.groupby("model")):
            ax.plot(group["pos"], group["mi"], marker="o", label=pretty_model_names.get(model, model), color=colors(idx))
            '''
            #optional: annotate each point with the true layer index
            for x, y, i in zip(group["pos"], group["mi"], group["layer_idx"]):
                ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=9)
            '''

        ax.set_xlabel("Layer indices (1-5)", fontsize=15)
        ax.set_ylabel(f"MI ({info_metric})", fontsize=15)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.5, 5.5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(title="Model", loc="best", fontsize=11)
        leg.get_title().set_fontsize(11) #title font size
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"mi_per_model_data_{info_metric}_k47.png"), bbox_inches="tight")
        plt.close(fig)
    
mi_estimate(
    classes=files_classes,
    clusters=files_clusters,
    plot_path=plot_path,
    output_path=out_path
)