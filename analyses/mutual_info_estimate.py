import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd
import regex as re

model_list = [
    "vgg16",
    "alexnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet151",
    "googlenet",
    "inceptionv3",
    "squeezenet",
    "mobilenet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201"
]

data_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs_data_subset_05092025"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/leaderboard.csv"
output_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"

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

    mi = ndd.mutual_information(df_joint.to_numpy(dtype=int))

    class_string = os.path.basename(class_)
    cluster_string = os.path.basename(cluster)

    model_pattern = re.compile(r"^([^_]+)")
    layer_pattern = re.compile(r"layer_(?:layer|bn\d*|conv)_\d+")
    model = re.match(model_pattern, class_string).group(1)
    layer = re.search(layer_pattern, class_string).group(0)

    data.append({"model": model, "layer": layer, "mi": mi})

        #f.write(f"{os.path.basename(class_)} vs {os.path.basename(cluster)} -> MI = {mi}\n\n")

df = pd.DataFrame(data, columns=["model", "layer", "mi"]).reset_index(drop=True)
mi_csv = df.to_csv(os.path.join(output_path, "mi_csv.csv"))
#print(df)
