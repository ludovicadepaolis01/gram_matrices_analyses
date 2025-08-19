import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd


data_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_project/vgg_analyses/plots"

all_files = glob.glob(os.path.join(data_path, "*.csv"))

files_classes  = sorted([file for file in all_files if "real_classes" in os.path.basename(file)])
files_clusters = sorted([file for file in all_files if "found_clusters" in os.path.basename(file)])

for class_, cluster in zip(files_classes, files_clusters):
    df_classes  = pd.read_csv(class_)
    df_clusters = pd.read_csv(cluster)

    df_joint = pd.DataFrame({
        "true_classes": df_classes["true_classes"],
        "cluster_id": df_clusters["cluster_id"]
    })


    mi = ndd.mutual_information(df_joint.to_numpy(dtype=int))
    print(f"{os.path.basename(class_)} vs {os.path.basename(cluster)} -> MI = {mi}")

'''
def get_mutual_info(n_classes, n_clusters):
    counts = np.random.randint(n_classes, n_clusters)
    H_classes = np.log2(n_classes)
    H_classes_given_cluster = np.zeros(n_clusters)
    cluster_frequency = np.zeros(n_clusters)

    for cluster in range(n_clusters):
        H_classes_given_cluster[cluster] = ndd.entropy(counts[:,cluster])
        cluster_frequency[cluster] = counts[:,cluster].sum()

    H_classes_conditional = (H_classes_given_cluster * cluster_frequency  / counts.sum()).sum()
    mutual_info = H_classes - H_classes_conditional

    return mutual_info

for class_, cluster in zip(df_classes, df_clusters):
    class_ = class_["true_classes"].tolist()
    cluster = cluster["cluster_id"].tolist()
    mi = get_mutual_info(cluster)
    print(mi)
'''