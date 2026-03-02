import ndd
import glob
import pandas as pd
import regex as re
import os
import argparse
from pathlib import Path
import numpy as np
import h5py
from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
import matplotlib.pyplot as plt
import math

mode = "orig"

model_list = [
    "vgg16",
    "vgg19",
    "alexnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet151",
    "googlenet",
    "inceptionv3",
    "densenet121",
    "densenet169",
    "densenet201"
]

pretty_model_names = {
    "alexnet":      "AlexNet",
    "densenet121":  "DenseNet-121",
    "densenet169":  "DenseNet-169",
    "densenet201":  "DenseNet-201",
    "inceptionv3":  "InceptionV3",
    "resnet18":     "ResNet18",
    "resnet34":     "ResNet34",
    "resnet50":     "ResNet50",
    "resnet101":    "ResNet101",
    "resnet152":    "ResNet152",
    "vgg16":        "VGG-16",
    "vgg19":        "VGG-19",
}

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", 
                    type=str, 
                    required=True, 
                    choices=model_list, 
                    help="Which model to run")

args = parser.parse_args()
model_name = args.model
pretty_model = pretty_model_names.get(model_name, model_name)

path = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/gram_matrices"
output_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"
file  = os.path.join(path, f"orig_gram_{model_name}_data.h5")

rdms_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/rdms"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

for d in [rdms_path, plot_path]:
    os.makedirs(d, exist_ok=True)

checkpoint_dir = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/log_analysis_checkpoints/log_analysis_ckpts_{model_name}_k47"
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

def log_function(x):
    
    return math.log(x, 2)

def hieararchical_clustering_by_mi(
        gram_vectors_data, 
        true_labels, 
        mi_function, 
        log_function,
        layer_name, 
        mode,
        real_classes=47, 
        plot=True,
        plot_path=".",
        checkpoint_dir=None
        ):

    if checkpoint_dir is None:
        raise ValueError("no checkpoint directory found")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{layer_name}_log_analysis_ckpt.npz")

    mi_dict = {}
    label_dict = {}

    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        done_ks = set(data["ks_done"].tolist())
        mi_dict.update(data["mi_dict"].item())
        labels_by_k = data["labels_by_k"].item()
        label_dict.update({int(k): labels_by_k[str(k)] for k in labels_by_k})
        print(f"resume {layer_name}: found done ks: {sorted(done_ks)}")
    else:
        done_ks = set()

    all_ks = list(range(2, real_classes+1)) #2-47
    ks = [k for k in all_ks if k not in done_ks]

    for k in ks:
        model = AgglomerativeClustering(n_clusters=k, linkage='ward',compute_distances=True)
        found_clusters = model.fit_predict(gram_vectors_data)
        data_for_mi = np.column_stack([true_labels, found_clusters])
        mutual_info = mi_function(data_for_mi)
        mutual_info = mutual_info/np.log(2)
        log_info = log_function(k)   
        mi_dict[k] = mutual_info
        label_dict[k] = found_clusters

        ks_done = sorted(set(done_ks) | set(mi_dict.keys())) 
        labels_by_k = {str(kk): label_dict[kk] for kk in label_dict}
        np.savez_compressed(
            ckpt_path,
            ks_done=np.array(ks_done),
            mi_dict=np.array(mi_dict),
            labels_by_k=np.array(labels_by_k)
        )
        print(f"ckpt layer {layer_name}: saved k={k}")

    if not mi_dict:
        raise ValueError("no clustering found :(")

    ks = sorted(mi_dict.keys())
    mi_values = [mi_dict[k] for k in ks]
    log2_47 = log_function(real_classes)
    theory_values = [min(log2_47, log_function(k)) for k in ks]

    mi_smooth = savgol_filter(mi_values, window_length=5, polyorder=2)  #tweak window_length
    mi_monotone = np.maximum.accumulate(mi_smooth)  #guards against small dips
    #elbow? or knee? any other anatomical curve
    knee_mi = KneeLocator(ks, mi_monotone, curve='concave', direction='increasing', S=7.0) #S is sensitivity curve param. the higher the smoother
    best_k_mi = knee_mi.knee if knee_mi.knee is not None else max(mi_dict, key=mi_dict.get) #which is max_mi
    best_labels = label_dict[best_k_mi]
    print(f"best k{best_k_mi}")
    print(f"best label{best_labels}")

    if plot:
        fname = f"{model_name}_{layer_name}_{mode}_k{best_k_mi}_mi_log.png"
        #os.makedirs(plot_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(ks, mi_values, marker='o', linestyle='-')
        plt.plot(ks, theory_values, marker='s', linestyle='-.')
        plt.axvline(best_k_mi, color='red', linestyle='--', label=f'Best k MI = {best_k_mi}')
        plt.xlabel("n. of found clusters")
        plt.ylabel("mutual information")
        plt.title(f"{model_name} layer {layer_name} MI and log per found clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, fname))
        plt.savefig(os.path.join(rdms_path, fname))
        plt.close() 

    #fit model with best k and plot
    final_model = AgglomerativeClustering(n_clusters=best_k_mi, linkage='ward')
    found_clusters_ = final_model.fit_predict(gram_vectors_data)

    return codes, best_k_mi, best_labels, mi_dict, found_clusters_

#set up empty auto-expanding dictionaries for collecting your data grouped by layer
vecs_by_layer = defaultdict(list)
labels_by_layer = defaultdict(list)

#open file explore and sort the data: 
#need to get one similarity matrix of 5640*5640 (gram computed on images) per mode per layer
with h5py.File(file, "r") as f:
    #print(list(f.keys()))
    for texture_name in f.keys():
        #print(texture_name)
        texture = f[texture_name]
        #print(texture)
        for batch_name in texture.keys():
            batch = texture[batch_name]
            for image_name in batch.keys():
                image = batch[image_name]
                for layer_name in image.keys():
                    gram = image[layer_name]["gram"][()]
                    vec = gram.ravel()
                    #print(len(vec))
                    vecs_by_layer[layer_name].append(vec)
                    labels_by_layer[layer_name].append(texture_name)

#compute similarity matrix, plot and hierarchical clustering
results = []  
for layer_vectors, layer_labels in [(vecs_by_layer, labels_by_layer)]:
    for layer, vectors in layer_vectors.items():
        #vectors = vectors[:200]
        labels = layer_labels[layer]#[:200]
        #print(labels)
        X = np.stack(vectors, axis=0).astype(np.float32)
        #normalize rows for cosine
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X/(norms + 1e-12)
        similarity_matrix  = Xn.dot(Xn.T)

        codes, _ = pd.factorize(np.asanyarray(labels))

        codes, best_k_mi, best_labels, mi_dict, found_clusters_ = hieararchical_clustering_by_mi(
            Xn, 
            codes,
            mi_function=ndd.mutual_information, 
            log_function=log_function,
            layer_name=layer, 
            mode=mode,
            plot_path=plot_path,
            checkpoint_dir=checkpoint_dir
        )

        clusters = np.asarray(found_clusters_).reshape(-1)
