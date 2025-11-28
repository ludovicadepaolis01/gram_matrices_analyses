import h5py
import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ndd
import pandas as pd
import argparse
from pathlib import Path

mode = "orig"

model_list = [
    "vgg16",
    "vgg19",
    "alexnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "mobilenet",
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
    "mobilenet":    "MobileNetV2",
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

#input path
path = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/gram_matrices"
file  = os.path.join(path, f"orig_gram_{model_name}_data.h5")

#output paths
rdms_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/rdms_debug"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"
csv_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs_k47"

for d in [rdms_path, plot_path, csv_path]:
    os.makedirs(d, exist_ok=True)

checkpoint_dir = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/analyses_checkpoints/analyses_ckpts_{model_name}_k47"
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

#function that runs hierarchical clustering
#it also plots: 
#1) MI by found cluster
#2) gram matrices clustering in 3d 
def hieararchical_clustering_by_mi(
        gram_vectors_data, 
        true_labels, 
        mi_function, 
        pca_function, 
        layer_name, 
        mode,
        real_classes=47, 
        plot=False, #set to True if you want to generate plots
        plot_path=".",
        checkpoint_dir=None
        ):

    if checkpoint_dir is None:
        raise ValueError("no checkpoint directory found")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{layer_name}_mi_ckpt.npz")

    mi_dict = {}
    label_dict = {}

    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        mi_dict.update(data["mi_dict"].item())
        labels_by_k = data["labels_by_k"].item()
        label_dict.update({int(k): labels_by_k[str(k)] for k in labels_by_k})
        done_ks = set(map(int, mi_dict.keys()))
        print(f"resume {layer_name}: found done ks: {sorted(done_ks)}")
    else:
        done_ks = set()

    all_ks = list(range(2, real_classes+1)) #2-47
    ks = [k for k in all_ks if k not in done_ks]

    pca = pca_function(n_components=3)
    gram_vectors_proj = pca.fit_transform(gram_vectors_data)
    
    for k in ks:
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        found_clusters = model.fit_predict(gram_vectors_data)
        data_for_mi = np.column_stack([true_labels, found_clusters])
        mutual_info = mi_function(data_for_mi)    
        mi_dict[k] = mutual_info
        label_dict[k] = found_clusters #remove astype or use np.int64

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
 
    best_k = 47
    best_labels = label_dict[best_k]

    #plot MI by found cluster
    if plot:
        fname = f"{pretty_model}_{layer_name}_{mode}_k{best_k}_mi.png"
        plt.figure(figsize=(10, 6))
        plt.plot(ks, mi_values, marker='o', linestyle='-')
        plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
        plt.xlabel("N. of found clusters")
        plt.ylabel("MI")
        plt.title(f"{pretty_model} layer {layer_name} entropy per found clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(rdms_path, fname))
        plt.close()

    #fit model with best k and plot
    final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    found_clusters_ = final_model.fit_predict(gram_vectors_data)

    #project gram vectors in 3d
    pca = pca_function(n_components=3)
    gram_vectors_proj = pca.fit_transform(gram_vectors_data)

    #visualize gram matrices clustering in 3d
    if plot: 
        fname = f"{pretty_model}_{layer_name}_{mode}_clusters.png"
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            gram_vectors_proj[:, 0], 
            gram_vectors_proj[:, 1], 
            gram_vectors_proj[:, 2],
            c=found_clusters_, 
            cmap='tab20', 
            s=50) 
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{pretty_model} {mode} layer {layer_name} k={best_k}")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cluster ID")
        plt.tight_layout()
        plt.savefig(os.path.join(rdms_path, fname))
        plt.close(fig)
    
    codes = true_labels
    return codes, best_k, best_labels, mi_dict, found_clusters_

#set up empty auto-expanding dictionaries for collecting your data grouped by layer
vecs_by_layer = defaultdict(list)
labels_by_layer = defaultdict(list)

#open file explore and sort the data: 
#need to get one similarity matrix of 5640*5640 (gram computed on images) by mode by layer
with h5py.File(file, "r") as f:
    for texture_name in f.keys():
        texture = f[texture_name]
        for batch_name in texture.keys():
            batch = texture[batch_name]
            for image_name in batch.keys():
                image = batch[image_name]
                for layer_name in image.keys():
                    gram = image[layer_name]["gram"][()]
                    vec = gram.ravel()
                    vecs_by_layer[layer_name].append(vec)
                    labels_by_layer[layer_name].append(texture_name)

#compute similarity matrix, plot and hierarchical clustering
results = []  
for layer_vectors, layer_labels in [(vecs_by_layer, labels_by_layer)]:
    for layer, vectors in layer_vectors.items():
        labels = layer_labels[layer]#[:200]
        X = np.stack(vectors, axis=0)
        #normalize rows for cosine
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X/(norms + 1e-12)
        similarity_matrix  = Xn.dot(Xn.T)

        #save similarity matrix
        matrix_path = os.path.join(rdms_path, f"{model_name}_{layer}_{mode}_cosine.npy")
        np.save(matrix_path, similarity_matrix)

        #plot similarity matrix heatmap 
        plt.figure(figsize=(20, 20)) #increase resolution with dpi argument, not recommended
        im = plt.imshow(similarity_matrix, aspect="auto", vmin=0, vmax=1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=50)
        #optionally add axis labels with ticks
        '''
        plt.xticks([])
        plt.yticks([])
        plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=20)
        plt.yticks(np.arange(len(labels)), labels, fontsize=20)
        '''
        plt.savefig(os.path.join(rdms_path, f"{model_name}_{layer}_{mode}_rsa.png"))
        plt.tight_layout()
        plt.close()

        codes, _ = pd.factorize(np.asanyarray(labels))

        codes, best_k, best_labels, mi_dict, found_clusters_ = hieararchical_clustering_by_mi(
            Xn, 
            codes,
            mi_function=ndd.mutual_information, 
            pca_function=PCA, 
            layer_name=layer, 
            mode=mode,
            plot_path=plot_path,
            checkpoint_dir=checkpoint_dir
        )
        
        clusters = np.asarray(found_clusters_).reshape(-1)

        csv_clusters = os.path.join(csv_path, f"{model_name}_{layer}_found_clusters_k47.csv")
        csv_classses = os.path.join(csv_path, f"{model_name}_{layer}_real_classes_k47.csv")
        pd.DataFrame({"cluster_id": clusters}).to_csv(csv_clusters, index=False)
        pd.DataFrame({"label": labels, "true_classes": codes}).to_csv(csv_classses, index=False)