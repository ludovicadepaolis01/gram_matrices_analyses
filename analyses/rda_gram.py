import h5py
import torch
import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import glob
import ndd
from kneed import KneeLocator
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

mode = "orig"

path = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/"
rdms_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/rdms/"
file  = os.path.join(path, f"orig_gram_vgg16_data.h5")
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

for png_file in glob.glob(os.path.join(plot_path, "*.png")):
    os.remove(png_file)

def hieararchical_clustering_by_mi(gram_vectors_data, entropy_function, pca_function, layer_name, mode, real_classes=47, plot=True, plot_path="."):
    mi_dict = {}
    label_dict = {}
    
    for k in real_classes:
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        found_clusters = model.fit_predict(gram_vectors_data)
        print(len(found_clusters))
        mutual_info = entropy_function(gram_vectors_data, found_clusters)    
        mi_dict[k] = mutual_info
        label_dict[k] = found_clusters

    if not mi_dict:
        raise ValueError("no clustering found :(")

    ks = list(mi_dict.keys())
    mi_values = list(mi_dict.values())
    max_mi = max(mi_values)
    print(mi_values)
    print(max_mi)
    best_k = label_dict[max_mi]
    print(best_k)

    '''
    #elbow? or knee? any other anatomical curve
    knee = KneeLocator(ks, entropies, curve='convex', direction='decreasing')
    best_k = knee.knee if knee.knee is not None else min(entropy_dict, key=entropy_dict.get)
    best_labels = label_dict[best_k]
    '''
    #plot minimum entropy by cluster
    if plot:
        fname = f"{mode}_{layer_name}_k{best_k}_mi.png"
        os.makedirs(plot_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(ks, mi_values, marker='o', linestyle='-')
        plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
        plt.xlabel("n. of found clusters")
        plt.ylabel("entropy")
        plt.title(f"layer {layer_name} entropy per found clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, fname))
        plt.close()

    #fit model with best k and plot
    final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    found_clusters = final_model.fit_predict(gram_vectors_data)
    pca = pca_function(n_components=3)
    gram_vectors_proj = pca.fit_transform(gram_vectors_data)

    if plot: 
        fname = f"{mode}_{layer}_clusters.png"
        os.makedirs(plot_path, exist_ok=True)
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            gram_vectors_proj[:, 0], 
            gram_vectors_proj[:, 1], 
            gram_vectors_proj[:, 2],
            c=found_clusters, 
            cmap='tab20', 
            s=50) 
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
        ax.set_xlabel("pc3")
        ax.set_title(f"{mode} layer {layer_name} k={best_k}")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cluster ID")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, fname))
        plt.close(fig)

    return best_k, best_labels, mi_dict, found_clusters

#for layer, vecs in orig_vecs_by_layer.items():
#    print(f"orig layer {layer}: {len(vecs)} vectors")
#for layer, vecs in reco_vecs_by_layer.items():
#    print(f"reco layer {layer}: {len(vecs)} vectors")

#set up empty auto-expanding dictionaries for collecting your data grouped by layer
vecs_by_layer = defaultdict(list)
labels_by_layer = defaultdict(list)

#open file explore and sort the data: 
#need to get one similarity matrix of 5640*5640 (gram computed on images) per mode per layer
with h5py.File(file, "r") as f:
    #print(list(f.keys()))

    for texture_name in f.keys():
        texture = f[texture_name]
        for batch_name in texture.keys():
            batch = texture[batch_name]
            for image_name in batch.keys():
                image = batch[image_name]
                for layer_name in image.keys():
                    gram = image[layer_name]["gram"][()]
                    vec = gram.ravel()
                    #print(len(vec))
                    vecs_by_layer[layer_name].append(vec)
                    labels_by_layer[layer_name].append(f"{texture}|{batch}|{image}")

#sanity check: if batched is 376 images/grams, if not 5640
#print(len(vecs_by_layer)) #5 layers of vgg
#print(vecs_by_layer)
print(labels_by_layer.keys())
#check keys
#print(len(labels_by_layer)) #5
#print(len(img_list)) #376 for 0th epoch 14 batched images * 47 categories
#print(len(reco_img_list)) #376

results = []  
#compute similarity matrix, plot and hierarchical clustering
for layer_vectors, layer_labels in [(vecs_by_layer, labels_by_layer)]:
    for layer, vectors in layer_vectors.items():
        #vectors = vectors[:200]
        labels = layer_labels[layer]#[:200]
        print(len(labels))
        X = np.stack(vectors, axis=0).astype(np.float32)
        #normalize rows for cosine
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X/(norms + 1e-12)
        #print(Xn.shape)
        similarity_matrix  = Xn.dot(Xn.T)
        #print(similarity_matrix.shape)

        #save similarity matrix
        matrix_path = os.path.join(os.path.dirname(rdms_path), f"{mode}_{layer}_cosine.npy")
        np.save(matrix_path, similarity_matrix)
        #print(f"Saved {mode}/{layer} similarity → {out}")

        #plot similarity matrix heatmap
        plt.figure(figsize=(30,30)) #increase resolution con dpi argument
        im = plt.imshow(similarity_matrix, aspect="auto", vmin=0, vmax=1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=30)
        plt.xticks([])
        plt.yticks([])
        # turn down font-size so they fit
        plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=20)
        plt.yticks(np.arange(len(labels)), labels, fontsize=20)
        #plt.title(f"{mode} cosine similarity layer {layer}")
        #filename = os.path.join(plot_path, f"{mode}_rsa_{layer}_s{optim_step}.png")
        plt.savefig(os.path.join(plot_path, f"{mode}_rsa_{layer}.png"))#, dpi=200)
        plt.tight_layout()
        plt.close()
        #plt.savefig(os.path.join(plot_path, f"{mode}_rsa_{layer}_s{optim_step}.png"), dpi=200)
        
        #plt.show() 

        best_k, best_labels, mi_dict, found_clusters = hieararchical_clustering_by_mi(
            Xn, 
            entropy_function=ndd.entropy, 
            pca_function=PCA, 
            layer_name=layer, 
            mode=mode,
            plot_path=plot_path)
        print(best_k)
        print(best_labels)
        print(mi_dict)
        print(found_clusters)

        clusters = np.asarray(found_clusters).reshape(-1)
        codes, classes = pd.factorize(labels)

        csv_clusters = os.path.join(plot_path, f"{layer}_found_clusters.csv")
        csv_classses = os.path.join(plot_path, f"{layer}_real_classes.csv")
        pd.DataFrame({"cluster_id": clusters}).to_csv(csv_clusters, index=False)
        pd.DataFrame({"label": labels, "true_classes": codes}).to_csv(csv_classses, index=False)