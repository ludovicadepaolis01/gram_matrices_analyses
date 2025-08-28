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

#optim_step = 30000
mode = "orig"

path = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/"
rdms_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/rdms/rdms_{mode}/"
file  = os.path.join(path, f"orig_gram_vgg16_data.h5")
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

for png_file in glob.glob(os.path.join(plot_path, "*.png")):
    os.remove(png_file)

#set up empty auto-expanding dictionaries for collecting your data grouped by layer
vecs_by_layer = defaultdict(list)
#reco_vecs_by_layer = defaultdict(list)

labels_by_layer = defaultdict(list)
#reco_labels_by_layer = defaultdict(list)

#open file explore and sort the data: 
#need to get one similarity matrix of 5640*5640 (gram computed on images) per mode per layer
with h5py.File(file, "r") as f:
    print(list(f.keys()))

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
             

    '''
    #print(f)
    #orig_layer_list = []
    #reco_layer_list = []
    #orig = f["orig"]
    #print(orig) #47 members
    #reco = f["reco"]
    #print(reco) #47 members
    
    #orig mode
    for texture in orig:
        #print(texture)
        orig_texture_group = orig[texture]
        #orig_texture_group = orig[texture]
        #print(texture_group) #1 member
        # loop epochs (e.g. "epoch_0", "epoch_1", …)

        for orig_batch in orig_texture_group:
            #print(orig_batch) #batch_14 correct because 120/8
            orig_batch_group = orig_texture_group[orig_batch]
            #print(orig_batch_group) #8 members because batch_size = 8
            for orig_img in orig_batch_group:
                #print(orig_img) #img_0 to img_7
                #orig_img_list.append(orig_img)
                orig_img_group = orig_batch_group[orig_img]
                #print(orig_img_group) #5 members that are the layers
                #now the keys here are the layer groups:
                for orig_layer in orig_img_group:
                    orig_gram = orig_img_group[orig_layer]["gram"][()]
                    orig_vec = orig_gram.ravel()
                    orig_vecs_by_layer[orig_layer].append(orig_vec)
                    orig_labels_by_layer[orig_layer].append(f"{texture}|{orig_batch}|{orig_img}")
             
        #reco mode
        for reco_batch in reco_texture_group:
            #print(reco_batch) #batch_14 correct because 120/8
            reco_batch_group = reco_texture_group[reco_batch]
            for reco_img in reco_batch_group:
                #reco_img_list.append(reco_img)
                reco_img_group = reco_batch_group[reco_img]
                for reco_layer in reco_img_group:
                    reco_gram = reco_img_group[reco_layer]["gram"][()]
                    reco_vec = reco_gram.ravel()
                    reco_vecs_by_layer[reco_layer].append(reco_vec)
                    reco_labels_by_layer[reco_layer].append(f"{texture}")#|{reco_batch}|{reco_img}")
                    #reco_labels_by_layer[reco_layer].append(f"{texture}|{reco_batch}|{reco_img}")
        '''

#sanity check: if batched is 376 images/grams, if not 5640
print(len(vecs_by_layer)) #5 layers of vgg
#check keys
print(len(vecs_by_layer)) #5
#check keys
#print(len(img_list)) #376 for 0th epoch 14 batched images * 47 categories
#print(len(reco_img_list)) #376

def optimal_clusters_by_entropy(data, true_labels, entropy_function, layer_name, mode, min_k=15, max_k=47, plot=True, plot_path="."):
    entropy_dict = {}
    label_dict = {}

    max_k = min(max_k, len(data))

    for k in range(min_k, max_k+1):
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            found_labels = model.fit_predict(data)
            print(len(found_labels))
            entropy_value = entropy_function(found_labels)
            entropy_dict[k] = entropy_value
            label_dict[k] = found_labels
        except ValueError as e:
            print(f"Salto k={k}: {e}")
            continue

    if not entropy_dict:
        raise ValueError("no clustering found :( change k")


    ks = list(entropy_dict.keys())
    entropies = list(entropy_dict.values())

    #elbow? or knee? any other anatomical curve
    knee = KneeLocator(ks, entropies, curve='convex', direction='decreasing')
    best_k = knee.knee if knee.knee is not None else min(entropy_dict, key=entropy_dict.get)
    best_labels = label_dict[best_k]

    if plot:
        fname = f"{mode}_{layer_name}_k{best_k}_entropy_knee.png"
        os.makedirs(plot_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(ks, entropies, marker='o', linestyle='-')
        plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
        plt.xlabel("n. of found clusters")
        plt.ylabel("entropy")
        plt.title(f"layer {layer_name} entropy per found clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, fname))
        plt.close()

    return best_k, best_labels, entropy_dict

def hierarchical_clustering(gram_vectors, best_k, pca_function, layer_name, mode, plot=True, plot_path="."):
    model = AgglomerativeClustering(
            n_clusters=best_k,
            #affinity='euclidean',
            linkage='ward')
    found_clusters = model.fit_predict(gram_vectors)

    pca = pca_function(n_components=3)
    gram_vectors_proj = pca.fit_transform(gram_vectors)

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
            s=50) #label=f"cluster {c}")
        #plt.legend(title="hierarchical clusters plot", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
        ax.set_xlabel("pc3")
        ax.set_title(f"{mode} layer {layer_name} k={best_k}")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cluster ID")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, fname))
        plt.close(fig)

    return found_clusters


#for layer, vecs in orig_vecs_by_layer.items():
#    print(f"orig layer {layer}: {len(vecs)} vectors")
#for layer, vecs in reco_vecs_by_layer.items():
#    print(f"reco layer {layer}: {len(vecs)} vectors")

results = []  
#compute similarity matrix, plot and hierarchical clustering
for mode, layer_vectors, layer_labels in [
    #("orig", orig_vecs_by_layer, orig_labels_by_layer),
    ("reco", reco_vecs_by_layer, reco_labels_by_layer),
]:
    for layer, vectors in layer_vectors.items():
        #vectors = vectors[:200]
        labels = layer_labels[layer]#[:200]
        #print(labels)
        X = np.stack(vectors, axis=0).astype(np.float32)
        #normalize rows for cosine
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X/(norms + 1e-12)
        #print(Xn.shape)
        similarity_matrix  = Xn.dot(Xn.T)
        #print(S)

        #save similarity matrix
        matrix_path = os.path.join(os.path.dirname(rdms_path), f"{mode}_{layer}_cosine.npy")
        np.save(matrix_path, similarity_matrix)
        #print(f"Saved {mode}/{layer} similarity → {out}")

        #similarity matrix heatmap
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
        plt.savefig(os.path.join(plot_path, f"{mode}_rsa_{layer}_s{optim_step}.png"), dpi=200)
        plt.tight_layout()
        plt.close()
        #plt.savefig(os.path.join(plot_path, f"{mode}_rsa_{layer}_s{optim_step}.png"), dpi=200)
        
        #plt.show() 

        best_k, best_labels, entropy_dict = optimal_clusters_by_entropy(Xn, labels, entropy_function=ndd.entropy, layer_name=layer, mode=mode, plot_path=plot_path)
        print(best_k)

        found_clusters = hierarchical_clustering(Xn, best_k, pca_function=PCA, layer_name=layer, mode=mode, plot_path=plot_path)
        #print(len(found_clusters))
        #print(found_clusters)

        clusters = np.asarray(found_clusters).reshape(-1)
        codes, classes = pd.factorize(labels)

        csv_clusters = os.path.join(plot_path, f"{layer}_found_clusters.csv")
        csv_classses = os.path.join(plot_path, f"{layer}_real_classes.csv")
        pd.DataFrame({"cluster_id": clusters}).to_csv(csv_clusters, index=False)
        pd.DataFrame({"label": labels, "true_classes": codes}).to_csv(csv_classses, index=False)
'''
print(len(orig_layer_list)) #1880 376 * 5 layers of vgg
print(len(reco_layer_list)) #1880
print(len(orig_img_list)) #376 for 0th epoch 14 batched images * 47 categories
print(len(reco_img_list)) #376
'''