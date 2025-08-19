import torch
import os
import numpy as np
from dataloader_dtd import class_loaders
from skdim.id import TwoNN
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

id_raw_dict = {}
id_gram_dict = {}
umap_data = []
umap_labels = []
label_to_class = {}
class_names = list(class_loaders.keys())

reducer = umap.UMAP(n_components=3)

def get_gram_matrix(image):
    gram_matrix = torch.einsum("bihw,bjhw->bij", image, image)
          
    return gram_matrix

def vectorize_gram_matrix(gram_matrix):
    upper_triangle = torch.triu(gram_matrix, diagonal=0)  #keep diagonal as well
    flat_unique_elements = upper_triangle.flatten(start_dim=1)
    
    return flat_unique_elements

for texture_name, texture_loader in class_loaders.items():
    images_list = []
    unique_elements_list = []

    for batch in texture_loader:
        batch_size = batch.size(0)
        flat_images = batch.view(batch_size, -1)
        images_list.append(flat_images.detach().cpu().numpy())

        gram_matrix = get_gram_matrix(batch)
        flat_unique_elements = vectorize_gram_matrix(gram_matrix)
        unique_elements_list.append(flat_unique_elements.detach().cpu().numpy())

        umap_data.append(flat_unique_elements.detach().cpu().numpy())
        umap_labels.extend([texture_name] * batch.size(0))

    images_vectors = np.vstack(images_list)
    gram_matrix_vectors = np.vstack(unique_elements_list)
    #gram_matrix_vectors = np.nan_to_num(gram_matrix_vectors, nan=0.0, posinf=1e6, neginf=-1e6)
    #gram_matrix_vectors = np.clip(gram_matrix_vectors, -1e6, 1e6)

    print(f"[{texture_name}] shape: {gram_matrix_vectors.shape}")
    print(f"max: {np.max(gram_matrix_vectors)}, min: {np.min(gram_matrix_vectors)}")
    print(f"contains inf: {np.isinf(gram_matrix_vectors).any()}, contains nan: {np.isnan(gram_matrix_vectors).any()}")
    print(f"contains too large: {(np.abs(gram_matrix_vectors) > 1e10).any()}")
    print(f"std across all elements: {np.std(gram_matrix_vectors)}")
    print(f"any row is constant: {np.any(np.std(gram_matrix_vectors, axis=1) == 0)}")

    id_raw = TwoNN().fit(images_vectors).dimension_
    id_gram = TwoNN().fit(gram_matrix_vectors).dimension_
    id_raw_dict[texture_name] = id_raw
    id_gram_dict[texture_name] = id_gram
    label_to_class[texture_name] = texture_name

    umap_embeddings = reducer.fit_transform(gram_matrix_vectors)
    print(umap_embeddings.shape)  

    print(f"{texture_name} - ID raw: {id_raw:.2f}, ID gram: {id_gram:.2f}")

#ID bar plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.35
ax.bar(x - width/2, [id_raw_dict[c] for c in class_names], width, label='Raw Images')
ax.bar(x + width/2, [id_gram_dict[c] for c in class_names], width, label='Gram Matrices')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=90)
ax.set_ylabel('Intrinsic Dimension')
ax.set_title('Intrinsic Dimension per Texture Class (DTD)')
ax.legend()
plt.tight_layout()
plt.savefig("/leonardo/home/userexternal/ldepaoli/lab/vae_project/dtd_dataset/gatys_analyses/plots/id_dtd_per_class.png")
plt.close()

umap_data_all = np.vstack(umap_data)
umap_labels = np.array(umap_labels)
umap_embeddings = reducer.fit_transform(umap_data_all)

#umap 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], c=umap_labels, cmap='tab20', s=10)
legend_labels = [label_to_class[i] for i in np.unique(umap_labels)]
handles = [plt.Line2D([], [], marker="o", color=sc.cmap(sc.norm(i)), linestyle="", label=label_to_class[i]) for i in np.unique(umap_labels)]
ax.legend(handles=handles, loc='best', fontsize=8)
ax.set_title("UMAP of Gram Matrices (DTD)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
plt.tight_layout()
plt.savefig("/leonardo/home/userexternal/ldepaoli/lab/vae_project/dtd_dataset/gatys_analyses/plots/umap_dtd_per_class.png")
plt.close()

with open("/leonardo/home/userexternal/ldepaoli/lab/vae_project/dtd_dataset/gatys_analyses/plots/id_dtd_summary.txt", "w") as f:
    for c in class_names:
        f.write(f"{c}: ID raw = {id_raw_dict[c]:.2f}, ID gram = {id_gram_dict[c]:.2f}\n")

