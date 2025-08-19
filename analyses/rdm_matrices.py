import torch
import os
import numpy as np
from scipy.stats import spearmanr
import gc

optim_step = 30000

path = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/vae_project/dtd_dataset/gatys_analyses"
rdms_reco_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/vae_project/dtd_dataset/rdms_s{optim_step}/"
rdms_orig_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/vae_project/dtd_dataset/rdms_orig/"
rdms_reco = sorted([f for f in os.listdir(rdms_reco_path) if f.endswith(".npy")])
print(rdms_reco)
rdms_orig = sorted([f for f in os.listdir(rdms_orig_path) if f.endswith(".npy")])
print(rdms_orig)

with open("spearman_correlations.txt", "w") as f:

    for reco_rdms, orig_rdms in zip(rdms_reco, rdms_orig):

        reco_path = os.path.join(rdms_reco_path, reco_rdms)
        orig_path = os.path.join(rdms_orig_path, orig_rdms)

        reco_matrix = np.load(reco_path).astype(np.float32)
        orig_matrix = np.load(orig_path).astype(np.float32)
        print(reco_matrix.shape)
        print(orig_matrix.shape)
        
        triu_indices = np.triu_indices_from(reco_matrix, k=1)#k=1 to exclude the diagonal
        reco_flat = reco_matrix[triu_indices]
        orig_flat = orig_matrix[triu_indices]
        
        corr, p = spearmanr(reco_flat, orig_flat)
        print(f"spearman r between {reco_rdms} and {orig_rdms} = {corr:.4f}, p = {p:.4e}\n")

        f.write(f"spearman r between {reco_rdms} and {orig_rdms} = {corr:.4f}, p = {p:.4e}\n")