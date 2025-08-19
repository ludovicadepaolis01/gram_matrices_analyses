import torch
#import torchvision.datasets
import torchvision.utils as u
#from torchvision.transforms.functional import pil_to_tensor
#from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from dataloader_dtd import class_loaders, class_subset_loaders
from dataloader_gaussian import gaussian_loader, gaussian_subset_loaders
#from torch.utils.data import dataset
#import torch.nn as nn
#from torch.nn import parameter
#from torch.nn import functional as F
#import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.optim import AdamW
import os
#from PIL import Image
#import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
#import torchvision.models as models
import pandas as pd
import gc
import h5py
#import glob
from pathlib import Path
import argparse

OMP_NUM_THREADS=1

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import (
    VGG16_representations, 
    alexnet_representations, 
    resnet18_representations, 
    resnet34_representations, 
    resnet18_representations, 
    resnet50_representations,
    resnet101_representations,
    resnet152_representations,
    googlenet_representations,
    inceptionv3_representations,
    squeezenet_representations,
    mobilenet_representations,
    densenet121_representations,
    densenet161_representations,
    densenet169_representations,
    densenet201_representations,
    )

model_dict = {
    "vgg16": VGG16_representations,
    "alexnet": alexnet_representations,
    "resnet18": resnet18_representations,
    "resnet34": resnet34_representations,
    "resnet50": resnet50_representations,
    "resnet101": resnet101_representations,
    "resnet151": resnet152_representations,
    "googlenet": googlenet_representations,
    "inceptionv3": inceptionv3_representations,
    "squeezenet": squeezenet_representations,
    "mobilenet": mobilenet_representations,
    "densenet121": densenet121_representations,
    "densenet161": densenet161_representations,
    "densenet169": densenet169_representations,
    "densenet201": densenet201_representations,
}

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=model_dict.keys(), help="Which model to run")
args = parser.parse_args()

model_name = args.model
model = model_dict[args.model]()
#print(model)
MSE = torch.nn.MSELoss()
device = "cuda"
optim_steps = 30000

reco_path = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/reco_images_{model_name}"
orig_path = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/orig_images_{model_name}"
info_plot_path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/info_plots_{model_name}"
gram_matrices_path = f"/leonardo_scratch/fast/Sis25_piasini/ldepaoli/gram_matrices_analyses/gram_{model_name}_data_s{optim_steps}.h5"

for d in [reco_path, orig_path, info_plot_path, gram_matrices_path]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(gram_matrices_path):
    with h5py.File(gram_matrices_path, "w") as _:
        pass

checkpoint_path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_project/gram_matrices_analyses/class_ckpts_{model_name}"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

def save_class_ckpt(ckpt_path, batch_idx, step, reco_image, optimizer):
    torch.save(
        {
            "batch_idx": int(batch_idx),
            "step": int(step),
            "reco_image": reco_image.detach().cpu(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )

def load_class_ckpt_if_any(ckpt_path):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return {
            "batch_idx": int(ckpt.get("batch_idx", 0)),
            "step": int(ckpt.get("step", 0)),
            "reco_image": ckpt.get("reco_image", None),
            "optimizer_state": ckpt.get("optimizer", None),
        }
    return None

#take the layer names for data storage thanks chatgpt
_ = model(torch.randn(1,3,224,224, device=device))  #dummy forward
layer_names = list(model.feature_maps.keys())
#print(layer_names)

model.zero_grad()
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

#parameters for training
mean = (0.5137, 0.4639, 0.4261)
std = (0.2576, 0.2330, 0.2412)

#parameters for dataloader
batch_size = 5

def denormalize(input, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    denorm = input*std+mean

    return denorm 

preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

'''
tensor_1 = preprocess(image_1)
tensor_1 = tensor_1.unsqueeze(0).to(device)
'''

data_gram_list = []

#h5f = h5py.File(gram_matrices_path, "a", libver=("latest", "latest"))
h5f = h5py.File(gram_matrices_path, "a")

for texture_name, texture_loader in class_subset_loaders.items():
    #print(texture_class)
    #img_index 0-14 because 120(img per class)/8(batch_size)
    ckpt_path = os.path.join(checkpoint_path, f"{texture_name}.pt")
    class_ckpt = load_class_ckpt_if_any(ckpt_path)
    start_batch = class_ckpt["batch_idx"] if class_ckpt else 0

    for batch, (orig_image, reco_image) in enumerate(zip(texture_loader, gaussian_subset_loaders)):
    #for img_idx, orig_image in enumerate(texture_loader):
        #print(f"img_idx {img_idx}")
        if batch < start_batch:
            continue  # skip already-completed batches for this class

        orig_image = orig_image.to(device)
        reco_image = nn.Parameter(reco_image.clone().detach().to(device)) #define gradient with respect to image as a parameter
        #reco_image = reco_image.clone().detach().to(device)
        #reco_image = torch.randn_like(orig_image) #in place of the gaussian dataloader
        #reco_image.clamp_(0.0, 1.0) #in place of the gaussian dataloader
        #reco_image.requires_grad = True
        optimizer = optim.Adam([reco_image], lr=1e-4)

        start_step = 0
        if class_ckpt and batch == start_batch:
            if class_ckpt["reco_image"] is not None:
                with torch.no_grad():
                    reco_image.copy_(class_ckpt["reco_image"].to(device))
            if class_ckpt["optimizer_state"] is not None:
                optimizer.load_state_dict(class_ckpt["optimizer_state"])
            start_step = class_ckpt["step"]
            if start_step > 0:
                print(f"[resume] class={texture_name} batch={batch} step={start_step}")

        with torch.no_grad():
            orig_gram_matrices, feature_map_m_list, orig_feature_map_list = model(orig_image)

        #optimization
        for step in range(optim_steps): #this will print a progression line for each batch = 15 progression lines (120/8)
            optimizer.zero_grad()

            reco_gram_matrices, _, reco_feature_map_list = model(reco_image)

            sum_gram_matrix_loss = 0
            for orig_gram_matrix, reco_gram_matrix, m in zip(orig_gram_matrices, reco_gram_matrices, feature_map_m_list):
                gram_matrix_loss = MSE(orig_gram_matrix, reco_gram_matrix)/(4*m)
                sum_gram_matrix_loss += gram_matrix_loss

            #backward pass over the images and update optimizer
            sum_gram_matrix_loss.backward()
            optimizer.step()

            #now create dataframes for the plots
            data_gram = {
                "optim_step": step,
                "texture": texture_name,
                "loss": sum_gram_matrix_loss.item()
            }

            csv_path = os.path.join(info_plot_path, f"gram_losses_{texture_name}_s{optim_steps}.csv")
            df_gram = pd.DataFrame([data_gram])
            if os.path.exists(csv_path):
                df_gram.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df_gram.to_csv(csv_path, mode="w", header=True, index=False)      

            if step % 1000 == 0:
                save_class_ckpt(ckpt_path, batch_idx=batch, step=step + 1, reco_image=reco_image, optimizer=optimizer)

        with torch.no_grad():
            #denormalize both images ????
            denorm_image = denormalize(orig_image, mean, std).to(device)
            denorm_reco = denormalize(reco_image, mean, std).to(device)
        orig = u.save_image(denorm_image, os.path.join(orig_path, f"orig_{texture_name}_b{batch}_s{step}.png"))
        reco = u.save_image(denorm_reco, os.path.join(reco_path, f"reco_{texture_name}_b{batch}_s{step}.png"))

        with torch.no_grad():
            reco_gram_matrices, _, _ = model(reco_image)

        for gram_idx, (orig_gram, reco_gram) in enumerate(zip(orig_gram_matrices, reco_gram_matrices)):
            B, C, _ = orig_gram.shape
            name = layer_names[gram_idx]
            for b in range(B):
                #g_orig = orig_gram[b].detach().cpu().numpy()
                g_reco = reco_gram[b].detach().cpu().numpy()
                base_path = f"{texture_name}/batch_{batch}/img_{b}/layer_{name}"
                #group_orig = h5f.require_group("orig/" + base_path)
                #print(group_orig)
                group_reco = h5f.require_group("reco/" + base_path)
                #print(group_reco)
                #group_orig.create_dataset("gram", data=g_orig, compression="gzip", compression_opts=4)
                #print(group_orig)
                if "gram" in group_reco:
                    del group_reco["gram"]  # overwrite if already present
                group_reco.create_dataset("gram", data=g_reco, compression="gzip", compression_opts=4)
                #print(group_reco)

    save_class_ckpt(ckpt_path, batch_idx=batch + 1, step=0, reco_image=reco_image, optimizer=optimizer)

    print(f"optimization_steps: {optim_steps}, texture: {texture_name}, gram loss: {sum_gram_matrix_loss}")#, reconstruction loss: {reconstruction_loss}")

    h5f.flush() 
    torch.cuda.empty_cache()
    gc.collect()

h5f.close()