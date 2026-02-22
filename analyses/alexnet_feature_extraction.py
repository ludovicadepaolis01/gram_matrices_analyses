import os
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision.transforms import Compose
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#params
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import alexnet_representations
model_name="alexnet"
print("loaded alexnet")

layer_indices = [0, 3, 6, 8, 10]
print(layer_indices)

#params for image transformations
resize = 224
image_index = 0
batch_size = 8
subset_size = 10

#input paths
dtd_path = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/clip_textures/data/dtd/images"
dtd_dir = os.listdir(dtd_path)
dtd_basename = os.path.basename(os.path.dirname(dtd_path))
print(dtd_basename)

model_features_path = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/features/{model_name}/dtd"
if not os.path.exists(model_features_path):
    os.makedirs(model_features_path, exist_ok=True)

plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"
if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)


img_paths = []
for cls in sorted(os.listdir(dtd_path)):
    cls_dir = os.path.join(dtd_path, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fn in sorted(os.listdir(cls_dir)):
        img_paths.append(os.path.join(cls_dir, fn))

class ImgDataset(Dataset):
    def __init__(self, img_list, resize=resize):
        self.img_paths = img_paths
        self.img_list = img_list

        #add a preload transformation variable that contains the heaviest(?) transformations
        self.transform = Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5137, 0.4639, 0.4261), (0.2576, 0.2330, 0.2412)),
        ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        p = self.img_paths[index]
        # robust open
        with Image.open(p) as im:
            x = self.transform(im)
        return x  # single tensor
    
dataset = ImgDataset(img_paths, resize=resize)
loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1)

num_images = 1
image_size = (3, 224, 224)
mean = 0.5
std = 0.2
#values of gaussian distrib centered around dtd normalizaion values

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, choices=layer_indices, 
                    help="Which alexnet layer to extract")

args = parser.parse_args()
idx = args.index

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def alexnet_features_extraction(data_path=dtd_path,
                                basename=dtd_basename,
                                model_name=model_name,
                                selected_idx=idx,
                                loader=loader,
                                alexnet_model=alexnet_representations,
                                model_features_path=model_features_path,
                                plot=True,
                                plot_path=plot_path,
                                ):
    
    model = alexnet_model(selected_idx).to(device)
    model.eval()

    image_loader = loader

    chunks = []
    vecs = []
    sum_vec = None
    with torch.no_grad():
        for images in image_loader:
            images = images.to(device, non_blocking=True)
            _, _, _, _, feature = model(images)
            feature = feature.to(torch.float16).cpu()
            print(feature.shape)
            chunks.append(feature)
            print(f"batch {feature.shape}")

            del images, feature

    full_features = torch.cat(chunks, dim=0)
    torch.save(full_features, os.path.join(model_features_path, f"{model_name}_features_{basename}_layer_{selected_idx}.pt"))
    print(f"{model_name} image features extracted:", full_features.shape)

alexnet_features_extraction()

if torch.cuda.is_available():
    torch.cuda.empty_cache()