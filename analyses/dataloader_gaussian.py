import torch.utils
import torch.utils.data
import h5py
#import matplotlib.pyplot as plt
import torch
import torchvision.utils as u
import os
import numpy as np
import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision.transforms import Compose
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import trange, tqdm

subset_size = 5
batch_size = 8
num_images = 5640
image_size = (3, 224, 224)
mean = 0.5
std = 0.2
#values of gaussian distrib centered around dtd normalizaion values

class GaussianImageDataset(Dataset):
    def __init__(self, num_images=num_images, image_size=image_size, mean=mean, std=std):
        self.num_images = num_images
        self.image_size = image_size
        self.mean = mean
        self.std = std
        #self.clamp = clamp

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gaussian_img = torch.randn(self.image_size) * self.std + self.mean
        #if self.clamp:
        #    gaussian_img = gaussian_img#.clamp(0, 1)
        return gaussian_img
    
subset_size = subset_size

gaussian_dataset = GaussianImageDataset()
gaussian_loader = DataLoader(gaussian_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
subset_indices = list(range(len(gaussian_dataset)))[:subset_size]
gaussian_subset = Subset(gaussian_dataset, subset_indices)
gaussian_subset_loaders = DataLoader(gaussian_subset, batch_size=batch_size, shuffle=True)    

#print(len(train_subset))
#for batch in gaussian_loader:
#    batch = batch.to("cuda")  
    
#print(batch.shape)