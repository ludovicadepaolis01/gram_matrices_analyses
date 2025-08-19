import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.optim as optim
from torch.optim import Adam
import torchvision.models as models
import os

OMP_NUM_THREADS=1

#resnet101_pretrained = models.resnet101()
#print(resnet101_pretrained)

#alexnet
class resnet101_representations(nn.Module):
     model_path = "/leonardo/home/userexternal/ldepaoli/resnet101-5d3b4d8f.pth"
     def __init__(self): #images
          super().__init__() #refers to the class that this class inherits from (nn.Modules)
          self.resnet101_pretrained = models.resnet101()
          state_dict = torch.load(resnet101_representations.model_path)
          self.resnet101_pretrained.load_state_dict(state_dict)
          self.resnet101_pretrained.to("cuda")
          self.resnet101_pretrained.requires_grad_(False) #to not compute gradients with respect to model parameters
          self.hooks = []
          self.feature_maps = {}

          count = 0
          #for (name, layer) in alexnet.features.named_modules(): #this is useful to extract only the ReLU of the of the conv not of the classifier
          for (name, module) in self.resnet101_pretrained.named_modules():
               #print(name)
               #print(module)
               if name == "bn1" or name.endswith(".0.bn1") and "downsample" not in name:
                    module.name = f"bn1_{count}"
                    hook = module.register_forward_hook(self.hook_func)
                    #print(f"Registered hook on: {name} as {module.name}")
                    self.hooks.append(hook)
               count += 1

     def hook_func(self, module, input, output): #all of the three arguments are necessary
          name = module.name
          #print(f"Hook fired for: {name}")
          #feature_maps = self.feature_maps
          self.feature_maps[name] = output#.detach()
          
     def gram_matrix(self, feature_map):
          #for tensor in feature_map:
          gram_matrix = torch.einsum("bihw,bjhw->bij", feature_map, feature_map)
          
          return gram_matrix
     
     def forward(self, images):
          gram_matrix_list = []
          feature_map_m_list = []
          feature_map_n_list = []
          feature_map_list = []
          model = self.resnet101_pretrained(images)

          #check if gram matrices (list) are not transferred from gpu to cpu from time to time
          #this could slow down the whole learning process
          #
          for key in self.feature_maps:
               feature_map = self.feature_maps[key]
               feature_map_height = feature_map.size(2)
               feature_map_width = feature_map.size(3)
               feature_map_m = feature_map_height*feature_map_width
               feature_map_m_list.append(feature_map_m)
               gram_matrices = self.gram_matrix(feature_map)
               gram_matrix_list.append(gram_matrices)
               feature_map_list.append(feature_map)
               
          return gram_matrix_list, feature_map_m_list, feature_map_list
     
def gaussian_image_tensor(size=400, mean=0.5, std=0.2):
    gaussian_image = torch.randn(3, size, size) * std + mean
    gaussian_image.clamp_(0.0, 1.0)
    return gaussian_image.to("cuda")#.unsqueeze(0)
