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
import re

OMP_NUM_THREADS=1

#path = "/leonardo/home/userexternal/ldepaoli/densenet121-a639ec97.pth"
#state_dict = torch.load(path, map_location="cpu")

#print(type(state_dict))
#print(list(state_dict.keys())[:10])

#densenet121_pretrained = models.densenet121()
#print(densenet121_pretrained)

#inceptionnet
class densenet121_representations(nn.Module):
     model_path = "/leonardo/home/userexternal/ldepaoli/densenet121-a639ec97.pth"
     def __init__(self): #images
          super().__init__() #refers to the class that this class inherits from (nn.Modules)
          self.densenet121_pretrained = models.densenet121()
          state_dict = torch.load(densenet121_representations.model_path)
          new_state_dict = {}
          for k, v in state_dict.items():
               new_key = re.sub(r'\.(\d+)', r'\1', k)
               new_state_dict[new_key] = v
          self.densenet121_pretrained.load_state_dict(new_state_dict)
          self.densenet121_pretrained.to("cuda")
          self.densenet121_pretrained.requires_grad_(False) #to not compute gradients with respect to model parameters
          self.hooks = []
          self.feature_maps = {}

          count = 0
          #for (name, layer) in alexnet.features.named_modules(): #this is useful to extract only the ReLU of the of the conv not of the classifier
          for (name, module) in self.densenet121_pretrained.named_modules():
               #print(name)
               #print(module)
               if name == "features.norm0" or name.endswith("denselayer1.norm2"):
                    module.name = f"bn_{count}"
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
          model = self.densenet121_pretrained(images)

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
