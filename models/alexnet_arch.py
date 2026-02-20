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

#alexnet_pretrained = models.alexnet()
#print(alexnet_pretrained)

layer_indices = [0, 3, 6, 8, 10]

#alexnet
class alexnet_representations(nn.Module):
     model_path = "/leonardo/home/userexternal/ldepaoli/models/alexnet-owt-7be5be79.pth"
     def __init__(self, selected_idx: int): #images
          super().__init__() #refers to the class that this class inherits from (nn.Modules)
          self.selected_idx = int(selected_idx)
          self.alexnet_pretrained = models.alexnet()
          self.alexnet_pretrained.eval()
          state_dict = torch.load(alexnet_representations.model_path)
          self.alexnet_pretrained.load_state_dict(state_dict)
          self.alexnet_pretrained.to("cuda")
          self.alexnet_pretrained.requires_grad_(False) #to not compute gradients with respect to model parameters
          self.feature_maps = {}     
          self._hooks = []

          for idx, layer in enumerate(self.alexnet_pretrained.features):
               if isinstance(layer, nn.Conv2d) and (idx in layer_indices):
                    layer._hook_idx = idx
                    h = layer.register_forward_hook(self._hook_func)
                    self._hooks.append(h)

     def _hook_func(self, module, inp, out):
          idx = module._hook_idx
          self.feature_maps[idx] = out.detach()

     def gram_matrix(self, feature_map: torch.Tensor) -> torch.Tensor:
          return torch.einsum("bihw,bjhw->bij", feature_map, feature_map)

     def forward(self, images: torch.Tensor):
          # reset per forward
          self.feature_maps = {}

          _ = self.alexnet_pretrained(images)

          grams_list = []
          m_list = []
          feature_maps_list = []

          # deterministic order
          for idx in layer_indices:
               fmap = self.feature_maps[idx]  # (B,C,H,W)
               feature_maps_list.append(fmap)
               m_list.append(int(fmap.size(2) * fmap.size(3)))
               grams_list.append(self.gram_matrix(fmap))

          # selected outputs
          sel_fmap = self.feature_maps[self.selected_idx]
          sel_gram = self.gram_matrix(sel_fmap)

          return grams_list, m_list, feature_maps_list, sel_gram, sel_fmap

def gaussian_image_tensor(size=400, mean=0.5, std=0.2):
    gaussian_image = torch.randn(3, size, size) * std + mean
    gaussian_image.clamp_(0.0, 1.0)
    return gaussian_image.to("cuda")#.unsqueeze(0)
