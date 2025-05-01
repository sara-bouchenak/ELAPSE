import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class ColoredDataset(Dataset):
    def __init__(self, dataset, classes=None, colors=[0, 1], std=0.3):
        self.mnist = dataset
        self.colors = colors

        if isinstance(colors, torch.Tensor):
            self.colors = colors
        elif isinstance(colors, list):
            self.colors = torch.Tensor(classes, 3, 1, 1).uniform_(colors[0], colors[1])
        else:
            raise ValueError('Unsupported colors!')
        
        dataset_length = len(self.mnist)
        np.random.seed(42)
        self.color_flags = random.choices([0, 1], [0.3, 0.7], k=dataset_length)
        self.perturb = std * torch.randn(len(self.mnist), 3, 1, 1)
        # np.savetxt('color_flags.txt', self.color_flags)
        # self.color_flags = np.loadtxt('color_flags.txt')
    
    def __len__(self):
        return len(self.mnist)
    

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if label % 2 == 0:
            label = 1  
        else:
            label = 0  

        flag = self.color_flags[idx]
        if flag == 0:
            color_img = (self.colors[label] + self.perturb[idx]).clamp(0, 1) * img
        else:
            color_img = img.expand(3, -1, -1) 

        return color_img, label, flag
    

    @property
    def targets(self):
        # Return the targets from the underlying dataset
        return self.mnist.targets