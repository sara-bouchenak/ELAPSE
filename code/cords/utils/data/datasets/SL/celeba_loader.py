from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms




class celeba(Dataset):
    def __init__(self, img_dir, attr_df, transform=None):
        self.img_dir = img_dir
        self.attr_df = attr_df
        self.transform = transform

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = self.attr_df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Extract attributes, e.g., smile, gender, etc.
        attrs = self.attr_df.iloc[idx, 1:].values.astype(int)
        sa = attrs[[20, 39]]  # sensitive attributes
        label = attrs[31]     # label

        return image, label, sa

class MyImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, attrs = self.dataset[index]
        sa = np.array([attrs[20], attrs[39]])  # Sensitive attributes
        y = attrs[31]  # Label (e.g., smile attribute)
        assert y in [0, 1], f"Label out of range: {y}"
        return img, y, sa

    def __len__(self):
        return len(self.dataset)
