from torch.utils.data import Dataset
import numpy as np


class MyImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, attrs = self.dataset[index]
        sa = np.array([attrs[20], attrs[39]])
        y = attrs[31]
        return img, y, sa#torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.dataset)


