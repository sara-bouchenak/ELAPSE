import torch.nn as nn
import torch


class MLPModel(nn.Module):
    def __init__(self, nb_features, nb_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(nb_features, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)           # Second fully connected layer
        self.fc3 = nn.Linear(32, nb_classes)   # Output layer


    def forward(self, x, last=False, freeze=False):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        score = self.fc3(x)
        if last:
            return score, x
        else:
            return score

    def get_embedding_dim(self):
        return 32