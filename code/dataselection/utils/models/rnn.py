import torch
import torch.nn as nn 
from collections import OrderedDict 
import numpy as np 



class LSTM_MFCC(nn.Module):
    """
    2x BiLSTM sur MFCC; pooling temporel moyen; FC->2
    Entrée: [B, T, C]
    """
    def __init__(self, in_dim: int = 40, hidden: int = 128, layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True, dropout=0.2  # <-- dropout intra-LSTM pour stabiliser
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden, num_classes)

    def forward(self, x, last=False, freeze=False):  # [B, T, C]
        if freeze:
            with torch.no_grad():
                x, _ = self.lstm(x)         # [B, T, 2H]
                x = x.mean(dim=1)           # [B, 2H]
                x = self.drop(x)
        else:
            x, _ = self.lstm(x)         # [B, T, 2H]
            x = x.mean(dim=1)           # [B, 2H]
            x = self.drop(x)

        out = self.fc(x)
        if last:
            return out, x
        else:
            return out
        

        x, _ = self.lstm(x)         # [B, T, 2H]
        x = x.mean(dim=1)           # [B, 2H]
        x = self.drop(x)
        return self.fc(x)


    def get_embedding_dim (self) : 
        return 256 
    

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        self.embDim = 64
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


    def get_embedding_dem(self) : 
        return self.embDim 