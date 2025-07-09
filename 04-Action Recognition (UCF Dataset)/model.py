# import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from embedder import embedNet

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = embedNet()
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 512)

        self.norm2 = nn.LayerNorm(512)
        
    
    def forward(self, x):
        x = self.embed(x)
        identity = x
        attn_output, _ = self.mha(x, x, x)
        x = attn_output + identity
        x = self.norm1(x)

        identity = x
        x = self.fc(x)
        x = x + identity
        x = self.norm2(x)

        return x


class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fc = nn.Linear(512, 101) # 101 classes
        self.softmax = nn.Softmax()
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1) # pooling layer over frames
        x = self.fc(x)
        x = self.softmax(x)
        return x



