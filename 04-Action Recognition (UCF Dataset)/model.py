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
        self.mha1 = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 512)

        self.mha2 = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)

        self.mha3 = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm3 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)

        self.fc4 = nn.Linear(512, 512)
        self.norm4 = nn.LayerNorm(512)

    
    def forward(self, x):
        x = self.embed(x)
        x += self.pos_encoding(x)


        identity = x # block 1 (mha1)
        attn_output, _ = self.mha1(x, x, x)
        x = attn_output + identity
        x = self.norm1(x)
        x = self.fc1(x)

        identity = x # block 2 (mha2)
        attn_output, _ = self.mha2(x, x, x)
        x = attn_output + identity
        x = self.norm2(x)
        x = self.fc2(x)

        identity = x # block 3 (mha3)
        attn_output, _ = self.mha3(x, x, x)
        x = attn_output + identity
        x = self.norm3(x)
        x = self.fc3(x)

        identity = x # block 4 (final fully connected layer)
        x = self.fc4(x)
        x = x + identity
        x = self.norm4(x)

        return x


class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fc = nn.Linear(512, 101) # 101 classes
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1) # pooling layer over frames
        x = self.fc(x)
        
        return x



