# import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from embedder import embedNet, positionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048): 
        # d_ff = dimension of feedfoward network inner layer
        # The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df_f = 2048. (Vaswani et. al. 2017)
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.embed = embedNet()
        self.pos_encoding = positionalEncoding()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
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



