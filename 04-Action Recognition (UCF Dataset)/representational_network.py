import torch
import torch.nn as nn
import math
from torchvision import models

class embedNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        out_dim = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) #remove last 1 layers
        for param in resnet.parameters(): # freeze all parameters in the res_net (does not need training)
            param.requires_grad = False
        self.out_dim = out_dim
        self.proj = nn.Linear(out_dim, 512)
        
    def forward(self, x):
        # batch, rgb channels, frames, height, width
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)  #batch * frames, rbg, h, w
        features = self.backbone(x)
        _, out_dim, h_p, w_p = features.shape
        features = features.view(b, t, out_dim, h_p, w_p)
        features = features.permute(0, 1, 3, 4, 2).reshape(b, t * h_p * w_p, out_dim)  # (batch, tokens, out_dim)
        features = self.proj(features)  # (batch, tokens, 512)
        return features

class posEnc(nn.Module):
    def __init__(self, max_len=1000, d_model=512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return self.pos_embedding[:, :seq_len, :]