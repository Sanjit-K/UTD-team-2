import torch, math
import torch.nn as nn
from torchvision import models
from devices import device

class Embed(nn.Module):
    def __init__(self):
        super(Embed, self).__init__()
        resNet = models.resnet50(weights='IMAGENET1K_V2').to(device)
        output_dim = resNet.fc.in_features
        resNet.fc = nn.Identity()
        resNet.fc = nn.Linear(output_dim, 512)

        for param in resNet.parameters():
            param.requires_grad = False
        for param in resNet.fc.parameters():
            param.requires_grad = True
        self.model = resNet

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)

        x = self.model(x).view(batch_size, num_frames, -1)
        return x

# From Pytorch documentation (W)
class pEncoding(nn.Module):
    def __init__(self, max_len=1000, d_model=512):
        super(pEncoding, self).__init__()
        self.pA = torch.zeros(max_len, d_model, device=device)
        self.position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pA[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pA[:, 1::2] = torch.cos(self.position * self.div_term)
        self.pA = self.pA.unsqueeze(0)
        self.register_buffer('pb', self.pA)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pA[:, :seq_len, :]
