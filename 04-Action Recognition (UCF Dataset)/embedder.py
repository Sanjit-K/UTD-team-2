import torch
import torch.nn as nn
import math
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class embedNet(nn.Module):
    def __init__(self):
        super().__init__()
        res_net = models.resnet50(weights='IMAGENET1K_V2')
        output_dim = res_net.fc.in_features
        res_net.fc = nn.Identity() # remove the final fully connected layer
        res_net.fc = nn.Linear(output_dim, 512) # replace with a new fully connected layer
        
        for param in res_net.parameters():
            param.requires_grad = False
        for param in res_net.fc.parameters():
            param.requires_grad = True
        self.model = res_net
    def forward(self,x):
        batch_size, num_frames, channels, height, width = x.shape

        x = x.view(batch_size * num_frames, channels, height, width) # multiply batch size with num frames to get total amount of images going to be used

        x = self.model(x)

        x = x.view(batch_size, num_frames, -1) # reshaped
     
        return x
    

class positionalEncoding(nn.Module):
    def __init__(self, max_len=1000, d_model=512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return self.pos_embedding[:, :seq_len, :]