import torch
import torch.nn as nn
import math
from torchvision import models


class embedNet(nn.Module):
    def __init__(self):
        super().__init__()
        res_net = models.resnet50(weights='IMAGENET1K_V2')
        output_dim = res_net.fc.in_features
        res_net.fc = nn.Identity() # remove the final fully connected layer
        res_net.fc = nn.Linear(output_dim, 512) # replace with a new fully connected layer
        
        for param in res_net.parameters(): # freeze all parameters in the res_net (does not need training)
            param.requires_grad = False
        for param in res_net.fc.parameters(): # unfreeze the parameters in the new fully connected layer (this layer needs training)
            param.requires_grad = True
        self.model = res_net
    def forward(self,x):
        batch_size, num_frames, channels, height, width = x.shape

        x = x.view(batch_size * num_frames, channels, height, width) # multiply batch size with num frames to get total amount of images going to be used

        x = self.model(x)

        x = x.view(batch_size, num_frames, -1) # reshaped

        pe = self.posEnc(num_frames) 
        x +=  pe # automatically broadcasts
        return x
    
    def posEnc(self, seq_len, dim=512): # seq_len = num_frames
        pe = torch.zeros(seq_len, dim)

        position = torch.arange(0, dim, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[0::2] = torch.sin(position[0::2] * div_term)
        pe[1::2] = torch.cos(position[1::2] * div_term)
        return pe