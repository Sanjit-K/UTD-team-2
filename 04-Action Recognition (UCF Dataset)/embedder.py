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

        pe = self.posEnc(num_frames, device=x.device)
        x +=  pe # automatically broadcasts
        return x
    
    def posEnc(self, seq_len, dim=512, device='cpu'):
        pe = torch.zeros(seq_len, dim, device=device)
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))  # (dim/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        return pe
