import torch
import torch.nn as nn
from embed import Embed, pEncoding

class TransformerModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention sublayer
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        # Feed-forward sublayer
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        return x

class Encode(nn.Module):
    def __init__(self, num_layers=4, embed_dim=512):
        super(Encode, self).__init__()
        self.embed = Embed()  # extracts per-frame features: (batch, timesteps, 512)
        self.pos_encoding = pEncoding(max_len=1000, d_model=embed_dim)
        # Stack transformer encoder blocks
        self.layers = nn.ModuleList([TransformerModule(embed_dim=embed_dim) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embed(x)
        # Add positional encoding
        x = x + self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ANet(nn.Module):
    def __init__(self, num_classes=101):
        super(ANet, self).__init__()
        self.encoder = Encode(num_layers=4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Process each frame via encoder
        x = self.encoder(x)  # shape (batch, timesteps, 512)
        # Temporal average pooling over frames
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
