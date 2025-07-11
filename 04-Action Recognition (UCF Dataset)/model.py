import torch.nn as nn
from embedder import embedNet, positionalEncoding

# CHANGED: # "For the base model, we use a rate of Pdrop = 0.1." (Vaswani et. al. 2017)
dropout=0.25
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048): 
        # d_ff = dimension of feedfoward network inner layer
        # "The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df_f = 2048." (Vaswani et. al. 2017)
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout1(attn_out) # "We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized." (Vaswani et. al. 2017)
        x = x + attn_out 
        x = self.norm1(x)
        

        ffn_out = self.ffn(x) 
        ffn_out = self.dropout2(ffn_out) # "We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized." (Vaswani et. al. 2017)
        x = x + ffn_out
        x = self.norm2(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.embed = embedNet()
        self.pos_encoding = positionalEncoding()
        self.dropout_embedding = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_encoding(x)
        x = self.dropout_embedding(x) # "We apply dropout to the sums of the embeddings and the positional encodings" (Vaswani et. al. 2017)
        for layer in self.layers:
            x = layer(x)
        return x
    
class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 101) # 101 classes
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1) # pooling layer over frames
        x = self.dropout(x)
        x = self.fc(x)

        return x



