import torch 
import torch.nn as nn 
import math 
import os, sys 

# fixed sinusoidal PE "Attention is all you need"
class SinPositionalEmbedding(nn.Module): 

    def __init__(self, d_model, max_len=5000): 
        super(SinPositionalEmbedding, self).__init__() 
        # Compute the positional encodings once in log space. 
        pe = torch.zeros(max_len, d_model).float()              # (max_len (pos), d_model) 
        pe.require_grad = False 

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (max_len (pos), 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model // 2, )

        pe[:, 0::2] = torch.sin(position * div_term)            # even (max_len (pos), d_model)
        pe[:, 1::2] = torch.cos(position * div_term)            # odd 
        pe = pe.unsqueeze(0)                                    # (1, max_len (pos), d_model)

        self.register_buffer('pe', pe) 

    def forward(self, x): 
        """
        Args:
            x (tensor) : shape (batch_size, seq_len, in_dim)
        Return:
            self.pe (tensor) : shape (1, seq_len, d_model)
        """
        return self.pe[:, :x.size(1)] 