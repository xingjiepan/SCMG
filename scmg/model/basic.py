import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(256,),
                 dropout_prob=0.05,
                ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
            
        self.layers.append(nn.Linear(prev_dim, output_dim))
                
    def forward(self, X):        
        for layer in self.layers:
            X = layer(X)
            
        return X