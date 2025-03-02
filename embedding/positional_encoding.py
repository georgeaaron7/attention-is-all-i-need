import torch
from torch import nn

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, max_len, device): #d_model - dimension of model, max_len - max seq len    

        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device, requires_grad=False) #tensor with zeros of size max len and d_model, not trainable

        pos = torch.arange(0, max_len, device=device) #creating tensor with values from 0 to max_len
        pos = pos.float().unsqueeze(dim=1) #converting to float and adding a dimension (1d to 2d)

        _2i = torch.arange(0, d_model, step=2, device=device).float() #creating tensor with values from 0 to d_model with step 2
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))) #sin values for the even index of the encoding matrix
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))) #cos values for the odd index of the encoding matrix

    def forward(self, x):
        seq_len = x.size(1) #getting sequence length
        return self.encoding[:seq_len, :] #returning positional encoding for given sequence length