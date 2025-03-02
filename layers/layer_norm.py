import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) #scaling parameter
        self.beta = nn.Parameter(torch.zeros(d_model)) #shifting parameter
        self.eps = eps #epsilon value

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) #mean of the input tensor, -1 means the last dimension
        var = x.var(-1, unbiased=False, keepdim=True) #variance of the input tensor, unbiased=false means to use the biased estimator
        out = (x - mean) / torch.sqrt(var + self.eps) #normalizing the input tensor
        out = self.gamma * out + self.beta #scaling and shifting the normalized tensor
        return out