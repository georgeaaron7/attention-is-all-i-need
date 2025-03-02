from torch import nn
from layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) #passing q, k, v through linear layers
        q, k, v = self.split(q), self.split(k), self.split(v) #splitting q, k, v into n_head number of heads
        out, attention = self.attention(q, k, v, mask) #passing q, k, v to attention mechanism

        out = self.concat(out)
        out = self.w_concat(out)
        return out, attention

    def split(self, tensor): #splits the input tensor into n_head number of heads
        batch_size, length, d_model = tensor.size()

        if d_model % self.n_head != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by n_head ({self.n_head})')

        d_tensor = d_model // self.n_head

        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor): #concatenates the input tensor
        batch_size, head, length, d_tensor = tensor.size()

        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor