from torch import nn

from embedding.positional_encoding import PositionalEncoding
from embedding.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):

        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model) #token embedding
        self.pos_emb = PositionalEncoding(d_model, max_len, device) #positional encoding
        self.drop_out = nn.Dropout(p=drop_prob) #dropout to minimise dependency on specific tokens

    def forward(self, x): #x - input tensor
        tok_emb = self.tok_emb(x) #getting token embedding
        pos_emb = self.pos_emb(x) #getting positional embedding
        return self.drop_out(tok_emb + pos_emb) #sum of token and pos embedding after dropout