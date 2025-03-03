import torch
from torch import nn

from blocks.decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device): 
        super().__init__()
        # embedding layer
        self.emb = TransformerEmbedding(d_model=d_model, 
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        # decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        # LM head
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        # pass through decoder layers
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output