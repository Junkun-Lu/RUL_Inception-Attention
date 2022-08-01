import torch.nn as nn
from Model.Multi_Head import MultiHeadAttention
from Model.Res_and_Layernorm import PoswiseFeedForwardNet

# --------------------------------------------------- Encoder Layer --------------------------------------
"""
A layer of Encoder
enc_inputs: [batch_size, max_len, d_model]
enc_outputs: [batch_size, max_len, d_model]
"""
class EncoderLayer(nn.Module):
  def __init__(self, d_model,
            d_k, d_v,
            n_heads_full, n_heads_log, n_heads_local,
            n_heads_prob, n_heads_FFT, n_heads_auto,
            moving_avg, dropout, d_ff, device):
    super(EncoderLayer, self).__init__()

    self.enc_self_attn = MultiHeadAttention(d_model,
                          d_k, d_v,
                          n_heads_full, n_heads_log, n_heads_local,
                          n_heads_prob, n_heads_FFT, n_heads_auto,
                          moving_avg, dropout, device).double()
    self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, device).double()

  def forward(self, enc_inputs):
    enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
    enc_outputs = self.pos_ffn(enc_outputs)
    return enc_outputs


# ------------------------------------ Encoder -----------------------------------------------------
"""
Encoder of Inception-Attention
enc_outputs: [batch_size, max_len, d_model]
"""
class Encoder(nn.Module):
  def __init__(self, d_model,
            d_k, d_v,
            n_heads_full, n_heads_log, n_heads_local,
            n_heads_prob, n_heads_FFT, n_heads_auto,
            moving_avg, dropout, d_ff, n_layers, device):
    super(Encoder, self).__init__()

    self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads_full, n_heads_log, n_heads_local,
                          n_heads_prob, n_heads_FFT, n_heads_auto, moving_avg, dropout, d_ff, device)
                  for _ in range(n_layers)])

  def forward(self, enc_inputs):
    for layer in self.layers:
      enc_outputs = layer(enc_inputs)

    return enc_outputs