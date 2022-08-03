import torch
import torch.nn as nn
from Model.CNN_Layer_CMAPSS import CNN_Layers
from Model.Encoder import Encoder
from Model.Predictor import FullPredictor, LinearPredictor, ConvPredictor


"""
This Model only used for CMAPSS dataset, Compare to other datasets we used. 
We used special CNN_Layer_CMAPSS Module get richer features from raw input data.
"nb_conv_layers", "filter_num" and "filter_size" are the variable used for CNN_Layers_CMAPSS.

input of model: [batch_size, max_len, feature_num]
output of model: [batch_size, max_len]
"""
class Inception_Attention(nn.Module):
  def __init__(self,dropout,
                    d_model,
                    max_len,
                    d_k,
                    d_v,

                    n_heads_full,
                    n_heads_log,
                    n_heads_local,
                    n_heads_prob,
                    n_heads_FFT,
                    n_heads_auto,

                    moving_avg,
                    d_ff,
                    n_layers,
                    device,

                    predictor_type):


        #  nb_conv_layers,
        #  filter_num,
        #  filter_size,
        #  dataset_name

    super(Inception_Attention, self).__init__()

    self.predictor_type = predictor_type   # choose which predictor used

    self.encoder = Encoder(d_model, d_k, d_v,
                  n_heads_full, n_heads_log, n_heads_local,
                  n_heads_prob, n_heads_FFT, n_heads_auto,
                  moving_avg, dropout, d_ff, n_layers, device).double()
    # output:[batch, max_len, d_model]


    if self.predictor_type == "full":
        self.predictor = FullPredictor(max_len, d_model).double()
    if self.predictor_type == "linear":
        self.predictor = LinearPredictor(d_model).double()
    if self.predictor_type =="conv":
        self.predictor = ConvPredictor(d_model).double()
    if self.predictor_type == "hybrid":
        self.predictor1 = FullPredictor(max_len, d_model).double()
        self.predictor2 = LinearPredictor(d_model).double()
        self.predictor3 = ConvPredictor(d_model).double()
        self.predictor = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3).double()
    # output:[batch_size, max_len]

  def forward(self, enc_inputs):

    enc_outputs = self.encoder(enc_inputs)     # output of Attention_Layers

    if self.predictor_type == "hybrid":      # output of Predictor
        x_full = self.predictor1(enc_outputs)
        if len(x_full.shape) == 1:
            x_full = torch.unsqueeze(x_full, 0)
        x_full = torch.unsqueeze(x_full, 2)

        x_linear = self.predictor2(enc_outputs)
        if len(x_linear.shape) == 1:
            x_linear = torch.unsqueeze(x_linear, 0)  
        x_linear = torch.unsqueeze(x_linear, 2)

        x_conv = self.predictor3(enc_outputs)
        if len(x_conv.shape) == 1:
            x_conv = torch.unsqueeze(x_conv, 0) 
        x_conv = torch.unsqueeze(x_conv, 2)

        x_hybrid = torch.cat([x_full, x_linear, x_conv], dim=-1)
        x_hybrid = nn.functional.pad(x_hybrid.permute(0, 2, 1), pad=(1, 1), mode='replicate')
        enc_outputs = self.predictor(x_hybrid).permute(0, 2, 1).squeeze()
    else:
        enc_outputs = self.predictor(enc_outputs)

    if len(enc_outputs.shape) == 1:  # prevent output.shape=[L], wenn B = 1
        enc_outputs = torch.unsqueeze(enc_outputs, 0)  

    return enc_outputs