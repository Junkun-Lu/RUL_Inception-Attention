import torch.nn as nn
import torch.nn.functional as F


class conv_bn(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, activation="relu"):
        super(conv_bn, self).__init__()
        self.kernel_size = kernel
        self.donwconv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        padding_size = int(self.kernel_size / 2)
        padding_x = nn.functional.pad(x, pad=(padding_size, padding_size), mode='replicate')
        enc_out = self.norm(self.activation(self.donwconv(padding_x)))
        return enc_out


# ----------------------- FullPredictor ----------------------------------------
"""
input = Encoder_output
input = [batch_size, max_len, d_model]
output = [batch_size, max_len]
"""
class FullPredictor(nn.Module):
    def __init__(self, max_len, d_model):
        super(FullPredictor, self).__init__()

        self.conv_bn1 = conv_bn(input_dim=d_model, output_dim=int(d_model / 2), kernel=3)
        self.conv_bn2 = conv_bn(input_dim=int(d_model / 2), output_dim=1, kernel=3)

        self.predict = nn.Linear(in_features=max_len, out_features=max_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = x.permute(0, 2, 1).squeeze()
        
        enc_out = self.predict(x)
        return enc_out


# ---------------------------- LinearPredictor ----------------------------------------
"""
input = Encoder_output
input = [batch_size, max_len, d_model]
output = [batch_size, max_len]
"""
class LinearPredictor(nn.Module):
    def __init__(self, d_model):
        super(LinearPredictor, self).__init__()

        self.predict1 = nn.Linear(d_model, int(d_model / 2), bias=True)
        self.activation1 = F.relu
        self.predict2 = nn.Linear(int(d_model / 2), int(d_model / 4), bias=True)
        self.activation2 = F.relu

        self.predict = nn.Linear(int(d_model / 4), 1, bias=True)

    def forward(self, x):
        enc_out = self.activation1(self.predict1(x))
        enc_out = self.activation2(self.predict2(enc_out))
        enc_out = self.predict(enc_out).squeeze()
        return enc_out


# ------------------------------- ConvPredictor -------------------------------------
"""
input = Encoder_output
input = [batch_size, max_len, d_model]
output = [batch_size, max_len]
"""
class ConvPredictor(nn.Module):
    def __init__(self, d_model, pred_kernel=3):
        super(ConvPredictor, self).__init__()
        self.pred_kernel = pred_kernel

        self.conv_bn1 = conv_bn(input_dim=d_model, output_dim=int(d_model / 2), kernel=3)

        self.conv_bn2 = conv_bn(input_dim=int(d_model / 2), output_dim=int(d_model / 4), kernel=3)

        self.predict = nn.Conv1d(in_channels=int(d_model / 4), out_channels=1, kernel_size=pred_kernel)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)

        padding_size = int(self.pred_kernel / 2)
        paddding_x = nn.functional.pad(x, pad=(padding_size, padding_size), mode='replicate')
        enc_out = self.predict(paddding_x).permute(0, 2, 1).squeeze()
        return enc_out