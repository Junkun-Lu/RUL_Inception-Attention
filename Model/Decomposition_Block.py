import torch
import torch.nn as nn


"""
Decomposition Block use after AutoCorrelation,
Divide the seasonal and trend information by Decomposition Block
input_size = [batch_size, max_len, d_model]
output = seasonal_information:  [batch_size, max_len, d_model]
"""

# -------------------------------- Moving average block --------------------------------------
# Moving average block to highlight the trend of time series

class Moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


# ------------------------ series decomposition block ---------------------------------------------
# res = seasonal information; moving_mean = trend information.

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = Moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean