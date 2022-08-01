import torch.nn as nn

"""
consider the time correlation, so use the causal Convolution to calculate the queries and keys.
Prevent the later time-point information affect the current time point information.
"""

# -------------------- Chomp1d ------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()