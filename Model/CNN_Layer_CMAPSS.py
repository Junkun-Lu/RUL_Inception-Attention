import torch.nn as nn

"""
This module only used for dataset CMAPSS

CNN layers are used before the Inception-Attention network architecture to extract features from the raw data.

input: [batch_size, max_len, feature_num]
output: [batch_size, max_len, d_model]

nb_conv_layers == the layer number of CNN_Layers
filter_num == out_channels number
filter_size = kernel_size[0]
"""

class CNN_Layers(nn.Module):
    def __init__(self, nb_conv_layers, filter_num, filter_size):
        super(CNN_Layers, self).__init__()

        # ------------- Part 1, Channel wise Feature Extraction -------------------------------

        layers_conv = []
        for i in range(nb_conv_layers):
            if i == 0:
                in_channel = 1
            else:
                in_channel = filter_num

            layers_conv.append(nn.Sequential(
                nn.Conv2d(in_channels = in_channel,
                     out_channels = filter_num,
                     kernel_size = (filter_size, 1),
                     stride = (1, 1),
                     padding = (int(filter_size / 2), 0)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num)
            ))
        self.layers_conv = nn.ModuleList(layers_conv)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers_conv:
            x = layer(x)

        x = x.permute(0, 2, 1, 3).contiguous()
        size = x.size()
        x = x.view(*size[:2], -1)
        return x