import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------- Preprocess_Layer_STFT ---------------------------------------
class Preprocess_Layer_STFT(nn.Module):
    def __init__(self, 
           preprocess_layer_num,
           preprocess_filter_num,  # 为了保证d_model在之后能够与head整除,尽量保证filter_num为使用masked-attention的倍数
           preprocess_kernel_size):
        """
        used for FEMTO and XJTU dataset, 输入为STFT
        input.shape = [B, L, F_H, F_W]
        进入卷积层[B*L, 1, F_H, F_W], 输出卷积层[B*L, preprocess_filter_num, F_H, F_W]
        flatten后为: output.shape = [B, L, preprocess_filter_num*F_H*F_W]
        """
        super(Preprocess_Layer_STFT, self).__init__()

        preprocess_layer_list = []
        for i in range(0, preprocess_layer_num):
            if i == 0:
                in_channel = 1
            else:
                in_channel = preprocess_filter_num
            
            preprocess_layer_list.append(nn.Sequential(
                nn.Conv2d(in_channels = in_channel,
                     out_channels = in_channel,
                     kernel_size = (preprocess_kernel_size, preprocess_kernel_size),
                     stride = (1, 1),
                     padding = (0, 0),
                     groups = in_channel,
                     bias = False),
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(in_channel),

                nn.Conv2d(in_channels = in_channel,
                     out_channels = preprocess_filter_num,
                     kernel_size = (1, 1),
                     stride = (1, 1), 
                     padding = (0, 0),
                     bias = False), 
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(preprocess_filter_num),

                nn.ReLU(inplace = True)
            ))

        self.preprocess_layers = nn.ModuleList(preprocess_layer_list)



    def forward(self, x):
        B, L, F_H, F_W = x.shape
        x = x.view(B*L, F_H, F_W)
        x = x.unsqueeze(1)
        # print("input of conv: ",x.shape)
        for layer in self.preprocess_layers:
            x = layer(x)
            # print(x.shape)
        # print("output of conv: ", x.shape)
        # global average_pooling 
        BL, C, FH, FW = x.shape
        x = F.avg_pool2d(input = x, 
                 kernel_size = (FH, FW), 
                 stride = (1, 1))
        # print(x.shape)

        x = x.view(B, L, -1)
        return x



# -------------------------------- Preprocess_Layer_FC ---------------------------------------
class Preprocess_Layer_FC(nn.Module):
    def __init__(self, 
           input_feature,
           d_model):
        """
        used for CMAPSS, input.shape = [B, L, F]
        output.shape = [B, L, D], D = d_model
        """
        super(Preprocess_Layer_FC, self).__init__()
        self.fc = nn.Linear(input_feature, d_model, bias=True)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x



# -------------------------------- Preprocess_Layer_Conv ---------------------------------------
class Preprocess_Layer_Conv(nn.Module):
    def __init__(self, 
           preprocess_layer_num,
           preprocess_filter_num,  # 为了保证d_model在之后能够与head整除,尽量保证filter_num为使用masked-attention的倍数
           preprocess_kernel_size,
           preprocess_stride,
           input_feature,
           d_model,
           dropout=0.1):
        """
        "Depthwise conv + Pointwise conv"
        考虑到参数过大,我们要通过卷积提取input_length中每一点的input_feature之间的相关信息并降低维度需要很多层,
        防止层数过多导致参数过大, 采用Separable Convolution降低维度???
        input.shape = [B, L, F]   -->  [B, 1, L, F]
        卷积后为 [B, C, L, F']  -->  output.shape 
        =>线性层 [B, L, C*F'] = [B, L, D], C为Encoder中选用的masked-attention的种类个数
        """
        super(Preprocess_Layer_Conv, self).__init__()

        layer_sparse_conv = []
        for i in range(0, preprocess_layer_num):
            if i == 0:
                in_channel = 1
            else:
                in_channel = preprocess_filter_num
            
            layer_sparse_conv.append(nn.Sequential(
                nn.Conv2d(in_channels = in_channel,
                     out_channels = in_channel,
                     kernel_size = (1, preprocess_kernel_size),
                     stride = (1, preprocess_stride),
                     padding = (0, int(preprocess_kernel_size/2)),
                     groups = in_channel,
                     bias = False),
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(in_channel),

                nn.Conv2d(in_channels = in_channel,
                     out_channels = preprocess_filter_num,
                     kernel_size = (1, 1),
                     stride = (1, 1), 
                     padding = (0, 0),
                     bias = False), 
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(preprocess_filter_num),

                nn.ReLU(inplace = True)
            ))
        self.layer_sparse_conv = nn.ModuleList(layer_sparse_conv)

        self.conv_out_fea = self._get_conv_out_fea(input_feature)  # 计算卷积后的维度
        # print(self.conv_out_fea)

        self.linear = nn.Linear(self.conv_out_fea, d_model)
        self.activation = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(dropout)

    def _get_conv_out_fea(self, input_feature):
        # 计算输出参数
        o = torch.zeros(1, 1, 1, input_feature)
        for layer in self.layer_sparse_conv:
            o = layer(o)           
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layer_sparse_conv:
            x = layer(x)

        # global avgpooling
        # B, C, L, Fea = x.shape
        # x = F.avg_pool2d(input = x, 
        #          kernel_size = (1, Fea), 
        #          stride = (1, 1))
        # print(x.shape)
        x = x.permute(0, 2, 1, 3).contiguous()
        size = x.size()
        x = x.view(*size[:2], -1)

        x = self.activation(self.linear(x))
        x = self.dropout(x)
        
        return x       