import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Attention_Family import Full_Attention, Local_Attention, LocalLog_Attention, ProbSparse_Attention, Auto_Attention, FFT_Attention
from Model.Attention_Layer import Attention_Layer






# ------------------------------------------------- Encoder Layer -----------------------------------
class Encoder_Layer(nn.Module):
    def __init__(self,
           attention_layer_types,
           d_model,
           n_heads_full,
           n_heads_local,
           n_heads_log,
           n_heads_prob,
           n_heads_auto,
           n_heads_fft = 1,
           d_keys = None,
           d_values = None,
           d_ff = None,
           dropout = 0.1,
           activation = 'relu',
           forward_kernel_size = 1,
           value_kernel_size = 1,
           causal_kernel_size = 3,
           output_attention = True,
           auto_moving_avg = 25,):
        """
        attention_layer_type: list类型,["Full", "Local", "Log", "Prob", "Auto", "FFT"]
        d_model: [B, L, D], 总的隐藏层数量, 为masked_attention的类别数量的倍数
        n_heads_full: type为Full_Attention的head数
        n_heads_local: type为local_attention的head数
        n_heads_log: type为locallog_attention的head数
        n_heads_prob: type为probsparse_attention的head数
        n_heads_auto: type为auto_attention的head数
        n_heads_fft: type为fft_attention的head数, 只有一个head
        d_ff: 最后forward时,中间过程的维度. 可大可小,如果没有被定义的话,那么就是d_model的两倍.
  
        """
        super(Encoder_Layer, self).__init__()

        n_heads = n_heads_full + n_heads_local + n_heads_log + n_heads_prob + n_heads_auto + n_heads_fft
        d_model_each_head = int(d_model / n_heads)

        # self.n_heads_full = n_heads_full
        # self.n_heads_local = n_heads_local
        # self.n_heads_log = n_heads_log
        # self.n_heads_prob = n_heads_prob
        # self.n_heads_auto = n_heads_auto
        # self.n_heads_fft = n_heads_fft
        
        ######################## 第一部分, 进行attention ########################
        attention_layer_list = []   # 将所有的masked-attention添加到里面
        for type_attention in attention_layer_types:
            if type_attention == "Full":
                attention_layer_list.append(Attention_Layer(attention = Full_Attention(mask_flag=False, 
                                      attention_dropout=dropout, 
                                      output_attention=output_attention),
                                      input_dim = d_model,
                                      output_dim = d_model_each_head*n_heads_full,
                                      type_attention = type_attention,
                                      d_model_type = d_model_each_head*n_heads_full,
                                      n_heads_type = n_heads_full,
                                      d_keys = d_keys,
                                      d_values = d_values,
                                      causal_kernel_size = causal_kernel_size,
                                      value_kernel_size = value_kernel_size,
                                      resid_drop = dropout,
                                      auto_moving_avg = auto_moving_avg))
            if type_attention == "Local":
                attention_layer_list.append(Attention_Layer(attention = Local_Attention(mask_flag=True, 
                                                     attention_dropout=dropout, 
                                                     output_attention=output_attention),
                                      input_dim = d_model,
                                      output_dim = d_model_each_head*n_heads_local,
                                      type_attention = type_attention,
                                      d_model_type = d_model_each_head*n_heads_local,
                                      n_heads_type = n_heads_local,
                                      d_keys = d_keys,
                                      d_values = d_values,
                                      causal_kernel_size = causal_kernel_size,
                                      value_kernel_size = value_kernel_size,
                                      resid_drop = dropout,
                                      auto_moving_avg = auto_moving_avg))   
            if type_attention == "Log":
                attention_layer_list.append(Attention_Layer(attention = LocalLog_Attention(mask_flag=True, 
                                                      attention_dropout=dropout, 
                                                      output_attention=output_attention),
                                      input_dim = d_model,
                                      output_dim = d_model_each_head*n_heads_log,
                                      type_attention = type_attention,
                                      d_model_type = d_model_each_head*n_heads_log,
                                      n_heads_type = n_heads_log,
                                      d_keys = d_keys,
                                      d_values = d_values,
                                      causal_kernel_size = causal_kernel_size,
                                      value_kernel_size = value_kernel_size,
                                      resid_drop = dropout,
                                      auto_moving_avg = auto_moving_avg))
            if type_attention == "Prob":
                attention_layer_list.append(Attention_Layer(attention = ProbSparse_Attention(mask_flag=True,
                                                       factor=5, 
                                                       scale=None,
                                                       attention_dropout=dropout,
                                                       output_attention=output_attention),
                                      input_dim = d_model,
                                      output_dim = d_model_each_head*n_heads_prob,
                                      type_attention = type_attention,
                                      d_model_type = d_model_each_head*n_heads_prob,
                                      n_heads_type = n_heads_prob,
                                      d_keys = d_keys,
                                      d_values = d_values,
                                      causal_kernel_size = causal_kernel_size,
                                      value_kernel_size = value_kernel_size,
                                      resid_drop = dropout,
                                      auto_moving_avg = auto_moving_avg))
            if type_attention == "Auto":
                attention_layer_list.append(Attention_Layer(attention = Auto_Attention(mask_flag=True,
                                                    factor=5, 
                                                    scale=None,
                                                    attention_dropout=dropout,
                                                    output_attention=output_attention),
                                      input_dim = d_model,
                                      output_dim = d_model_each_head*n_heads_auto,
                                      type_attention = type_attention,
                                      d_model_type = d_model_each_head*n_heads_auto,
                                      n_heads_type = n_heads_auto,
                                      d_keys = d_keys,
                                      d_values = d_values,
                                      causal_kernel_size = causal_kernel_size,
                                      value_kernel_size = value_kernel_size,
                                      resid_drop = dropout,
                                      auto_moving_avg = auto_moving_avg))  
            if type_attention == "FFT":
                 attention_layer_list.append(Attention_Layer(attention = FFT_Attention(),
                                       input_dim = d_model,
                                       output_dim = d_model_each_head*n_heads_fft,
                                       type_attention = type_attention,
                                       d_model_type = d_model_each_head*n_heads_fft,
                                       n_heads_type = n_heads_fft,
                                       d_keys = d_keys,
                                       d_values = d_values,
                                       causal_kernel_size = causal_kernel_size,
                                       value_kernel_size = value_kernel_size,
                                       resid_drop = dropout,
                                       auto_moving_avg = auto_moving_avg))  
        # 整合所有的mask_attention
        self.attention_layer_list = nn.ModuleList(attention_layer_list)

        # layernorm
        self.norm1 = nn.LayerNorm(d_model)


        ######################## 第二部分, 线性层 ########################
        d_ff = d_ff or int(d_model*2)
        self.forward_kernel_size = forward_kernel_size

        self.conv1 = nn.Conv1d(in_channels = d_model,
                    out_channels = d_ff,
                    kernel_size = self.forward_kernel_size)
        self.activation1 = F.relu if activation == "relu" else F.gelu

        self.conv2 = nn.Conv1d(in_channels = d_ff,
                    out_channels = d_model,
                    kernel_size = self.forward_kernel_size)
        self.activation2 = F.relu if activation == "relu" else F.gelu
        
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        # x.shape = [B, L, D]
        outs_list, attns_list = [], []
        for attention_layer in self.attention_layer_list:
            out_value, out_attn = attention_layer(x, x, x)
            outs_list.append(out_value)
            attns_list.append(out_attn)
            # print(out_value.shape)

        # 这里需要讨论是否是Auto,是否需要做decomp position


        # 每个attention进行数据融合
        new_x = torch.cat(outs_list, dim=-1)

        # residual
        x = x + new_x

        forward_padding_size = int(self.forward_kernel_size/2)
        
        y = x = self.norm1(x)    # [B, L, D]

        padding_y1 = nn.functional.pad(y.permute(0, 2, 1),
                        pad = (forward_padding_size, forward_padding_size),
                        mode = 'replicate')
        y = self.dropout(self.activation1(self.conv1(padding_y1)))

        padding_y2 = nn.functional.pad(y,
                        pad = (forward_padding_size, forward_padding_size),
                        mode = 'replicate')
        y = self.dropout(self.activation2(self.conv2(padding_y2).permute(0, 2, 1)))

        y = self.norm2(x + y)

        return y, attns_list 


# ----------------------------- Encoder -------------------------------------
class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        attns_list = []

        for encoder_layer in self.encoder_layers:
            x, attn_list = encoder_layer(x)
            attns_list.append(attn_list)
        
        return x, attns_list