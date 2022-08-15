import torch.nn as nn
from Model.Attention_Family import series_decomp


"""
该文件为每一个attention机制的计算
对于FFT模块不需要计算QKV,且只有一个头; 对于Auto模块需要增加decomposition block
"""
class Attention_Layer(nn.Module):
    def __init__(self,
           attention,
           input_dim,        
           output_dim,
           type_attention,
           d_model_type,
           n_heads_type,
           d_keys = None,
           d_values = None,
           causal_kernel_size = 3,
           value_kernel_size = 1,
           resid_drop = 0.1,
           auto_moving_avg = 25):
        """
        attention: 选择什么类型的masked-attention;
        input_dim: Encoder_Module输入数据的shape-[B, L, D]中的"D";
        output_dim: 假设我们有n类masked_attention, output_dim = D/n
        d_model_type = output_dim
        n_heads_type: 当前选择的masked_attention有多少个head,注意FFT_Attention只有一个head
        d_keys, d_values = d_model_type/n_heads_type
        """
        super(Attention_Layer, self).__init__()
        # d_keys, d_values, 每个head中key与value的维度
        self.d_keys = d_keys or int(d_model_type/n_heads_type)
        self.d_values = d_values or int(d_model_type/n_heads_type)
        # 当前masked-attention有多少个head
        self.n_heads_type = n_heads_type
        self.type_attention = type_attention

        # causal convolution 计算queries, keys, values
        self.causal_kernel_size = causal_kernel_size
        self.value_kernel_size= value_kernel_size   # 用于计算values, 选择为1

        if self.type_attention != "FFT":
            self.query_projection = nn.Conv1d(in_channels   = input_dim, 
                                              out_channels  = self.d_keys * self.n_heads_type, 
                                              kernel_size   = self.causal_kernel_size)
            
            self.key_projection = nn.Conv1d(in_channels     = input_dim, 
                                            out_channels    = self.d_keys * self.n_heads_type, 
                                            kernel_size     = self.causal_kernel_size)
            
            self.value_projection = nn.Conv1d(in_channels   = input_dim, 
                                              out_channels  = self.d_values * self.n_heads_type, 
                                              kernel_size   = self.value_kernel_size) 

        # fft_projection 专门用于FFT计算,得到fft层的input
        if self.type_attention == "FFT":
            self.fft_projection = nn.Conv1d(in_channels     = input_dim,
                                            out_channels    = output_dim,
                                            kernel_size     = self.value_kernel_size)

        # 选择当前使用的attention type
        self.inner_attention = attention

        # out_projection
        self.out_projection = nn.Conv1d(in_channels     = self.d_values * self.n_heads_type,
                                        out_channels    = output_dim,
                                        kernel_size     = self.value_kernel_size)
        self.activation = nn.ReLU(inplace=True)
        self.resdi_dropout = nn.Dropout(resid_drop) 

        # decomp block
        """这个是和muti_head组合在一起发生的,通过Auto-correlation得到数据的out_value, out_attn两个参数后,
        判断是否为Auto-Attention, 如果是,则增加decomp block, 划分趋势与周期
        ??? 文中Autoformer主要考虑了周期之间的信息,考虑的是交通等数据, 但是在RUL中,周期性不明显,是否用趋势会更好???
        原文中在Encoder关注周期,Decoder关注趋势
        输出在整合在一起做ADD&NORM
        """
        if self.type_attention == "Auto":
            self.decomp_block = series_decomp(auto_moving_avg)


    def forward(self, queries, keys, values):
        """
        input x: [batch, length, channel], channel=input_dim
        return y: [batch, length, output_dim]
        """
        B, L_Q, I_Q = queries.shape
        _, L_K, I_K = keys.shape
        _, L_V, I_V = values.shape
        H = self.n_heads_type 
        if self.type_attention == "FFT":
            # fft projection
            fft_values = values
            fft_padding_size = int(self.value_kernel_size/2)
            padding_fft = nn.functional.pad(fft_values.permute(0, 2, 1),
                                            pad     = (fft_padding_size, fft_padding_size),
                                            mode    = 'replicate')
            fft_values = self.fft_projection(padding_fft).permute(0, 2, 1) # [B, L, output_dim] 
            
            out, attn = self.inner_attention(fft_values)
            return out, attn

        else:
            # query, key, value projection
            queries_padding_size =  int(self.causal_kernel_size/2)
            padding_queries = nn.functional.pad(queries.permute(0, 2, 1),
                                                pad     = (queries_padding_size, queries_padding_size),
                                                mode    = 'replicate')
            queries = self.query_projection(padding_queries).permute(0, 2, 1) # [B, L, d_k*n]   

            keys_padding_size =  int(self.causal_kernel_size/2)
            padding_keys = nn.functional.pad(keys.permute(0, 2, 1),
                                             pad    = (keys_padding_size, keys_padding_size),
                                             mode   = 'replicate')
            keys = self.key_projection(padding_keys).permute(0, 2, 1) # [B, L, d_k*n]        

            values_padding_size = int(self.value_kernel_size/2)
            padding_values = nn.functional.pad(values.permute(0, 2, 1),
                                               pad  = (values_padding_size, values_padding_size),
                                               mode = 'replicate')
            values = self.value_projection(padding_values).permute(0, 2, 1) # [B, L, d_v*n]        


            # 将数据Q, K, V转化为[B, L, H, -1]的形式, 作为attention的输入
            query = queries.view(B, L_Q, H, -1)
            key  = keys.view(B, L_K, H, -1)
            value = values.view(B, L_V, H, -1)
        
            out, attn = self.inner_attention(query, key, value)

            # out_projection的运算
            # return out的固定格式[B, L, output_dim]
            out = out.view(B, L_Q, -1)
            padding_out = nn.functional.pad(out.permute(0, 2, 1),
                                            pad     = (values_padding_size, values_padding_size),
                                            mode    = 'replicate')
             
            out = self.activation(self.out_projection(padding_out)).permute(0, 2, 1)
            out = self.resdi_dropout(out)

            # 判断是否为auto, 如果是auto, 需要增加decomp block, 如果不是, 可以直接输出
            if self.type_attention == "Auto":
                out, _ = self.decomp_block(out)
                return out, attn
            # 剩余的直接返回
            else:    
                return out, attn