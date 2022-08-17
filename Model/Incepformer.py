import torch
import torch.nn as nn
from Model.Preprocess_Layer import Preprocess_Layer_STFT, Preprocess_Layer_FC, Preprocess_Layer_Conv
from Model.Encoder import Encoder_Layer, Encoder
from Model.Predictor import LinearPredictor, FullPredictor, ConvPredictor


class Incepformer(nn.Module):
    def __init__(self,
           preprocess_type,
           preprocess_layer_num,
           preprocess_filter_num,  
           preprocess_kernel_size,
           preprocess_stride,
           input_feature,
           d_model,
           
           attention_layer_types,
           n_heads_full,
           n_heads_local,
           n_heads_log,
           n_heads_prob,
           n_heads_auto,
           n_heads_fft,
           d_keys,
           d_values,
           d_ff,
           dropout,
           activation,
           forward_kernel_size,
           value_kernel_size,
           causal_kernel_size,
           output_attention,
           auto_moving_avg,
           enc_layer_num,
           
           predictor_type,
           input_length):
        """
        ==> Preprocess Module <==
        preprocess_type: 根据要处理的数据选择是Proprocess_Layer_FC, 是Proprocess_Layer_STFT, 还是preprocess_Layer_Conv,
        preprocess_layer_num: Proprocess_Layer_Conv的参数, 卷积层的数量;
        preprocess_filter_num: Proprocess_Layer_Conv的参数, filter_num的数量, 为len(attention_layer_types)的倍数;
        preprocess_kernel_size: Proprocess_Layer_Conv的参数, 卷积核大小;(多选25)
        preprocess_stride: Proprocess_Layer_Conv, stride大小
        input_feature: Proprocess_Layer_FC的参数, 输入数据的特征数[B, L, F]中的"F";
        d_model: 隐藏层维度, 保证d_model = len(attention_layer_types) * random_number; Encoder-Module输入[B, L, D]中的"D";

        ==> Encoder Module <==
        attention_layer_types: 选择在Encoder中需要用到的masked-attention机制,list类型. ["Full", "Local", "LocalLog", "ProbSparse", "Auto", "FFT"];
        n_heads_full: type为Full_Attention的head数;
        n_heads_local: type为Local_Attention的head数;
        n_heads_log: type为LocalLog_Attention的head数;
        n_heads_prob: type为ProbSparse_Attention的head数;
        n_heads_auto: type为Auto_Attention的head数;
        n_heads_fft: type为FFT_Attention的head数, 只有一个head;
        d_ff: 最后forward时,中间过程的维度, 如果没有被定义的话, 那么就是d_model的两倍;
        dropout: Encoder中dropout的值;
        activation: "relu" 或 "gelu";
        forward_kernel_size: 用于Encoder中的线性层计算;
        value_kernel_size: 用于计算因果卷积中的value;
        causal_kernel_size: 用于计算因果卷积中的query和key;
        output_attention: True - 输出, False - 不是输出;
        auto_moving_avg: decomp block中参数, 只用于auto_attention;        
        enc_layer_num: Encoder_Layer的数量;

        ==> Predictor Module <==
        predictor_type: 预测器的类型, "full", "linear", "conv", "hybrid";
        input_length: 输入数据的sequence_length, [B, L, D]中的"L";
        """
        super(Incepformer, self).__init__()

        # preprocess module parameter:
        self.preprocess_type        = preprocess_type
        self.preprocess_layer_num   = preprocess_layer_num
        self.preprocess_filter_num  = preprocess_filter_num
        self.preprocess_kernel_size = preprocess_kernel_size
        self.preprocess_stride      = preprocess_stride
        self.input_feature          = input_feature
        self.d_model                = d_model

        # encoder module parameter:
        self.attention_layer_types  = attention_layer_types
        self.n_heads_full           = n_heads_full
        self.n_heads_local          = n_heads_local
        self.n_heads_log            = n_heads_log
        self.n_heads_prob           = n_heads_prob
        self.n_heads_auto           = n_heads_auto
        self.n_heads_fft            = n_heads_fft
        self.d_keys                 = d_keys
        self.d_values               = d_values
        self.d_ff                   = d_ff
        self.dropout                = dropout
        self.activation             = activation
        self.forward_kernel_size    = forward_kernel_size
        self.value_kernel_size      = value_kernel_size
        self.causal_kernel_size     = causal_kernel_size
        self.output_attention       = output_attention
        self.auto_moving_avg        = auto_moving_avg
        self.enc_layer_num          = enc_layer_num

        # predictor module parameter:
        self.predictor_type         = predictor_type
        self.input_length           = input_length
        

        # module
        if self.preprocess_type == "FC":
            self.preprocess_layer = Preprocess_Layer_FC(input_feature   = self.input_feature,
                                                        d_model         = self.d_model).double()
        if self.preprocess_type == "STFT":
            self.preprocess_layer = Preprocess_Layer_STFT(preprocess_layer_num      = self.preprocess_layer_num,
                                                          preprocess_filter_num     = self.preprocess_filter_num,  
                                                          preprocess_kernel_size    = self.preprocess_kernel_size).double()
        if self.preprocess_type == "Conv":
            self.preprocess_layer = Preprocess_Layer_Conv(preprocess_layer_num      = self.preprocess_layer_num,
                                                          preprocess_filter_num     = self.preprocess_filter_num,  
                                                          preprocess_kernel_size    = self.preprocess_kernel_size,
                                                          preprocess_stride         = self.preprocess_stride,
                                                          input_feature             = self.input_feature,
                                                          d_model                   = self.d_model,
                                                          dropout                   = self.dropout).double()
            
        self.encoder = Encoder([Encoder_Layer(attention_layer_types     = self.attention_layer_types,
                                              d_model                   = self.d_model,
                                              n_heads_full              = self.n_heads_full,
                                              n_heads_local             = self.n_heads_local,
                                              n_heads_log               = self.n_heads_log,
                                              n_heads_prob              = self.n_heads_prob,
                                              n_heads_auto              = self.n_heads_auto,
                                              n_heads_fft               = self.n_heads_fft,
                                              d_keys                    = self.d_keys,
                                              d_values                  = self.d_values,
                                              d_ff                      = self.d_ff,
                                              dropout                   = self.dropout,
                                              activation                = self.activation,
                                              forward_kernel_size       = self.forward_kernel_size,
                                              value_kernel_size         = self.value_kernel_size,
                                              causal_kernel_size        = self.causal_kernel_size,
                                              output_attention          = self.output_attention,
                                              auto_moving_avg           = self.auto_moving_avg)
                                for layer_num in range(self.enc_layer_num)]).double()

        if self.predictor_type == "full":
            self.predictor = FullPredictor(d_model = self.d_model, input_length = self.input_length).double()
        if self.predictor_type == "linear":
            self.predictor = LinearPredictor(d_model = self.d_model).double()
        if self.predictor_type == "conv":
            self.predictor = ConvPredictor(d_model = self.d_model).double()
        if self.predictor_type == "hybrid":
            self.predictor1 = FullPredictor(d_model = self.d_model, input_length = self.input_length).double()
            self.predictor2 = LinearPredictor(d_model = self.d_model).double()
            self.predictor3 = ConvPredictor(d_model = self.d_model).double()	
            self.predictor  = nn.Conv1d(in_channels = 3, out_channels = 1, kernel_size  = 3).double()

    def forward(self, x):
        """
        通过Preprocess Module将输入数据[B, L, F]或[b, L, F_H, F_W]变为 ==> [B, L, D],
        并将[B, L, D]作为输入数据输入到Encoder中;

        在Encoder中, 根据选择了n种masked-attention将d_model平均分给每一种masked-attention,
        针对每一种maksed-attention机制, 通过因果卷积将输入数据[B, L, D]映射为[B, L, d_keys*H/d_values*H], 并转化为[B, L, H, d_keys/d_values]输入到masked-attention中,
        针对FFT_Attention, 我们只用了value_projection映射的values数据[B, L, d_values*H_fft](H_fft=1)作为输入数据
        每一种Attention的output都为[B, L, d_model/n], 并最后cat在一起得到[B, L, D]
        通过线性层后得到Encoder Module的输出 ==> [B, L, D];

        在Predictor中, 根据predictor_type确定预测器的类型,最终的输出为[B, L], 保证我们是对整个input_length做预测
        """

        enc_out = self.preprocess_layer(x)

        enc_out, attns_list = self.encoder(enc_out)

        if self.predictor_type == "hybrid":
            pred1 = self.predictor1(enc_out)
            if len(pred1.shape) == 1:
                pred1 = torch.unsqueeze(pred1, 0)
            pred1 = torch.unsqueeze(pred1, 2)

            pred2 = self.predictor2(enc_out)
            if len(pred2.shape) == 1:
                pred2 = torch.unsqueeze(pred2, 0)
            pred2 = torch.unsqueeze(pred2, 2)

            pred3 = self.predictor3(enc_out) 
            if len(pred3.shape) == 1:
                pred3 = torch.unsqueeze(pred3, 0)
            pred3 = torch.unsqueeze(pred3, 2)

            hybrid_pred = torch.cat([pred1,pred2,pred3], dim=-1) 
            enc_out  = nn.functional.pad(hybrid_pred.permute(0, 2, 1), 
                            pad=(1, 1),
                            mode='replicate')
            enc_pred = self.predictor(enc_out).permute(0, 2, 1).squeeze()
        else:
            enc_pred = self.predictor(enc_out) # shape为[batch, length]
        
        if len(enc_pred.shape) == 1:
            enc_pred = torch.unsqueeze(enc_pred, 0)    


        if self.output_attention:
            return enc_pred, attns_list    
        else:
            return enc_pred, None   