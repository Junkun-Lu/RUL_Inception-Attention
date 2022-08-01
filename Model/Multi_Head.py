import torch
import torch.nn as nn
from Model.Attention_Family import FullAttention, LogSparseAttention, LocalAttention, ProbAttention, FFT, AutoCorrelation
from Model.Decomposition_Block import series_decomp
from Model.Causal_Conv import Chomp1d


"""
Calculate of queries, keys, and values
input = [batch_size, max_len, d_model]
output shape after QKV calculation: [batch_size, d_k*n_heads, max_len]  --> shape change and get:
queries_shape = [batch_size, max_len, n_head_attention, d_k]
keys_shape = [batch_size, max_len, n_head_attention, d_k]
values_shape = [batch_size, max_len, n_head_attention, d_v]
only FFT-Block don't use queries, keys, and values.
output of Multi-head = [batch_size, max_len, d_model]
"""
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 d_k, d_v,
                 n_heads_full, n_heads_log, n_heads_local,
                 n_heads_prob, n_heads_FFT, n_heads_auto,
                 moving_avg, dropout, device):
        super(MultiHeadAttention, self).__init__()


        self.d_model       =  d_model       # d_model = input_len
        self.n_heads_full  =  n_heads_full
        self.n_heads_log   =  n_heads_log
        self.n_heads_local =  n_heads_local
        self.n_heads_prob  =  n_heads_prob
        self.n_heads_FFT   =  n_heads_FFT
        self.n_heads_auto  =  n_heads_auto
        self.device        =  device
        self.chomp1d = Chomp1d(2)   # 对应因果卷积


        # FullAttention -- Q, K, V
        self.Conv_Q_full = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_full,
                                     kernel_size=3, stride=1, padding=2)
        self.Conv_K_full = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_full,
                                     kernel_size=3, stride=1, padding=2)
        self.Conv_V_full = nn.Conv1d(in_channels=d_model, out_channels=d_v * n_heads_full,
                                     kernel_size=1, stride=1, padding=0)


        # LogSparseAttention -- Q, K, V
        self.Conv_Q_log = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_log,
                                    kernel_size=3, stride=1, padding=2)
        self.Conv_K_log = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_log,
                                    kernel_size=3, stride=1, padding=2)
        self.Conv_V_log = nn.Conv1d(in_channels=d_model, out_channels=d_v * n_heads_log,
                                    kernel_size=1, stride=1, padding=0)


        # LocalAttention -- Q, K, V
        self.Conv_Q_local = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_local,
                                      kernel_size=3, stride=1, padding=2)
        self.Conv_K_local = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_local,
                                      kernel_size=3, stride=1, padding=2)
        self.Conv_V_local = nn.Conv1d(in_channels=d_model, out_channels=d_v * n_heads_local,
                                      kernel_size=1, stride=1, padding=0)


        # ProbSparseAttention -- Q, K, V
        self.Conv_Q_prob = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_prob,
                                     kernel_size=3, stride=1, padding=2)
        self.Conv_K_prob = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_prob,
                                     kernel_size=3, stride=1, padding=2)
        self.Conv_V_prob = nn.Conv1d(in_channels=d_model, out_channels=d_v * n_heads_prob,
                                     kernel_size=1, stride=1, padding=0)


        # FFT don't need to calculate QKV, dirrect use input_Q
        self.projection_sizechange_FFT = nn.Linear(d_model, d_v * n_heads_FFT)


        # Autocorrelation -- Q, K, V
        # output of Autocorrelation is context of Auto-Correlation block and Decomposition block
        self.Conv_Q_auto = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_auto,
                                     kernel_size=3, stride=1, padding=2)
        self.Conv_K_auto = nn.Conv1d(in_channels=d_model, out_channels=d_k * n_heads_auto,
                                     kernel_size=3, stride=1, padding=2)
        self.Conv_V_auto = nn.Conv1d(in_channels=d_model, out_channels=d_v * n_heads_auto,
                                     kernel_size=1, stride=1, padding=0)
        self.auto_projection = nn.Linear(d_v * n_heads_auto, d_model)
        self.decomp = series_decomp(moving_avg)
        self.dropout_auto = nn.Dropout(dropout)
        self.projection_sizechange_auto = nn.Linear(d_model, d_v * n_heads_auto)


        # Fully Connected Layer
        # The number of heads should be calculated
        n_heads = n_heads_full + n_heads_log + n_heads_local + n_heads_prob + n_heads_FFT + n_heads_auto
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        """

        # residual parameter and others parameter for calculating
        residual, batch_size, seq_len = input_Q, input_Q.size(0), input_Q.size(1)
        # local_length = input_Q.size(1) // 10  # local in LocalAttention = seq_len/10
        context_ls = []  # this list used to save outputs from different head_attention

        # calculation about FullAttention
        if self.n_heads_full > 0:
            Q_full = self.chomp1d(self.Conv_Q_full(input_Q.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_full, -1)
            K_full = self.chomp1d(self.Conv_K_full(input_K.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_full, -1)
            V_full = self.Conv_V_full(input_V.transpose(1, 2)).view(batch_size, seq_len, self.n_heads_full, -1)

            context_full, attn_full = FullAttention()(Q_full, K_full, V_full, None)
            context_ls.append(context_full)

        # calculation about LogSparseAttention
        if self.n_heads_log > 0:
            Q_log = self.chomp1d(self.Conv_Q_log(input_Q.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_log, -1)
            K_log = self.chomp1d(self.Conv_K_log(input_K.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_log, -1)
            V_log = self.Conv_V_log(input_V.transpose(1, 2)).view(batch_size, seq_len, self.n_heads_log, -1)

            context_log, attn_log = LogSparseAttention()(Q_log, K_log, V_log)
            context_ls.append(context_log)

        # calculation about LocalAttention
        if self.n_heads_local > 0:
            Q_local = self.chomp1d(self.Conv_Q_local(input_Q.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_local, -1)
            K_local = self.chomp1d(self.Conv_K_local(input_K.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_local, -1)
            V_local = self.Conv_V_local(input_V.transpose(1, 2)).view(batch_size, seq_len, self.n_heads_local, -1)

            context_local, attn_local = LocalAttention()(Q_local, K_local, V_local, None)
            context_ls.append(context_local)

        # calculation about ProbSparseAttention
        if self.n_heads_prob > 0:
            Q_prob = self.chomp1d(self.Conv_Q_prob(input_Q.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_prob, -1)
            K_prob = self.chomp1d(self.Conv_K_prob(input_K.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_prob, -1)
            V_prob = self.Conv_V_prob(input_V.transpose(1, 2)).view(batch_size, seq_len, self.n_heads_prob, -1)

            context_prob, attn_prob = ProbAttention()(Q_prob, K_prob, V_prob, None)
            context_ls.append(context_prob)

        # calculation about FFT
        if self.n_heads_FFT > 0:
            context_FFT = FFT()(input_Q)
            context_FFT = self.projection_sizechange_FFT(context_FFT).view(batch_size, seq_len, self.n_heads_FFT, -1)
            context_ls.append(context_FFT)

        # calculation about AutoCorrelation
        if self.n_heads_auto > 0:
            Q_auto = self.chomp1d(self.Conv_Q_auto(input_Q.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_auto, -1)
            K_auto = self.chomp1d(self.Conv_K_auto(input_K.transpose(1, 2))).view(batch_size, seq_len, self.n_heads_auto, -1)
            V_auto = self.Conv_V_auto(input_V.transpose(1, 2)).view(batch_size, seq_len, self.n_heads_auto, -1)

            context_auto, attn_auto = AutoCorrelation()(Q_auto, K_auto, V_auto, None)
            context_auto = context_auto.view(batch_size, seq_len, -1)
            context_auto = self.auto_projection(context_auto)

            context_auto = residual + self.dropout_auto(context_auto)
            context_auto, _ = self.decomp(context_auto)
            context_auto = self.projection_sizechange_auto(context_auto).view(batch_size, seq_len, self.n_heads_auto, -1)
            context_ls.append(context_auto)

        # context the output from different head attention
        # input of FC = [batch_size,, seq_len, n_heads, d_v] --> [batch_size, seq_len, n_heads*d_v]
        # output = [batch_size, max_len, d_model]
        context = torch.cat(context_ls, dim=2)
        _, _, H, V = context.shape
        context = context.reshape(batch_size, -1, H * V)
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).double().to(self.device)(output + residual)
