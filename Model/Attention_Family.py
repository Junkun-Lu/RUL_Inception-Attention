import torch
import torch.nn as nn
import math
from math import sqrt
import numpy as np
import random


"""
该文件包含6种不同的masked-attention机制
"""

################################## Full-Attention block ##################################
# Full-Attention mask
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

# --------------------------- Full-Attention --------------------------------
class Full_Attention(nn.Module):
    def __init__(self, mask_flag=True, 
              attention_dropout=0.1, 
              output_attention=False):
        """
        FullAttention用的是原有的attention机制,使用mask:TriangularCausalMask
        dropout  是对分数的dropout     
        """
        super(Full_Attention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        """
        queries : [Batch, Length, Heads, E]
        keys    : [Batch, Length, Heads, E]
        values  : [Batch, Length, Heads, D]
        返回的是两个东西
        1.  新的value  格式依旧是 [Batch, Length, Heads, D]
        2.  attention 的map
        """
        B, L, H, E = queries.shape
        _, _, _, D = values.shape
        #print(".............................",queries.device)
        scale =  1./math.sqrt(E) #每个head的dimension

        queries = queries.permute(0, 2, 1, 3)  # [batch, heads, length, chanell]
        keys = keys.permute(0, 2, 3, 1)  # [batch, heads, chanell, length]
        scores = torch.matmul(queries, keys)
        scores = scale * scores
        
        if self.mask_flag:           
            attn_mask = TriangularCausalMask(B, L, device=queries.device)   # 选择不同的mask机制
            scores.masked_fill_(attn_mask.mask, -np.inf) #其实就是把不想要的地方设为True，然后再在这些地方加上 -inf

        pre_att = self.dropout(torch.softmax(scores , dim=-1))
        
        values = values.permute(0, 2, 1, 3) # [batch, heads, length, chanell]
        attn_values = torch.matmul(pre_att, values).permute(0,2,1,3) #[batch, length, heads, chanell]
        
        if self.output_attention:
            return (attn_values.contiguous(), pre_att)
        else:
            return (attn_values.contiguous(), None)



################################## Local-Attention block ##################################
class LocalMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            window_size = math.ceil(1.2*np.log2(L)/2)  #halb
            mask = torch.ones((L, L)).to(device)
            mask = torch.triu(mask,-window_size).T
            mask = torch.triu(mask,-window_size)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
    @property            
    def mask(self):
        return self._mask

# --------------------------- Local-Attention -------------------------------
class Local_Attention(nn.Module):
    def __init__(self, mask_flag=True, 
              attention_dropout=0.1, 
              output_attention=False):
        """
        FullAttention用的是原有的attention机制,使用mask:TriangularCausalMask
        dropout  是对分数的dropout     
        """
        super(Local_Attention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        """
        queries : [Batch, Length, Heads, E]
        keys    : [Batch, Length, Heads, E]
        values  : [Batch, Length, Heads, D]
        返回的是两个东西
        1.  新的value  格式依旧是 [Batch, Length, Heads, D]
        2.  attention 的map
        """
        B, L, H, E = queries.shape
        _, _, _, D = values.shape
        #print(".............................",queries.device)
        scale =  1./math.sqrt(E) #每个head的dimension

        queries = queries.permute(0, 2, 1, 3)  # [batch, heads, length, chanell]
        keys = keys.permute(0, 2, 3, 1)  # [batch, heads, chanell, length]
        scores = torch.matmul(queries, keys)
        scores = scale * scores
        
        if self.mask_flag:
            attn_mask = LocalMask(B, L, device=queries.device)   # 选择不同的mask机制
            scores.masked_fill_(attn_mask.mask, -np.inf) #其实就是把不想要的地方设为True，然后再在这些地方加上 -inf

        pre_att = self.dropout(torch.softmax(scores , dim=-1))
        
        values = values.permute(0, 2, 1, 3) # [batch, heads, length, chanell]
        attn_values = torch.matmul(pre_att, values).permute(0,2,1,3) #[batch, length, heads, chanell]
        
        if self.output_attention:
            return (attn_values.contiguous(), pre_att)
        else:
            return (attn_values.contiguous(), None)



################################## LocalLog-Attention block ##################################
class LocalLogMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.zeros((L, L), dtype=torch.float).to(device)
            for i in range(L):
                mask[i] = self.row_mask(i, L)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)

            
    def row_mask(self,index, L):
        local_window_size = math.ceil(np.log2(L)/2) # 1/2 window size
        # 对当前行进行初始化
        mask = torch.zeros((L), dtype=torch.float)

        if((index - local_window_size + 1) < 0):
            mask[:index] = 1 # Local attention
        else:
            mask[index - local_window_size + 1:(index + 1)] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index - local_window_size + 1 - int(1.5**i)
                if new_index >= 0:
                    mask[new_index] = 1
                else:
                    break
                    
        if ((index + local_window_size-1 )>=L):
            mask[index:] = 1 
        else:
            mask[index:index+local_window_size] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index + local_window_size-1 +int(1.5**i)
                if new_index < L:
                    mask[new_index] = 1
                else:
                    break
        return mask               

    @property          
    def mask(self):
        return self._mask

# -------------------------- LocalLog-Attention -----------------------------
class LocalLog_Attention(nn.Module):
    def __init__(self, mask_flag=True, 
              attention_dropout=0.1, 
              output_attention=False):
        """
        FullAttention用的是原有的attention机制,使用mask:TriangularCausalMask
        dropout  是对分数的dropout     
        """
        super(LocalLog_Attention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        """
        queries : [Batch, Length, Heads, E]
        keys    : [Batch, Length, Heads, E]
        values  : [Batch, Length, Heads, D]
        返回的是两个东西
        1.  新的value  格式依旧是 [Batch, Length, Heads, D]
        2.  attention 的map
        """
        B, L, H, E = queries.shape
        _, _, _, D = values.shape
        #print(".............................",queries.device)
        scale =  1./math.sqrt(E) #每个head的dimension

        queries = queries.permute(0, 2, 1, 3)  # [batch, heads, length, chanell]
        keys = keys.permute(0, 2, 3, 1)  # [batch, heads, chanell, length]
        scores = torch.matmul(queries, keys)
        scores = scale * scores
        
        if self.mask_flag:
            #print(self.mask_typ)           
            attn_mask = LocalLogMask(B, L, device=queries.device)   # 选择不同的mask机制
            #print("....................",attn_mask.mask.device)
            scores.masked_fill_(attn_mask.mask, -np.inf) #其实就是把不想要的地方设为True，然后再在这些地方加上 -inf

        pre_att = self.dropout(torch.softmax(scores , dim=-1))
        
        values = values.permute(0, 2, 1, 3) # [batch, heads, length, chanell]
        attn_values = torch.matmul(pre_att, values).permute(0,2,1,3) #[batch, length, heads, chanell]
        
        if self.output_attention:
            return (attn_values.contiguous(), pre_att)
        else:
            return (attn_values.contiguous(), None)



################################## ProbSparse-Attention block ##################################
# ProbSparse-Attention
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

# ------------------------- ProbSparse-Attention ---------------------------
class ProbSparse_Attention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbSparse_Attention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        # L_K 是肯定等于 L_Q的
        # sample_k 是为了算M 要采样的K的个数
        # n_top 选多少个Q
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
               
        #index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        sample_list = []
        for i in range(L_Q):
            index_list = list(np.arange(L_Q))
            #shuffle(index_list)
            sample_list.append(random.sample(index_list,sample_k))
        sample_array = np.array(sample_list)
        index_sample = torch.tensor(sample_array,dtype=torch.long)            
        
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        #print(M_top)

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k
        #print(Q_K)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        #print("................................",V.device)

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V)
        if self.output_attention:
            #这里和context_in是一样的
            attns = (torch.ones([B, H, L_V, L_V])/L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], 
                  torch.arange(H)[None, :, None], 
                  index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        # L_Q 和 L_K 是肯定相等的
        assert L_Q==L_K

        #queries = queries.view(B, H, L_Q, -1)
        #keys = keys.view(B, H, L_K, -1)
        #values = values.view(B, H, L_K, -1)

        keys = keys.permute(0, 2, 1, 3) 
        queries = queries.permute(0, 2, 1, 3) 
        values = values.permute(0, 2, 1, 3) 
        # U_part 也是肯定等于 u的 这里需要检查 U_part不能大于L_K
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        if U_part > L_K:
            #print(".")
            U_part = L_K
            u = L_K
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        context = context.permute(0,2,1,3)
        
        return context.contiguous(), attn



################################## Auto-Attention block ##################################
# 1. auto correlation block
class Auto_Attention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(Auto_Attention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            #  when query&key have different size of value
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 1, 3, 2))
        else:
            return (V.contiguous(), None)

# 2.decomposition block
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
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


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



################################## FFT-Attention block ##################################
class FFT_Attention(nn.Module):
  def __init__(self):
    super(FFT_Attention, self).__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x, None