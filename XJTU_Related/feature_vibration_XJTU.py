import sys
sys.path.append("..")
import numpy as np


# 移动滑窗
def vibration_sliding_window(fea_bearing, rul_bearing, window_length):
    seg_fea_ls = []
    seg_rul_ls = []
    for i in range(0, len(rul_bearing)-window_length+1):
        seg_fea = fea_bearing[i:i+window_length, :]
        seg_fea = np.expand_dims(seg_fea, axis=0)
        seg_fea_ls.append(seg_fea)
        seg_rul = rul_bearing[i:i+window_length, :]
        seg_rul = seg_rul.transpose(1,0)
        seg_rul_ls.append(seg_rul)
    seg_fea = np.concatenate(seg_fea_ls, axis=0)
    seg_rul = np.concatenate(seg_rul_ls, axis=0)
    return seg_fea, seg_rul