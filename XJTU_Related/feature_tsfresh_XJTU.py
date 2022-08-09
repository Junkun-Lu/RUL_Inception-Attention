import pandas as pd
import numpy as np
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test, target_real_feature_binary_test
from sklearn import preprocessing


def identify_and_remove_unique_columns(Dataframe):
    Dataframe = Dataframe.copy()
  
    unique_counts = Dataframe.nunique()
    # 先去掉提取的值不变的特征
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    unique_to_drop = list(record_single_unique['feature'])
    Dataframe = Dataframe.drop(columns = unique_to_drop)
    
    # 再去掉相关性差的特征,根据是否是binary还是实值进行判断
    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    record_single_unique["type"] = record_single_unique["nunique"].apply(lambda x:"real" if x>2 else "binary")
    for i in range(record_single_unique.shape[0]):
       col = record_single_unique.loc[i,"feature"]
       _type = record_single_unique.loc[i,"type"]
       if _type == "real":
           p_value = target_real_feature_real_test(Dataframe[col], Dataframe["RUL"])
       else:
           le = preprocessing.LabelEncoder()
           p_value = target_real_feature_binary_test(pd.Series(le.fit_transform(Dataframe[col])), Dataframe["RUL"])
       if p_value>0.05:
           unique_to_drop.append(col)
    
    return  unique_to_drop


def tsfresh_sliding_window(fea_bearing, rul_bearing, window_length):
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