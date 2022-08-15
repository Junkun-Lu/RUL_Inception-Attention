import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

"""
本文件需要更改图片保存地址
"""
# -------------------- 绘制合力的vibration图片 --------------------
"""
注意更改图片储存地址
"""
def vibration_train_picture(train_x_dataframe, train_file_num_ls, train_bearing_data_set): 
    plt.figure(figsize=(12, 3*len(train_file_num_ls)))  
    for i in range(0, len(train_file_num_ls)):
        temp_split_dataframe = train_x_dataframe.iloc[:2560*train_file_num_ls[i], -1]
        train_x_dataframe = train_x_dataframe.iloc[2560*train_file_num_ls[i]:, -1]
        train_x_dataframe = train_x_dataframe.reset_index()
        
        nums = i + 1
        plt.subplot(len(train_file_num_ls), 1, nums)
        plt.plot(temp_split_dataframe)
        plt.title("Vibration of {}".format(train_bearing_data_set[i]))
        plt.xlabel("sample file number")
        plt.ylabel("vibration signal") 
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/picture_FEMTO/Vibration_of_training_set/vibration_of_training_set_{}.jpg'.format(train_bearing_data_set[0].split("_")[0]))  

def vibration_test_picture(test_x_dataframe, test_bearing_data_set):
    plt.figure(figsize=(12, 3))
    plt.plot(test_x_dataframe[['vibration']])
    plt.title("Vibration of {}".format(test_bearing_data_set[0]))
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")
    plt.tight_layout()     
    plt.savefig('/content/drive/MyDrive/picture_FEMTO/Vibration_of_test_set/vibration_of_test_set_{}.jpg'.format(test_bearing_data_set[0]))  


# -------------------- STFT in each file calculate --------------------
def STFT_process(vibration, STFT_window_len, STFT_overlap_num):
    fs = 25.6*10e3   # sample frequence
    f, t, zxx = signal.stft(vibration, fs, window="hann", nperseg=STFT_window_len, noverlap=STFT_overlap_num)
    zxx = abs(zxx)    # remove imaginary type
    zxx = np.expand_dims(zxx, axis=0)
    return zxx, f, t

# -------------------- draw STFT picture --------------------
"""
注意更改图片储存地址
"""
# 绘制每个Bearing第一个和最后一个文件的STFT图
def STFT_picture_train(train_stft_feature, train_stft_f, train_stft_t, train_file_num_ls, train_bearing_data_set):
    plt.figure(figsize=(12, 3*len(train_file_num_ls)))
    for i in range(0, len(train_file_num_ls)):
        split_train_stft_feature = train_stft_feature[0:train_file_num_ls[i], ::]
        train_stft_feature = train_stft_feature[train_file_num_ls[i]:, ::]

        # plot1 - STFT of bearingx_i in first file
        plt.subplot(len(train_file_num_ls), 2, int(2*i+1))
        plt.pcolormesh(train_stft_t, train_stft_f, split_train_stft_feature[0, ::], shading='gouraud')
        plt.title("STFT of {} in first file".format(train_bearing_data_set[i]))
        plt.xlabel("time(s)")
        plt.ylabel("Frequency(Hz)")  

        # plot2 - STFT of bearingx_i in last file
        plt.subplot(len(train_file_num_ls), 2, int(2*i+2))  
        plt.pcolormesh(train_stft_t, train_stft_f, split_train_stft_feature[-1, ::], shading='gouraud')
        plt.title("STFT of {} in first file".format(train_bearing_data_set[i]))
        plt.xlabel("time(s)")
        plt.ylabel("Frequency(Hz)")
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/picture_FEMTO/STFT_picture/STFT_train_picture/STFT_of_training_set_{}.jpg'.format(train_bearing_data_set[0].split("_")[0]))      

def STFT_picture_test(test_stft_feature, test_stft_f, test_stft_t, test_bearing_data_set):
    plt.figure(figsize=(12, 3))
    # plot1 - STFT of Bearingx_j in first file
    plt.subplot(121)   
    plt.pcolormesh(test_stft_t, test_stft_f, test_stft_feature[0,::], shading='gouraud')
    plt.title("STFT of {} in first file".format(test_bearing_data_set[0]))
    plt.xlabel("time(s)")
    plt.ylabel("Frequency(Hz)")

    # plot2 - STFT of Bearingx_j in last file
    plt.subplot(122)   
    plt.pcolormesh(test_stft_t, test_stft_f, test_stft_feature[-1,::], shading='gouraud')
    plt.title("STFT of {} in last file".format(test_bearing_data_set[0]))
    plt.xlabel("time(s)")
    plt.ylabel("Frequency(Hz)")

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/picture_FEMTO/STFT_picture/STFT_test_picture/STFT_test_{}.jpg'.format(test_bearing_data_set[0]))    

  
# -------------------- STFT的移动滑窗 --------------------
def STFT_sliding_window(stand_train_stft_feature, y_array, window_length):
    seg_x_ls = []
    seg_y_ls = []
    for i in range(0, len(y_array)-window_length+1):
        seg_x = stand_train_stft_feature[i:i+window_length, ::]
        seg_x = np.expand_dims(seg_x, axis=0)
        seg_x_ls.append(seg_x)
        seg_y = y_array[i:i+window_length, :]
        seg_y = seg_y.transpose(1,0)
        seg_y_ls.append(seg_y)
    seg_x_np = np.concatenate(seg_x_ls, axis=0)
    seg_y_np = np.concatenate(seg_y_ls, axis=0)
    return seg_x_np, seg_y_np