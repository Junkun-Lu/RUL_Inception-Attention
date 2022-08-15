import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import extract_features, EfficientFCParameters 
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from XJTU_Related.feature_STFT_XJTU import vibration_train_picture, vibration_test_picture
from XJTU_Related.feature_STFT_XJTU import STFT_process, STFT_picture_train, STFT_picture_test
from XJTU_Related.feature_STFT_XJTU import STFT_sliding_window
from XJTU_Related.feature_tsfresh_XJTU import identify_and_remove_unique_columns, tsfresh_sliding_window
from XJTU_Related.feature_vibration_XJTU import vibration_sliding_window



# load each file:
def load_file_acc(file_path, id, bearing_num):
    file_acc_df = pd.read_csv(file_path)
    # 增加一列用来在tsfresh中作为分类提取数据的标签
    file_acc_df["file_index"] = id+1
    file_acc_df.set_index("file_index", inplace=True)
    file_acc_df["id"] = int(str(bearing_num)+f"{str(id+1).zfill(4)}")
    file_acc_df["file_time"] = [i for i in range(0, len(file_acc_df))]
    return file_acc_df


# 读取一个bearing中的文件
def get_bearing_acc(folder_path, bearing_num):
    file_name_ls = os.listdir(folder_path)
    file_num = len(file_name_ls)  # Bearing文件夹中有多少个acc文件

    # 遍历读取所有acc文件,并将其组合在一起
    acc_ls = []
    for id in range(0, file_num):
        acc_file_path = folder_path + "/" + f"{str(id+1)}.csv"
        file_acc_df = load_file_acc(acc_file_path, id, bearing_num)
        acc_ls.append(file_acc_df)
    acc_df = pd.concat(acc_ls, axis=0, ignore_index=False)  # df的id是从1开始的,从'load_file_acc'中,id会加1

    return acc_df, file_num


# calculate rul
def rul_calculate(file_num, bearing_num):
    rul_ls = []
    for i in range(1, int(file_num)+1):
        rul_time = (file_num - i)  
        rul_ls.append(rul_time)
    rul_dataframe = pd.DataFrame(rul_ls, columns=['RUL'])
    rul_dataframe["id"] = [int(str(bearing_num)+f"{str(id+1).zfill(4)}") for id in range(0, file_num)]
    rul_dataframe.set_index("id", inplace=True)
    return rul_dataframe 


def data_visual_rul(rul_dataframe, bearing_acc_dataframe, bearing_name):    
    plt.figure(figsize=(24, 6))
    # plot1 - rul
    x = list(range(1, len(rul_dataframe)+1))
    plt.subplot(131)
    plt.plot(x, rul_dataframe.values)
    plt.title("RUL")
    plt.xlabel("sample file number")
    plt.ylabel("RUL 10(s)")

    # plot2 - Horizontal_vibration_signals
    plt.subplot(132)
    plt.plot(bearing_acc_dataframe[["Horizontal_vibration_signals"]])
    plt.title("Horizontal_vibration_signals")
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")

    # plot3 - Vertical_vibration_signals
    plt.subplot(133)
    plt.plot(bearing_acc_dataframe[["Vertical_vibration_signals"]])
    plt.title("Vertical_vibration_signals")
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/picture_XJTU/acc&rul_picture/acc&rul_{}.jpg'.format(bearing_name))


# 加载训练集/验证集数据
def data_load(root_dir, bearing_data_set, flag="train"):
    if flag == "train":
        print("------------------------------ start load training data ------------------------------")
        acc_dataframe_ls, rul_dataframe_ls = [], []
        file_num_ls = []
        for bearing_num, bearing_name in enumerate(bearing_data_set):
            bearing_num += 1
            folder_path = root_dir + "/" + bearing_name

            # load acc_data from bearing
            bearing_acc_dataframe, bearing_file_num = get_bearing_acc(folder_path, bearing_num) 
            print("{0} have {1} acceleration files".format(bearing_name, bearing_file_num))
            acc_dataframe_ls.append(bearing_acc_dataframe)
            file_num_ls.append(bearing_file_num) 

            # load rul from bearing
            bearing_rul_dataframe = rul_calculate(bearing_file_num, bearing_num)   
            rul_dataframe_ls.append(bearing_rul_dataframe)

            # picture of bearing_acc_dataframe and bearing_rul_dataframe
            data_visual_rul(bearing_rul_dataframe, bearing_acc_dataframe, bearing_name)
        
        train_x_dataframe = pd.concat(acc_dataframe_ls, axis=0, ignore_index=False)
        train_y_dataframe = pd.concat(rul_dataframe_ls, axis=0, ignore_index=False)

        return train_x_dataframe, train_y_dataframe, file_num_ls

    else:
        print("------------------------------ start load test data ------------------------------")
        bearing_name = bearing_data_set[0]
        bearing_num = 5    # train_set in each condition have 2 bearing dataset
        folder_path = root_dir + "/" + bearing_name

        # load acc_data from bearing
        bearing_acc_dataframe, bearing_file_num = get_bearing_acc(folder_path, bearing_num) 
        file_num_ls = [bearing_file_num]
        print("{0} have {1} acceleration files".format(bearing_name, bearing_file_num))

        # load rul from bearing
        bearing_rul_dataframe = rul_calculate(bearing_file_num, bearing_num)

        # picture of bearing_acc_dataframe and bearing_rul_dataframe
        data_visual_rul(bearing_rul_dataframe, bearing_acc_dataframe, bearing_name)

        test_x_dataframe, test_y_dataframe = bearing_acc_dataframe, bearing_rul_dataframe

        return test_x_dataframe, test_y_dataframe, file_num_ls



# ------------------------ 数据加载与处理函数 --------------------------
def train_test_generator_XJTU(pre_process_type, root_dir, train_bearing_data_set, test_bearing_data_set, STFT_window_len, STFT_overlap_num, window_length, validation_rate):
    # 加载原始数据并组合,绘制rul与水平&纵向加速度图
    train_x_dataframe, train_y_dataframe, train_file_num_ls = data_load(root_dir, train_bearing_data_set, flag="train")
    test_x_dataframe, test_y_dataframe, test_file_num_ls = data_load(root_dir, test_bearing_data_set, flag="test")

    # --------------------------------------- Vibration -------------------------------------------------------------------------------
    if pre_process_type == "Vibration":
        # 计算训练集和测试集的合力vibration, 再将训练集根据文件分为两个
        train_x_dataframe["vibration"] = train_x_dataframe.apply(lambda x: math.sqrt(x['Horizontal_vibration_signals']**2 + x['Vertical_vibration_signals']**2), axis=1)
        vibration_train_picture(train_x_dataframe, train_file_num_ls, train_bearing_data_set)
        test_x_dataframe["vibration"] = test_x_dataframe.apply(lambda x: math.sqrt(x['Horizontal_vibration_signals']**2 + x['Vertical_vibration_signals']**2), axis=1)
        vibration_test_picture(test_x_dataframe, test_bearing_data_set)

        # **standarlization**        
        vibration_train_x_np = train_x_dataframe[["vibration"]].values
        vibration_test_x_np = test_x_dataframe[["vibration"]].values 

        vibration_mean = np.mean(vibration_train_x_np, axis=0).reshape(-1, 1) 
        vibration_std = np.std(vibration_train_x_np, axis=0).reshape(-1, 1) 

        stand_train_vibration_feature = (vibration_train_x_np - vibration_mean) /  vibration_std
        stand_test_vibration_feature = (vibration_test_x_np - vibration_mean) / vibration_std

        # **split based file_num**
        train_split_ls = []
        for i in range(0, int(len(stand_train_vibration_feature)/32768)):
            split_file_feature_train = stand_train_vibration_feature[32768*i:32768*(i+1), :]
            split_file_feature_train = split_file_feature_train.transpose(1, 0)
            train_split_ls.append(split_file_feature_train)
        
        train_file_splited_feature = np.concatenate(train_split_ls, axis=0)

        test_split_ls = []
        for i in range(0, int(len(stand_test_vibration_feature)/32768)):
            split_file_feature_test = stand_test_vibration_feature[32768*i:32768*(i+1), :]
            split_file_feature_test = split_file_feature_test.transpose(1, 0)
            test_split_ls.append(split_file_feature_test)
        
        test_file_splited_feature = np.concatenate(test_split_ls, axis=0)


        # **window_length**
        # window_length_train
        train_y_array = train_y_dataframe.values
        win_train_x_ls, win_train_y_ls = [], []
        for i in range(0, len(train_file_num_ls)):
            split_trian_x_array = train_file_splited_feature[0:train_file_num_ls[i], :]
            split_train_y_array = train_y_array[0:train_file_num_ls[i], :]
            train_file_splited_feature = train_file_splited_feature[train_file_num_ls[i]:, :]
            train_y_array = train_y_array[train_file_num_ls[i]:, :]

            win_train_x, win_train_y = vibration_sliding_window(split_trian_x_array, split_train_y_array, window_length)
            win_train_x_ls.append(win_train_x) 
            win_train_y_ls.append(win_train_y)
        X_train = np.concatenate(win_train_x_ls, axis=0)
        y_train = np.concatenate(win_train_y_ls, axis=0)

        print("the shape of training set is {0} and the shape of train label is {1}".format(X_train.shape, y_train.shape))

        # window_length_test
        test_y_array = test_y_dataframe.values
        test_X, test_y = STFT_sliding_window(test_file_splited_feature, test_y_array, window_length)
        print("the shape of test_X is {0} and the shape of test_y is {1}".format(test_X.shape, test_y.shape))     

    # --------------------------------------- STFT -------------------------------------------------------------------------------
    if pre_process_type == "STFT":        
        # 计算训练集和测试集的合力vibration, 再将训练集根据文件分为两个
        train_x_dataframe["vibration"] = train_x_dataframe.apply(lambda x: math.sqrt(x['Horizontal_vibration_signals']**2 + x['Vertical_vibration_signals']**2), axis=1)
        vibration_train_picture(train_x_dataframe, train_file_num_ls, train_bearing_data_set)
        test_x_dataframe["vibration"] = test_x_dataframe.apply(lambda x: math.sqrt(x['Horizontal_vibration_signals']**2 + x['Vertical_vibration_signals']**2), axis=1)
        vibration_test_picture(test_x_dataframe, test_bearing_data_set)

        # **calculate STFT_train**
        train_stft_feature_ls = []
        train_stft_f_ls = []
        train_stft_t_ls = []
        for i in range(0, int(len(train_x_dataframe)/32768)):
            train_file_dataframe = train_x_dataframe.iloc[i*32768:(i+1)*32768, -1]
            train_temp_vibration = train_file_dataframe.values
            temp_zxx, temp_f, temp_t = STFT_process(train_temp_vibration, STFT_window_len, STFT_overlap_num)
            train_stft_feature_ls.append(temp_zxx)
            train_stft_f_ls.append(temp_f)
            train_stft_t_ls.append(temp_t)
        train_stft_feature = np.concatenate(train_stft_feature_ls, axis=0)  # 输出[file_num,, w, h]
        train_stft_f = train_stft_f_ls[0]
        train_stft_t = train_stft_t_ls[0]
        # 绘制STFT_train相关图片
        STFT_picture_train(train_stft_feature, train_stft_f, train_stft_t, train_file_num_ls, train_bearing_data_set)
        
        # **calculate STFT_test**
        test_stft_feature_ls = []
        test_stft_f_ls = []
        test_stft_t_ls = []
        for j in range(0, int(len(test_x_dataframe)/32768)):
            test_file_dataframe = test_x_dataframe.iloc[j*32768:(j+1)*32768, -1]
            test_temp_vibration = test_file_dataframe.values
            temp_zxx, temp_f, temp_t = STFT_process(test_temp_vibration, STFT_window_len, STFT_overlap_num)
            test_stft_feature_ls.append(temp_zxx)
            test_stft_f_ls.append(temp_f)
            test_stft_t_ls.append(temp_t)
        test_stft_feature = np.concatenate(test_stft_feature_ls, axis=0)  # 输出[file_num, w, h]
        test_stft_f = test_stft_f_ls[0]
        test_stft_t = test_stft_t_ls[0] 
        # 绘制STFT_test相关图片
        STFT_picture_test(test_stft_feature, test_stft_f, test_stft_t, test_bearing_data_set)      


        # **standarlization**
        STFT_mean = np.mean(train_stft_feature, axis=(0,1)).reshape(1, 1, -1)
        STFT_std = np.std(train_stft_feature, axis=(0,1)).reshape(1, 1, -1)
        stand_train_stft_feature = (train_stft_feature - STFT_mean) / STFT_std

        stand_test_stft_feature = (test_stft_feature - STFT_mean) / STFT_std


        # **window_length**
        # window_length_train
        train_y_array = train_y_dataframe.values
        win_train_x_ls, win_train_y_ls = [], []
        for i in range(0, len(train_file_num_ls)):
            split_stand_train_stft_feature = stand_train_stft_feature[0:train_file_num_ls[i], ::]
            split_train_y_array = train_y_array[0:train_file_num_ls[i], :]
            stand_train_stft_feature = stand_train_stft_feature[train_file_num_ls[i]:, ::]
            train_y_array = train_y_array[train_file_num_ls[i]:, :]

            win_train_x, win_train_y = STFT_sliding_window(split_stand_train_stft_feature, split_train_y_array, window_length)   
            win_train_x_ls.append(win_train_x) 
            win_train_y_ls.append(win_train_y)
        X_train = np.concatenate(win_train_x_ls, axis=0)
        y_train = np.concatenate(win_train_y_ls, axis=0)

        print("the shape of training set is {0} and the shape of train label is {1}".format(X_train.shape, y_train.shape))

        # window_length_test
        test_y_array = test_y_dataframe.values
        test_X, test_y = STFT_sliding_window(stand_test_stft_feature, test_y_array, window_length)
        print("the shape of test_X is {0} and the shape of test_y is {1}".format(test_X.shape, test_y.shape))

    # -------------------------------------------- tsfresh ---------------------------------------------------------------------
    if pre_process_type == "tsfresh":
        # 提取特征并将训练集和label组合在一起
        train_fea_df = extract_features(train_x_dataframe, column_id="id", column_sort="file_time", default_fc_parameters=EfficientFCParameters())
        impute(train_fea_df) 
        train_fea_rul_df = pd.concat([train_fea_df, train_y_dataframe], axis=1)
        train_fea_rul_df = train_fea_rul_df.reset_index(drop=True)

        test_fea_df = extract_features(test_x_dataframe, column_id="id", column_sort="file_time", default_fc_parameters=EfficientFCParameters())   
        impute(test_fea_df)
        test_fea_rul_df = pd.concat([test_fea_df, test_y_dataframe], axis=1)
        test_fea_rul_df = test_fea_rul_df.reset_index(drop=True)

        # 找到drop_feature_list
        drop_feature_list = identify_and_remove_unique_columns(train_fea_rul_df)

        # 去掉数据集中相关性不大的特征
        train_fea_rul_df = train_fea_rul_df.drop(drop_feature_list, axis=1)
        test_fea_rul_df = test_fea_rul_df.drop(drop_feature_list, axis=1)
        train_fea_rul_df.to_csv("/content/drive/MyDrive/picture_XJTU/tsfresh_feature_XJTU/train_{}.csv".format(train_bearing_data_set[0].split("_")[0]), index=True, sep=',')
        test_fea_rul_df.to_csv("/content/drive/MyDrive/picture_XJTU/tsfresh_feature_XJTU/test_{}.csv".format(test_bearing_data_set[0]), index=True, sep=',')
        print(len(list(train_fea_rul_df)), " feature are selected")
 
        # **standardlization**
        tsfresh_mean = train_fea_rul_df.iloc[:, :-1].mean()
        tsfresh_std = train_fea_rul_df.iloc[:, :-1].std()
        tsfresh_std.replace(0, 1, inplace=True)
        train_fea_rul_df.iloc[:, :-1] = (train_fea_rul_df.iloc[:, :-1] - tsfresh_mean) / tsfresh_std

        test_fea_rul_df.iloc[:, :-1] = (test_fea_rul_df.iloc[:, :-1] - tsfresh_mean) / tsfresh_std

        # **sliding_window_tsfresh**
        # sliding_window_train
        train_fea_np = train_fea_rul_df.iloc[:, :-1].values
        train_rul_np = train_fea_rul_df.iloc[:, -1:].values
        win_train_x_ls, win_train_y_ls = [], []
        for i in range(0,len(train_file_num_ls)):
            temp_fea_np = train_fea_np[:train_file_num_ls[i], :]
            temp_rul_np = train_rul_np[:train_file_num_ls[i], :]
            train_fea_np = train_fea_np[train_file_num_ls[i]:, :]
            train_rul_np = train_rul_np[train_file_num_ls[i]:, :]

            win_train_x, win_train_y = tsfresh_sliding_window(temp_fea_np, temp_rul_np, window_length)
            win_train_x_ls.append(win_train_x)
            win_train_y_ls.append(win_train_y)
        X_train = np.concatenate(win_train_x_ls, axis=0)
        y_train = np.concatenate(win_train_y_ls, axis=0)
        print("the shape of training set is {0} and the shape of train label is {1}".format(X_train.shape, y_train.shape))           
        
        # sliding_window_test
        fea_np = test_fea_rul_df.iloc[:, :-1].values
        rul_np = test_fea_rul_df.iloc[:, -1:].values
        test_X, test_y = tsfresh_sliding_window(fea_np, rul_np, window_length)
        print("the shape of test_X is {0} and the shape of test_y is {1}".format(test_X.shape, test_y.shape))

    # train_validation_split
    train_X, vali_X, train_y, vali_y =  train_test_split(X_train, y_train, test_size=validation_rate, random_state=42)
    print("the shape of train_X is {0} and the shape of train_y is {1}".format(train_X.shape, train_y.shape))
    print("the shape of vali_X is {0} and the shape of vali_y is {1}".format(vali_X.shape, vali_y.shape))

    return train_X, vali_X, test_X, train_y, vali_y, test_y