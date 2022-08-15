import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



# this function used find features with poor correlation
def identify_and_remove_unique_columns(Dataframe):
    Dataframe = Dataframe.copy()
    del Dataframe["engine_id"]
    del Dataframe["cycle"]

    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
        columns={'index': 'feature', 0: 'nunique'})
    unique_to_drop = list(record_single_unique['feature'])

    return unique_to_drop


def Cmapss_train_vali_batch_generator(training_data, sequence_length=15):
    """
    data generate for turbofan dataset
    Generator function for creating random batches of training-data
    """

    engine_ids = list(training_data["engine_id"].unique())
    # print(engine_ids)
    temp = training_data.copy()
    for id_ in engine_ids:
        indexes = temp[temp["engine_id"] == id_].index
        traj_data = temp.loc[indexes]
        cutoff_cycle = max(traj_data['cycle']) - sequence_length + 1

        if cutoff_cycle <= 0:
            drop_range = indexes
            print("sequence_length + window_size is too large")
        else:
            cutoff_cycle_index = traj_data['cycle'][traj_data['cycle'] == cutoff_cycle + 1].index
            drop_range = list(range(cutoff_cycle_index[0], indexes[-1] + 1))

        temp.drop(drop_range, inplace=True)
    indexes = list(temp.index)
    del temp

    feature_number = training_data.shape[1] - 3

    x_shape = (len(indexes), sequence_length, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    y_shape = (len(indexes), sequence_length)
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)

    for batch_index, index in enumerate(indexes):
        y_batch[batch_index] = training_data.iloc[index:index + sequence_length, -1]
        x_batch[batch_index] = training_data.iloc[index:index + sequence_length, 2:-1].values

    return x_batch, y_batch


def Cmapss_test_batch_generator(test_data, sequence_length=5):
    engine_ids = list(test_data["engine_id"].unique())

    feature_number = test_data.shape[1] - 3

    x_batch = []
    y_batch = []

    for _id in set(test_data['engine_id']):
        test_of_one_id = test_data[test_data['engine_id'] == _id]

        if test_of_one_id.shape[0] >= sequence_length:
            x_batch.append(test_of_one_id.iloc[-sequence_length:, 2:-1].values)
            y_batch.append(test_of_one_id.iloc[-sequence_length:, -1].values)

    return np.array(x_batch), np.array(y_batch)


# 计算difference
def cal_diff(df, sensor_name, diff_periods=1):
    sensor_diff = []

    for _id in set(df['engine_id']):
        trainFD001_of_one_id = df[df['engine_id'] == _id]
        s = pd.Series(trainFD001_of_one_id[sensor_name])

        if len(s) > diff_periods:
            sensor_diff_temp = s.diff(periods=diff_periods)

            for i in range(diff_periods):
                sensor_diff.append(s.iloc[i] - s.iloc[0])

            for j in range(len(s) - diff_periods):
                sensor_diff.append(sensor_diff_temp.iloc[diff_periods + j])
        else:
            for h in range(len(s)):
                sensor_diff.append(s.iloc[h] - s.iloc[0])
    return sensor_diff


# load and process training&validation or test data
def cmapss_data_train_vali_loader(data_path,
                  Data_id,
                  flag="train",
                  sequence_length=40,
                  MAXLIFE=120,
                  difference=False,
                  normalization="znorm",
                  validation=0.1):
    # --------------- read the train data, test data and labels for test ---------------

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_FD = pd.read_table("{}/train_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)
    train_FD.columns = column_name

    test_FD = pd.read_table("{}/test_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)
    test_FD.columns = column_name

    RUL_FD = pd.read_table("{}/RUL_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)


    # ---------------- difference ------------------------

    if difference:
        diff_columns = train_FD.columns[2:]
        for i in range(len(diff_columns)):
            sensor_name_temp = diff_columns[i]
            diff = cal_diff(train_FD, sensor_name=sensor_name_temp)
            name = sensor_name_temp + '_diff'
            train_FD[name] = diff
        for i in range(len(diff_columns)):
            sensor_name_temp = diff_columns[i]
            diff = cal_diff(test_FD, sensor_name=sensor_name_temp)
            name = sensor_name_temp + '_diff'
            test_FD[name] = diff


    # --------------- define the label for train and test ---------------
    # piecewise linear RUL  for Training data
    # MAXLIFE define after this timepoint, the work time is RUL

    id = 'engine_id'
    rul = []
    for _id in set(train_FD[id]):
        trainFD_of_one_id = train_FD[train_FD[id] == _id]
        cycle_list = trainFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list)

        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            #
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)
        rul.extend(kink_RUL)

    train_FD["RUL"] = rul

    # piecewise linear RUL  for Test data

    id = 'engine_id'
    rul = []
    for _id_test in set(test_FD[id]):
        true_rul = int(RUL_FD.iloc[_id_test - 1])
        testFD_of_one_id = test_FD[test_FD[id] == _id_test]
        cycle_list = testFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list) + true_rul
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        rul.extend(kink_RUL)

    test_FD["RUL"] = rul


    # --------------- acoording to the labels of training dataset, delete redundant input sensors ---------------

    col_to_drop = identify_and_remove_unique_columns(train_FD)
    print(col_to_drop)
    train_FD = train_FD.drop(col_to_drop, axis=1)
    test_FD = test_FD.drop(col_to_drop, axis=1)


    # ---------------- Normalization --------------------------------

    if normalization == "znorm":
        mean = train_FD.iloc[:, 2:-1].mean()
        std  = train_FD.iloc[:, 2:-1].std()
        std.replace(0, 1, inplace=True)

        # training dataset
        train_FD.iloc[:, 2:-1] = (train_FD.iloc[:, 2:-1] - mean) / std

        # Testing dataset
        test_FD.iloc[:, 2:-1] = (test_FD.iloc[:, 2:-1] - mean) / std
    if normalization == "minmax":
        min_ = train_FD.iloc[:, 2:-1].min()
        max_ = train_FD.iloc[:, 2:-1].max()
        dis  = max_ - min_
        dis.replace(0, 1, inplace=True)

        # training dataset
        train_FD.iloc[:, 2:-1] = (train_FD.iloc[:, 2:-1] - min_) / dis

        # Testing dataset
        test_FD.iloc[:, 2:-1] = (test_FD.iloc[:, 2:-1] - min_) / dis


    # ------------------- batch generator -------------------------------

    if flag == "train":

        data_x, data_y = Cmapss_train_vali_batch_generator(train_FD, sequence_length)
        X_train, X_vali, y_train, y_vali = train_test_split(data_x, data_y, test_size=validation, random_state=42)
        print("the shape of X_train is:", X_train.shape)
        print("the shape of y_train is:", y_train.shape)
        return X_train, y_train, X_vali, y_vali

    else:
        # process test_data
        data_x, data_y = Cmapss_test_batch_generator(test_FD, sequence_length)

        return data_x, data_y
