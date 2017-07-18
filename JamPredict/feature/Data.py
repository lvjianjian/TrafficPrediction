#!/usr/bin/env python
# encoding=utf-8

from conf import Paramater
import pandas as pd
import numpy as np
import time
import h5py
import os
from deepst.datasets import STMatrix

"""
@Author: zhongjianlv

@Create Date: 17-6-14, 15:50

@Description:

@Update Date: 17-6-14, 15:50
"""


def timeToStr(timeStamp):
    """
     将pd.timestamp 转换成 字符串，格式为%Y%m%d%H%M
    :param timeStamp:
    :return:
    """
    return timeStamp.strftime("%Y%m%d%H%M")


def loadRawData(path, isComplete=True, complete_value=0):
    """
    读取数据 datas为4阶ndarray,shape(sample_size, z_num, x_num, y_num), 厚度为拥堵程度（0）和轨迹数量权重（1）
            times为1阶ndarray, 存储时间
    补全所有time及其下的data
    :param path: 数据文件名
    :param isComplete: 是否补全
    :param complete_value: 补全值,默认为0,通畅
    :return:
    """
    if (isComplete):
        h5file = path + "_Complete.h5"
    else:
        h5file = path + "_NoComplete.h5"

    if (os.path.exists(h5file)):
        h5 = h5py.File(h5file, 'r')
        datas = h5['datas'].value
        times = h5['times'].value
        x_num = h5['x_num'].value
        y_num = h5['y_num'].value
        interval = h5['interval'].value
        startTime = h5['startTime'].value
        endTime = h5['endTime'].value
        return datas, times, x_num, y_num, interval, startTime, endTime

    file_names = path.split("/")
    names__split = file_names[len(file_names) - 1].split("_")
    x_num = int(names__split[0])
    y_num = int(names__split[1])
    interval = int(names__split[2])
    if (isComplete):
        timedelta = pd.Timedelta(str(interval) + 'minutes')
    datas = None
    times = None
    with open(path) as file:
        lines = file.readlines()
        startTime = lines[0].split("|")[0].strip()
        startTimeStamp = pd.Timestamp(startTime)
        currentTimeStamp = startTimeStamp
        endTime = lines[len(lines) - 1].split("|")[0].strip()
        # rng = pd.date_range(startTime, endTime, freq=str(interval) + 'min')
        # ts = pd.Series(np.random.randn(len(rng)), index=rng, dtype=np.ndarray)
        for line in lines:
            value = np.zeros((Paramater.Z_NUM, x_num, y_num), dtype=int)
            split = line.split("|")
            time = split[0].strip()
            if (isComplete):
                while (currentTimeStamp != pd.Timestamp(time)):
                    ctime = timeToStr(currentTimeStamp)
                    # print "complete,", ctime
                    value = np.zeros((Paramater.Z_NUM, x_num, y_num), dtype=int)
                    value[:] = complete_value
                    if datas is None:
                        datas = [value]
                    else:
                        datas = np.concatenate((datas, [value]))

                    if times is None:
                        times = np.array([ctime])
                    else:
                        times = np.concatenate((times, [ctime]))
                    currentTimeStamp += timedelta

            for i in range(1, len(split)):
                xycw = split[i].split(",")
                for j in range(Paramater.Z_NUM):
                    value[j][int(xycw[0])][int(xycw[1])] = int(xycw[j + 2])
                    value[j][int(xycw[0])][int(xycw[1])] = int(xycw[j + 2])
            # ts[pd.Timestamp(split[0])] = value
            if datas is None:
                datas = [value]
            else:
                datas = np.concatenate((datas, [value]))

            if times is None:
                times = np.array([time])
            else:
                times = np.concatenate((times, [time]))
            # print time
            if (isComplete):
                currentTimeStamp += timedelta
        assert times.shape[0] == datas.shape[0]

    if (not os.path.exists(h5file)):
        h5 = h5py.File(h5file, 'w')
        h5.create_dataset('datas', data=datas)
        h5.create_dataset('times', data=times)
        h5.create_dataset('x_num', data=x_num)
        h5.create_dataset('y_num', data=y_num)
        h5.create_dataset('interval', data=interval)
        h5.create_dataset('startTime', data=startTime)
        h5.create_dataset('endTime', data=endTime)

    return datas, times, x_num, y_num, interval, startTime, endTime


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def loadDataFromRaw(paths, nb_flow=1, len_closeness=None, len_period=None, len_trend=None,
                    len_test=None, preprocess_name='preprocessing.pkl',
                    meta_data=True, meteorol_data=True, holiday_data=True):
    """
    """

    if len_closeness is None:
        len_closeness = 0
    if len_period is None:
        len_period = 0
    if len_trend is None:
        len_trend = 0

    assert (len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for path in paths:
        print("file name: ", path)
        datas, times, x_num, y_num, interval, startTime, endTime = loadRawData(path)
        T = 24 * (60 // interval)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(datas, times, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    # data_train = np.vstack(copy(data_all))[:-len_test]
    # print('train_data shape: ', data_train.shape)
    # mmn = MinMaxNormalization()
    # mmn.fit(data_train)
    # data_all_mmn = [mmn.transform(d) for d in data_all]
    #
    # fpkl = open(preprocess_name, 'wb')
    # for obj in [mmn]:
    #     pickle.dump(obj, fpkl)
    # fpkl.close()
    mmn = None

    data_all_mmn = data_all

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    # meta_feature = []
    # if meta_data:
    #     # load time feature
    #     time_feature = timestamp2vec(timestamps_Y)
    #     meta_feature.append(time_feature)
    # if holiday_data:
    #     # load holiday
    #     holiday_feature = load_holiday(timestamps_Y)
    #     meta_feature.append(holiday_feature)
    # if meteorol_data:
    #     # load meteorol data
    #     meteorol_feature = load_meteorol(timestamps_Y)
    #     meta_feature.append(meteorol_feature)
    #
    # meta_feature = np.hstack(meta_feature) if len(
    #     meta_feature) > 0 else np.asarray(meta_feature)
    # metadata_dim = meta_feature.shape[1] if len(
    #     meta_feature.shape) > 1 else None
    # if metadata_dim < 1:
    #     metadata_dim = None
    # if meta_data and holiday_data and meteorol_data:
    #     print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
    #           'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)
    metadata_dim = None

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
                                            :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
                                        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
                                      :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    # if metadata_dim is not None:
    #     meta_feature_train, meta_feature_test = meta_feature[
    #                                             :-len_test], meta_feature[-len_test:]
    #     X_train.append(meta_feature_train)
    #     X_test.append(meta_feature_test)
    # for _X in X_train:
    #     print(_X.shape, )
    # print()
    # for _X in X_test:
    #     print(_X.shape, )
    # print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


if __name__ == '__main__':
    DataFileName = "48_48_20_cate"
    # datas, times, x_num, y_num, interval,_,_ = loadRawData(Paramater.DATAPATH + DataFileName)
    loadDataFromRaw([Paramater.DATAPATH + DataFileName], len_closeness=3, len_period=1,len_trend=0)
