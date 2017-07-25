#!/usr/bin/env python
# encoding=utf-8

import os

import h5py
import numpy as np
import pandas as pd
import pickle
from Matrix import MyMatrix
from jampredict.utils import Paramater, minmax_normalization

"""
@Author: zhongjianlv

@Create Date: 17-6-14, 15:50

@Description:

@Update Date: 17-6-14, 15:50
"""


class Error(Exception):
    pass


error = Error  # backward compatibility


def timeToStr(timeStamp):
    """
     将pd.timestamp 转换成 字符串，格式为%Y%m%d%H%M
    :param timeStamp:
    :return:
    """
    return timeStamp.strftime("%Y%m%d%H%M")


def xyToInt(x, y, y_num):
    return x * y_num + y


def intToXY(i, y_num):
    y = i % y_num
    x = i / y_num
    return (x, y)


def loadRawData(condition_path, nospeed_path, isComplete=True, complete_condition_value=Paramater.CONDITION_CLEAR,
                complete_weight_value=0):
    """
    读取数据 datas为4阶ndarray,shape(sample_size, z_num, x_num, y_num), 厚度为拥堵程度（0）和轨迹数量权重（1）
            times为1阶ndarray, 存储时间
    补全所有time及其下的data
    :param path: 数据文件名
    :param isComplete: 是否补全
    :param complete_value: 补全值,默认为4,通畅
    :return:
    """
    if (isComplete):
        h5file = condition_path + "_Complete.h5"
    else:
        h5file = condition_path + "_NoComplete.h5"

    if (os.path.exists(h5file)):
        h5 = h5py.File(h5file, 'r')
        datas = h5['datas'].value
        times = h5['times'].value
        x_num = h5['x_num'].value
        y_num = h5['y_num'].value
        interval = h5['interval'].value
        startTime = h5['startTime'].value
        endTime = h5['endTime'].value
        nospeed_regions = h5['noConditionRegions'].value
        return datas, times, x_num, y_num, interval, startTime, endTime, nospeed_regions

    file_names = condition_path.split("/")
    names__split = file_names[len(file_names) - 1].split("_")
    x_num = int(names__split[0])
    y_num = int(names__split[1])
    interval = int(names__split[2])

    with open(nospeed_path) as file:
        lines = file.readline()
        size = int(lines.strip())
        regions = file.readline().strip().split(",")
        regions.remove('')
        nospeed_regions = []
        for i in regions:
            nospeed_regions.append(intToXY(int(i), y_num))

    if (isComplete):
        timedelta = pd.Timedelta(str(interval) + 'minutes')
    datas = None
    times = None
    with open(condition_path) as file:
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
                    value[0] = complete_condition_value
                    value[1] = complete_weight_value
                    for x, y in nospeed_regions:
                        value[0][x][y] = Paramater.CONDITION_NO
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
                # if (len(xycw) < 4):
                #     print line
                #     assert
                for j in range(Paramater.Z_NUM):
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
        print "save in ", h5file
        h5 = h5py.File(h5file, 'w')
        h5.create_dataset('datas', data=datas)
        h5.create_dataset('times', data=times)
        h5.create_dataset('x_num', data=x_num)
        h5.create_dataset('y_num', data=y_num)
        h5.create_dataset('interval', data=interval)
        h5.create_dataset('startTime', data=startTime)
        h5.create_dataset('endTime', data=endTime)
        h5.create_dataset('noConditionRegions', data=nospeed_regions)
        print type(nospeed_regions)

    return datas, times, x_num, y_num, interval, startTime, endTime, nospeed_regions


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 0:
            i += 1
            days_incomplete.append(timestamps[i][:8])
            while int(timestamps[i][8:]) != 0 and i < len(timestamps):
                i += 1
        elif i + T - 1 < len(timestamps) and (
                        int(timestamps[i + T - 1][8:10]) * 60 + int(timestamps[i + T - 1][10:])) == 24 * 60 // T * (
                    T - 1):
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
            break
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


_copy_dispatch = d = {}
dispatch_table = {}
_deepcopy_dispatch = d = {}


def _keep_alive(x, memo):
    """Keeps a reference to the object x in the memo.

    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)] = [x]


def _deepcopy_atomic(x, memo):
    return x


def deepcopy(x, memo=None, _nil=[]):
    """Deep copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info.
    """

    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    cls = type(x)

    copier = _deepcopy_dispatch.get(cls)
    if copier:
        y = copier(x, memo)
    else:
        try:
            issc = issubclass(cls, type)
        except TypeError:  # cls is not a class (old Boost; see SF #502085)
            issc = 0
        if issc:
            y = _deepcopy_atomic(x, memo)
        else:
            copier = getattr(x, "__deepcopy__", None)
            if copier:
                y = copier(memo)
            else:
                reductor = dispatch_table.get(cls)
                if reductor:
                    rv = reductor(x)
                else:
                    reductor = getattr(x, "__reduce_ex__", None)
                    if reductor:
                        rv = reductor(2)
                    else:
                        reductor = getattr(x, "__reduce__", None)
                        if reductor:
                            rv = reductor()
                        else:
                            raise Error(
                                "un(deep)copyable object of type %s" % cls)
                y = _reconstruct(x, rv, 1, memo)

    memo[d] = y
    _keep_alive(x, memo)  # Make sure x lives at least as long as d
    return y


def _reconstruct(x, info, deep, memo=None):
    if isinstance(info, str):
        return x
    assert isinstance(info, tuple)
    if memo is None:
        memo = {}
    n = len(info)
    assert n in (2, 3, 4, 5)
    callable, args = info[:2]
    if n > 2:
        state = info[2]
    else:
        state = None
    if n > 3:
        listiter = info[3]
    else:
        listiter = None
    if n > 4:
        dictiter = info[4]
    else:
        dictiter = None
    if deep:
        args = deepcopy(args, memo)
    y = callable(*args)
    memo[id(x)] = y

    if state is not None:
        if deep:
            state = deepcopy(state, memo)
        if hasattr(y, '__setstate__'):
            y.__setstate__(state)
        else:
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            else:
                slotstate = None
            if state is not None:
                y.__dict__.update(state)
            if slotstate is not None:
                for key, value in slotstate.iteritems():
                    setattr(y, key, value)

    if listiter is not None:
        for item in listiter:
            if deep:
                item = deepcopy(item, memo)
            y.append(item)
    if dictiter is not None:
        for key, value in dictiter:
            if deep:
                key = deepcopy(key, memo)
                value = deepcopy(value, memo)
            y[key] = value
    return y


def copy(x):
    """Shallow copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info.
    """

    cls = type(x)

    copier = _copy_dispatch.get(cls)
    if copier:
        return copier(x)

    copier = getattr(cls, "__copy__", None)
    if copier:
        return copier(x)

    reductor = dispatch_table.get(cls)
    if reductor:
        rv = reductor(x)
    else:
        reductor = getattr(x, "__reduce_ex__", None)
        if reductor:
            rv = reductor(2)
        else:
            reductor = getattr(x, "__reduce__", None)
            if reductor:
                rv = reductor()
            else:
                raise Error("un(shallow)copyable object of type %s" % cls)

    return _reconstruct(x, rv, 0)


def loadDataFromRaw(paths, noSpeedRegionPath, nb_flow=1, len_closeness=None, len_period=None, len_trend=None,
                    len_test=None, maxMinNormalization=False, preprocess_name='preprocessing.pkl',
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
        datas, times, x_num, y_num, interval, startTime, endTime, noConditionRegions = loadRawData(path,
                                                                                                   noSpeedRegionPath)
        T = 24 * (60 // interval)
        # remove a certain day which does not have T timestamps
        data, timestamps = remove_incomplete_days(datas, times, T)
        data = data[:, :nb_flow]
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    if maxMinNormalization:
        data_train = np.vstack(copy(data_all))[:-len_test]
        print('train_data shape: ', data_train.shape)
        mmn = minmax_normalization.MinMaxNormalization()
        mmn.fit(data_train)
        data_all_mmn = [mmn.transform(d) for d in data_all]

        fpkl = open(preprocess_name, 'wb')
        for obj in [mmn]:
            pickle.dump(obj, fpkl)
        fpkl.close()
    else:
        mmn = None
        data_all_mmn = data_all

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = MyMatrix(data, timestamps, T)
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
    print "the final data:"
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, XP_train.shape, XT_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, XP_test.shape, XT_test.shape, Y_test.shape)

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
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, nb_flow


def getCellSize(matrix_size, x_num, y_num, z_num, noConditionRegionSize):
    return matrix_size * (z_num * (x_num * y_num - noConditionRegionSize))


def getMatrixSize(cell_size, x_num, y_num, z_num, noConditionRegionSize):
    return cell_size / (z_num * (x_num * y_num - noConditionRegionSize))


def transformMatrixToCell(X, Y, noConditionRegions, has_external):
    # print
    # print type(X), len(X)
    # print X[0].shape
    # print type(noConditionRegions), len(noConditionRegions)
    # print type(Y)
    sample_size = Y.shape[0]
    z_num = Y.shape[1]
    x_num = Y.shape[2]
    y_num = Y.shape[3]
    if type(noConditionRegions) == np.ndarray:
        noConditionRegionsSet = set()
        for i in range(noConditionRegions.shape[0]):
            noConditionRegionsSet.add((noConditionRegions[i][0], noConditionRegions[i][1]))
        noConditionRegions = noConditionRegionsSet
    else:
        noConditionRegions = set(noConditionRegions)
    if not has_external:
        size = getCellSize(sample_size, x_num, y_num, z_num, len(noConditionRegions))
        feature_size = 0
        for _x in X:
            feature_size += _x.shape[1]
        new_x = np.ndarray(shape=(size, feature_size))
        new_y = np.ndarray(shape=(size))
        index = 0
        for sample_index in range(sample_size):
            for i in range(z_num):
                for x in range(x_num):
                    for y in range(y_num):
                        if (x, y) in noConditionRegions:
                            continue
                        f_index = 0
                        for _x in X:
                            for k in range(_x.shape[1] / z_num):
                                new_x[index][f_index] = _x[sample_index, k * z_num + i, x, y]
                                f_index += 1
                        new_y[index] = Y[sample_index, i, x, y]
                        index += 1
    else:
        raise Exception("has_external not impl")
    return new_x, new_y


def transformCellToMatrix(predict, sample_size, x_num, y_num, z_num, noConditionRegions):
    index = 0
    matrix = np.ndarray(shape=(sample_size, z_num, x_num, y_num))
    for sample_index in range(sample_size):
        for i in range(z_num):
            for x in range(x_num):
                for y in range(y_num):
                    if (x, y) in noConditionRegions:
                        continue
                    matrix[sample_index, i, x, y] = predict[index]
                    index += 1
    return matrix


if __name__ == '__main__':
    # datas, times, x_num, y_num, interval, startTime, endTime, noConditionRegions = loadRawData(
    #     Paramater.PROJECTPATH + "data/48_48_20_LinearInterpolationFixed_condition",
    #     Paramater.PROJECTPATH + "data/48_48_20_noSpeedRegion_0.05",
    #     complete_condition_value=Paramater.CONDITION_CLEAR)
    # print datas[:, 0], times
    loadDataFromRaw([Paramater.DATAPATH + "48_48_20_LinearInterpolationFixed_condition"],
                    Paramater.PROJECTPATH + "data/48_48_20_noSpeedRegion_0.05", len_closeness=3, len_period=1,
                    len_trend=1, len_test=300)
