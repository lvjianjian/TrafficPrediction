#!/usr/bin/python
# -*- coding:utf-8 -*-


from jampredict.model.Baseline import BaseLine
import time
import os
from jampredict.utils import Paramater
from jampredict.utils.Cache import *
from jampredict.feature import Data
from jampredict.utils import Metric

len_closeness = 3
len_period = 1
len_trend = 1

CACHEDATA = True
is_mmn = False
nb_flow = 1

len_test = 300
hasExternal = False


# random_state = 1337


def main():
    # load data
    print("loading data...")

    ts = time.time()
    if is_mmn:
        fname = os.path.join(Paramater.DATAPATH, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_noExternal_mmn.h5'.format(len_closeness, len_period, len_trend))
    else:
        fname = os.path.join(Paramater.DATAPATH, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_noExternal.h5'.format(len_closeness, len_period, len_trend))

    f2name = fname.replace(".h5", "_cell.h5")
    if CACHEDATA and os.path.exists(f2name):
        print f2name
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = read_cache(
            f2name, is_mmn)
        print("load %s successfully" % f2name)
    else:
        if os.path.exists(fname) and CACHEDATA:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = read_cache(
                fname, is_mmn)

            print("load %s successfully" % fname)
        else:
            datapaths = [Paramater.DATAPATH + "48_48_20_LinearInterpolationFixed_condition"]
            noConditionRegionsPath = Paramater.PROJECTPATH + "data/48_48_20_noSpeedRegion_0.05"
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = Data.loadDataFromRaw(
                paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow, len_closeness=len_closeness,
                len_period=len_period, len_trend=len_trend
                , len_test=len_test, maxMinNormalization=is_mmn, preprocess_name='preprocessing.pkl', meta_data=False,
                meteorol_data=False,
                holiday_data=False)
            if CACHEDATA:
                cache(fname, X_train, Y_train, X_test, Y_test,
                      external_dim, timestamp_train, timestamp_test, noConditionRegions, is_mmn, x_num, y_num, z_num)
        X_train, Y_train = Data.transformMatrixToCell(X_train, Y_train, noConditionRegions, hasExternal)
        X_test, Y_test = Data.transformMatrixToCell(X_test, Y_test, noConditionRegions, hasExternal)

        if CACHEDATA:
            cache(f2name, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test,
                  list(noConditionRegions), is_mmn, x_num, y_num, z_num)

    bl = BaseLine(maxC=1, maxD=1, maxW=1, minSupport=10, minConfidence=10)
    bl.fit(X_train, Y_train, len_closeness, len_period, len_trend)


if __name__ == '__main__':
    main()
