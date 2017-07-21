#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-20, 13:47

@Description:

@Update Date: 17-7-20, 13:47
"""

from sklearn.tree import DecisionTreeClassifier
import time, os
from jampredict.utils import Paramater
from jampredict.utils.Cache import *
from jampredict.feature import Data

len_closeness = 3
len_period = 1
len_trend = 1

CACHEDATA = True
is_mmn = False
nb_flow = 1

len_test = 300
hasExternal = False


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
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions = read_cache(
            f2name, is_mmn)
        print("load %s successfully" % f2name)
    else:
        if os.path.exists(fname) and CACHEDATA:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions = read_cache(
                fname, is_mmn)

            print("load %s successfully" % fname)
        else:
            datapaths = [Paramater.DATAPATH + "48_48_20_LinearInterpolationFixed_condition"]
            noConditionRegionsPath = Paramater.PROJECTPATH + "data/48_48_20_noSpeedRegion_0.05"
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions = Data.loadDataFromRaw(
                paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow, len_closeness=len_closeness,
                len_period=len_period, len_trend=len_trend
                , len_test=len_test, maxMinNormalization=is_mmn, preprocess_name='preprocessing.pkl', meta_data=False,
                meteorol_data=False,
                holiday_data=False)
            if CACHEDATA:
                cache(fname, X_train, Y_train, X_test, Y_test,
                      external_dim, timestamp_train, timestamp_test, noConditionRegions, is_mmn)

        X_train, Y_train = Data.transformMatrixToCell(X_train, Y_train, noConditionRegions, hasExternal)
        X_test, Y_test = Data.transformMatrixToCell(X_test, Y_test, noConditionRegions, hasExternal)

        if CACHEDATA:
            cache(f2name, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test,
                  list(noConditionRegions), is_mmn)

    classfier = DecisionTreeClassifier(criterion="entropy")
    classfier.fit(X_train, Y_train)
    score = classfier.score(X_test, Y_test)
    print score


if __name__ == '__main__':
    main()
