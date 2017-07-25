#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-20, 13:47

@Description:

@Update Date: 17-7-20, 13:47
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time, os
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

grid_cv = False

random_state = 1337


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

    # grid cv
    if grid_cv:
        max_depth = [None, 5, 10, 15]
        min_samples_split = [2, 4, 6]
        min_samples_leaf = [1, 2, 3]
        criterion = ["gini", "entropy"]
        param_grid = dict(max_depth=max_depth, min_samples_split=min_samples_split,
                          min_samples_leaf=min_samples_leaf,
                          criterion=criterion)

        grid = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_state), scoring="accuracy",
                            param_grid=param_grid,
                            n_jobs=-1, verbose=1)
        grid.refit = False
        grid_result = grid.fit(X_train, Y_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        max_depth = grid_result.best_params_['max_depth']
        min_samples_split = grid_result.best_params_['min_samples_split']
        min_samples_leaf = grid_result.best_params_['min_samples_leaf']
        criterion = grid_result.best_params_["criterion"]

    else:
        max_depth = 10
        min_samples_split = 4
        min_samples_leaf = 1
        criterion = "gini"


    classfier = DecisionTreeClassifier(criterion=criterion,
                                       max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split,
                                       random_state=random_state)
    print "DT train ing.."
    classfier.fit(X_train, Y_train)
    print "train finish"
    score = classfier.score(X_test, Y_test)
    print score

    predict = classfier.predict(X_test)
    predict = Data.transformCellToMatrix(predict, Data.getMatrixSize(predict.shape[0], x_num, y_num, z_num,
                                                                     len(noConditionRegions)), x_num, y_num, z_num,
                                         noConditionRegions)
    Y_test = Data.transformCellToMatrix(Y_test, Data.getMatrixSize(Y_test.shape[0], x_num, y_num, z_num,
                                                                   len(noConditionRegions)), x_num, y_num, z_num,
                                        noConditionRegions)
    print("RMSE:", Metric.RMSE(predict, Y_test, noConditionRegions))
    print("accuracy", Metric.accuracy(predict, Y_test, noConditionRegions))


if __name__ == '__main__':
    main()
