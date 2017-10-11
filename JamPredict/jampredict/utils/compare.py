#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-11, 13:46

@Description:

@Update Date: 17-10-11, 13:46
"""

import time
import re
import os
from jampredict.utils import *
from jampredict.utils import Cache
from jampredict.feature import Data
import pandas as pd
from Metric import RMSE
from jampredict.feature.Data import get_no_speed_region


def find_big_change(month, fn, big_change_save_path):
    print("loading data...")
    ts = time.time()
    datapath = os.path.join(Paramater.DATAPATH, "2016", month)
    noConditionRegionsPath = os.path.join(datapath, "48_48_20_noSpeedRegion_0.05")
    datapath = os.path.join(datapath, fn)
    datas, times, x_num, y_num, interval, startTime, endTime, nospeed_regions = \
        Data.loadRawData(datapath, noConditionRegionsPath, isComplete=False)

    datas = datas[:, 0]
    diff = []
    for i in range(1, len(times)):
        diff.append(np.sqrt(np.sum((datas[i] - datas[i - 1]) ** 2)))

    df_diff = pd.DataFrame({"time": times[1:], "diff": diff}).sort(columns="diff", ascending=False)
    df_diff.to_csv(big_change_save_path)  # "./data/2016/all/" + fn +"_diff.csv"

    print df_diff
    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::72]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))


def compare_big_change_predict(diff_fn, compare_fn1, compare_fn2, no_speed_fn, next_num=2):
    pattern = re.compile(r'.*(\d\d)_(\d\d)_(\d\d).*')
    match = re.match(pattern, diff_fn)
    x_num = int(match.group(1))
    y_num = int(match.group(2))
    time_window = int(match.group(3))
    df_diff = pd.read_csv(diff_fn, index_col=0)
    top_diffs = df_diff.time[:3000].values
    compare_diffs = []
    # exit(1)
    for diff_time in top_diffs:
        new_time = pd.Timestamp(str(diff_time))
        # print (int(new_time.strftime(TIME_FORMAT)))
        for _n in range(next_num):
            new_time = new_time + pd.Timedelta(minutes=time_window)
            # print (int(new_time.strftime(TIME_FORMAT)))
            compare_diffs.append(int(new_time.strftime(TIME_FORMAT)))
    compare_diffs = np.asarray(compare_diffs, dtype=np.int64)
    p1, r1, t1 = Cache.read_result(compare_fn1)
    p2, r2, t2 = Cache.read_result(compare_fn2)

    nospeed = get_no_speed_region(no_speed_fn, y_num)
    index1 = np.in1d(t1, compare_diffs)
    index2 = np.in1d(t2, compare_diffs)

    print "fn:", compare_fn1, ",rmse:", RMSE(p1[index1], r1[index1], nospeed), ",compute: ", index1.sum()
    print "fn:", compare_fn2, ",rmse:", RMSE(p2[index2], r2[index2], nospeed), ",compute: ", index2.sum()


def main():
    compare_big_change_predict("../../data/2016/all/48_48_20_LinearInterpolationFixed_diff.csv",
                               "../../result/speed.c5.p3.t1.resunit6.lr0.0002.External.MMN_predict_rmse2.24067",
                               "../../result/testMyModel4(ernn_h64_l2_step5)_speed.c5.p3.t1.resunit6.lr0.0002.External.MMN_predict_rmse2.22299",
                               "../../data/2016/all/48_48_20_noSpeedRegion_0.05")


if __name__ == '__main__':
    main()
