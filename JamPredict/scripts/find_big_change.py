#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-8, 10:05

@Description:

@Update Date: 17-9-8, 10:05
"""
import time
import os
from jampredict.utils import *
from jampredict.feature import Data
import pandas as pd

is_mmn = False
month = "all"


def main():
    print("loading data...")
    ts = time.time()
    fn = "48_48_20_MaxSpeedFillingFixed_5"
    datapath = os.path.join(Paramater.DATAPATH, "2016", month)
    datapath = os.path.join(datapath, fn)
    noConditionRegionsPath = os.path.join(datapath, "48_48_20_noSpeedRegion_0.05")
    datas, times, x_num, y_num, interval, startTime, endTime, nospeed_regions = \
        Data.loadRawData(datapath, noConditionRegionsPath, isComplete=False)

    datas = datas[:, 0]
    diff = []
    for i in range(1, len(times)):
        diff.append(np.sqrt(np.sum((datas[i] - datas[i - 1]) ** 2)))

    df_diff = pd.DataFrame({"time": times[1:], "diff": diff}).sort(columns="diff", ascending=False)
    df_diff.to_csv("./data/2016/all/" + fn +"_diff.csv")

    print df_diff
    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::72]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))


if __name__ == '__main__':
    main()
