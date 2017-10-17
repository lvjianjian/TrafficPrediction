#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-18, 09:56

@Description:

@Update Date: 17-9-18, 09:56
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
from jampredict.utils.Cache import *
from jampredict.utils import Paramater
from jampredict.feature import Data
import seaborn as ses

if __name__ == '__main__':
    datas, times, x_num, y_num, interval, startTime, endTime, nospeed_regions = \
        Data.loadRawData(os.path.join(Paramater.DATAPATH, "2016/all/48_48_20_LinearInterpolationFixed"),
                         os.path.join(Paramater.DATAPATH, "48_48_20_noSpeedRegion_0.05"), False)
    x_index = 24
    y_index = 24
    datas = datas[:, 0]
    f, ax = plt.subplots(1, 1, figsize=(15, 7))
    print datas[:500, x_index, y_index]
    ses.tsplot(datas[:500, x_index, y_index], ax=ax)
    plt.savefig(os.path.join(Paramater.PROJECTPATH, "fig/test2.jpg"))
