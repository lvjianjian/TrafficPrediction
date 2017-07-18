#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-6-14, 16:06

@Description:

@Update Date: 17-6-14, 16:06
"""

import os

path = os.environ.get("JamPredictPath")
if (path is None):
    PROJECTPATH = "/home/zhongjianlv/TrafficPrediction/JamPredict/"
else:
    PROJECTPATH = path

DATAPATH = PROJECTPATH + "data/"

# 第一层为拥挤度(0-3), 第二层为轨迹数量
Z_NUM = 2
