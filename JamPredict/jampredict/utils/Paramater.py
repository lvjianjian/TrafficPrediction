#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-6-14, 16:06

@Description:

@Update Date: 17-6-14, 16:06
"""

import os
import platform

path = os.environ.get("JamPredictPath")
if (path is None):
    sysstr = platform.system()
    if(sysstr =="Windows"):
        PROJECTPATH = "E:\\ZhongjianLv\\project\\jamprediction\\RoadStatistics\\JamPredict\\"
        # print ("Call Windows tasks")
    elif(sysstr == "Linux"):
        # print ("Call Linux tasks")
        PROJECTPATH = "/home/zhongjianlv/TrafficPrediction/JamPredict/"
else:
    PROJECTPATH = path

DATAPATH = PROJECTPATH + "data"

# 第一层为拥挤度(0-4), 第二层为轨迹数量
Z_NUM = 2

CONDITION_NO = 0
CONDITION_JAM = 1
CONDITION_SLIGHT_JAM = 2
CONDITION_GENERAL = 3
CONDITION_CLEAR = 4