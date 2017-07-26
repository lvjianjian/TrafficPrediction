#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-6-14, 16:17

@Description:

@Update Date: 17-6-14, 16:17
"""

import time
import numpy as np
import os
from Paramater import DATAPATH
import h5py
import urllib2


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    # vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


def load_holiday(timeslots, fname=os.path.join(DATAPATH, 'BJ_Holiday.txt')):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]


def load_meteorol(timeslots, fname=os.path.join(DATAPATH, 'BJ_Meteorology.h5')):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data
