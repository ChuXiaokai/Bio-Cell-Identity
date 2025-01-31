# encoding: utf8
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from Trans import load
from FindDot import GMMFind, compare, calc_speed
import input

"""
定义参数
"""


dir = input.directory
prefix = input.prefix
num_imgs = len(os.listdir(dir))
threshold = float(input.thedshold)
pic_format = input.pic_type


# 读取所有的图片
Clusters = []
center = []
for i in range(num_imgs):


    if i < 10:
        i = '000'+str(i)
    elif 10<= i < 100:
        i = '00' + str(i)
    elif i >= 100:
        i = '0'+str(i)

    img = load(dir+prefix + str(i) + '.'+pic_format, threshold=threshold, pic_format=pic_format)

    center = [img.shape[0]/2, img.shape[1]/2]
    # img = load('imgs/1-white/visg-eb1b00'+str(i)+'.png', threshold=threshold)
    cls = GMMFind(img)
    Clusters.append(cls)



# 判断任意两张图点的对应关系
Seq = []
for i in range(0, num_imgs-1):
    corresponding = compare(Clusters[i], Clusters[i+1])
    Seq.append(corresponding)


# 生成轨迹
import pickle
pickle.dump(Seq, open('Seq.pk', 'wb'))


trajectories = []
for i in range(0, len(Seq)-1):
    s = Seq[i]
    if len(trajectories) == 0:
        for c, next in s:
            trajectories.append([0, [c, next], 1])  # [start_img_index, [current_n, next_node]]
    else:
        for c, next in s:

            # 找有没有已经出现过的
            flag = 0
            for j, each_tra in enumerate(trajectories):
                [start_img_index, tra, end_img_index] = each_tra

                # 该层是该条轨迹的上一层，并且节点对应
                if end_img_index == i and tra[-1] == c:
                    tra.append(next)
                    trajectories[j] = [start_img_index, tra, i+1]
                    flag = 1
                    break

            # 新出现的一个轨迹
            if flag == 0:
                trajectories.append([i, [c, next], i+1])

inwards = 0
outwards = 0
inwards_speeds = []
outwards_speeds = []
for t in trajectories:
    first_img = t[0]
    last_img = t[-1]
    first_node = t[1][0]
    last_node = t[1][-1]

    first_location = Clusters[first_img][first_node][0]
    last_location = Clusters[last_img][last_node][0]

    # cosine similarity
    x_center = [center[0]-first_location[0], center[1]-first_location[1]]
    x_y = [last_location[0]-first_location[0], last_location[1]-first_location[1]]

    if x_center[0]*x_y[0] + x_center[1]*x_y[1] > 0:
        sp = calc_speed(t, Clusters)
        if 0 < sp < 1:
            inwards += 1
            inwards_speeds.append(sp)
    else:
        sp = calc_speed(t, Clusters)
        if 0 < sp < 1:
            outwards += 1
            outwards_speeds.append(sp)
print('{2}:  inwards:{0}; outwards:{1}'.format(inwards, outwards, dir))
# inwards_speeds = pd.Series(inwards_speeds)
# outwards_speeds = pd.Series(outwards_speeds)
# print('inwards')
# print(inwards_speeds.describe())
# print('outwards')
# print(outwards_speeds.describe())






