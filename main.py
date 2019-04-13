# encoding: utf8
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from Trans import load
from FindDot import GMMFind, compare

"""
定义参数
"""
dir = sys.argv[1]
prefix = sys.argv[2]
num_imgs = len(os.listdir(dir))
threshold = float(sys.argv[3])
pic_format = sys.argv[4]
pixel_size = 0.142  # um
interval = 5  # s

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

# print('================')
# for t in trajectories:
#     print(t)

inwards = 0
outwards = 0
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
        inwards += 1
    else:
        outwards += 1
print('{2}:  inwards:{0}; outwards:{1}'.format(inwards, outwards, dir))

# 计算每个点的平均速度
ALL_SPEED = []
for t in trajectories:
    pic_list = [img_index for img_index in range(t[0], t[-1]+1)]  # 这条轨迹所属的图片序列
    node_list = t[1]  # 对于每张图片来说，该轨迹点对应的index
    location_list = []

    # 获取该点在每张图片中的位置
    for i in range(len(pic_list)):
        location = Clusters[pic_list[i]][node_list[i]][0]
        location_list.append(location)
    location_list = np.array(location_list)

    # 前后两张图片计算速度
    speeds = []
    for i in range(0, len(location_list)-1):
        sp = np.linalg.norm(location_list[i+1] - location_list[i])*pixel_size/interval*60
        speeds.append(sp)
    node_mean_speeds = np.mean(speeds)
    ALL_SPEED.append(node_mean_speeds)

import pandas as pd
print(ALL_SPEED)
ALL_SPEED = pd.Series(ALL_SPEED)
print(ALL_SPEED.describe())

# print('平均速度为:{0} um/min'.format(np.mean(ALL_SPEED)))







