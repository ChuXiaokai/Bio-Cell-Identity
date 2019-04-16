# encoding: utf8
"""
找到每张图片中的亮点
可以用GMM模型
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

USE_CENTER_AS_LOC = True
min_cls = 2  # 一个聚类的最小单元
threshold = 250
threshold = threshold / 1000. /60 *5 /0.142

def judge(img, FLAG, stack, cur_x, cur_y, neighbor_x, neighbor_y, num_nodes, cluster):
    if neighbor_x < 0 or neighbor_x >= img.shape[0] or neighbor_y <0 or neighbor_y >= img.shape[1]:
        return FLAG, stack, num_nodes, img, cluster

    if FLAG[neighbor_x][neighbor_y] == 1 and img[neighbor_x][neighbor_y] <= img[cur_x][cur_y]:
        FLAG[neighbor_x][neighbor_y] = 0
        stack.append([neighbor_x, neighbor_y])
        cluster.append([neighbor_x, neighbor_y])
        num_nodes += 1
    return FLAG, stack, num_nodes, img, cluster


def gmm(img, x, y, FLAG):

    # 以(x, y)为中心点，按高斯分布
    current_center = [x, y]
    num_nodes = 1 # 这个族里面有多少个点

    cluster = [current_center]
    stack = [current_center]
    while len(stack) > 0:
        x, y = stack.pop()

        # 上下左右八个格子
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x-1, y-1, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x-1, y, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x - 1, y +1, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x, y - 1, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x, y + 1, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x+1, y - 1, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x+1, y, num_nodes, cluster)
        FLAG, stack, num_nodes, img, cluster = judge(img, FLAG, stack, x, y, x + 1, y+1, num_nodes,cluster)


    FLAG[x][y] = 0
    for x, y in cluster:
        img[x][y] = 0

    # 计算中心点
    center_location = np.mean(cluster, axis=0)

    return img, FLAG, num_nodes, center_location



def GMMFind(img):

    """建立标记点"""
    FLAG = np.zeros(img.shape)  # 标记是否每个点都被对应好了
    FLAG[img > 0] = 1  # 将所有有颜色的点坐标标记为1

    # 建立族群点索引
    cluster_indices = []

    while True:
        if np.sum(img) == 0:
            break

        # find the index of the largest dot
        index = np.argmax(img)
        x = int(index/img.shape[1])
        y = index - img.shape[1]*x


        # gmm找到族
        img, FLAG, num_nodes, center_location = gmm(img, x, y, FLAG)
        if USE_CENTER_AS_LOC:  # 如果使用族群的位置中心作为中心
            cluster_indices.append([center_location, num_nodes])
        else:  # 使用族群最亮的点作为中心
            cluster_indices.append([[x, y], num_nodes])

        # io.imshow(img)
        # plt.show()
    # print(len(cluster_indices))

    # 将图中族群内节点数目 < 2的剔除
    def bigger_cls(x):
        return x[1] > min_cls
    cluster_indices = list(filter(bigger_cls, cluster_indices))

    return cluster_indices

def _distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)

"""找前后两张图内 节点的对应关系"""
def compare(cls1, cls2):
    correspondings = []
    for i in range(len(cls1)):
        [x1, y1], num_nodes_1 = cls1[i]

        # 用来记录的
        neighbor_center = []
        min_dis = 100000000
        min_num_nodes = 100000000
        min_index = 100000000

        # 找最近的点
        for j in range(len(cls2)):
            [x2, y2], num_nodes_2 = cls2[j]

            # 0.5是根据先验知识，即点的移动速度最大为600nm/min
            if _distance([x1, y1], [x2,y2]) < min_dis and _distance([x1, y1], [x2,y2]) < threshold:  # 找距离最近的 且 控制范围
                min_dis = np.abs(x1-x2) + np.abs(y1-y2)
                min_num_nodes = num_nodes_2
                min_index = j

            elif _distance([x1, y1], [x2,y2]) == min_dis: # 距离相等比较族大小
                if np.abs(num_nodes_1-num_nodes_2) < np.abs(num_nodes_1-min_num_nodes):
                    min_dis = np.abs(x1 - x2) + np.abs(y1 - y2)
                    min_num_nodes = num_nodes_2
                    min_index = j

        if min_index != 100000000:  # 找到了对应点
            correspondings.append([i, min_index])

    # print(correspondings)
    return correspondings



pixel_size = 0.142  # um
interval = 5  # s
def calc_speed(t, Clusters):
    """
    :param t: 一条轨迹
    :return:
    """
    # 计算每个点的平均速度
    pic_list = [img_index for img_index in range(t[0], t[-1] + 1)]  # 这条轨迹所属的图片序列
    node_list = t[1]  # 对于每张图片来说，该轨迹点对应的index
    location_list = []

    # 获取该点在每张图片中的位置
    for i in range(len(pic_list)):
        location = Clusters[pic_list[i]][node_list[i]][0]
        location_list.append(location)
    location_list = np.array(location_list)

    # 前后两张图片计算速度
    speeds = []
    for i in range(0, len(location_list) - 1):
        sp = np.linalg.norm(location_list[i + 1] - location_list[i])
        sp = sp * pixel_size / interval * 60.
        speeds.append(sp)
    node_mean_speeds = np.mean(speeds)
    return node_mean_speeds
