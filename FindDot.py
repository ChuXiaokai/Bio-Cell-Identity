# encoding: utf8
"""
找到每张图片中的亮点
可以用GMM模型
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

min_cls = 2  # 一个聚类的最小单元

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

    return img, FLAG, num_nodes



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
        img, FLAG, num_nodes = gmm(img, x, y, FLAG)
        cluster_indices.append([[x, y], num_nodes])

        # io.imshow(img)
        # plt.show()
    # print(len(cluster_indices))

    # 将图中族群内节点数目 < 2的剔除
    def bigger_cls(x):
        return x[1] > min_cls
    cluster_indices = list(filter(bigger_cls, cluster_indices))

    return cluster_indices



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

            if np.abs(x1-x2) + np.abs(y1-y2) < min_dis and np.abs(x1-x2) + np.abs(y1-y2) < 20+20:  # 找距离最近的 且 控制范围
                min_dis = np.abs(x1-x2) + np.abs(y1-y2)
                min_num_nodes = num_nodes_2
                min_index = j

            elif np.abs(x1-x2) + np.abs(y1-y2) == min_dis: # 距离相等比较族大小
                if np.abs(num_nodes_1-num_nodes_2) < np.abs(num_nodes_1-min_num_nodes):
                    min_dis = np.abs(x1 - x2) + np.abs(y1 - y2)
                    min_num_nodes = num_nodes_2
                    min_index = j

        if min_index != 100000000:  # 找到了对应点
            correspondings.append([i, min_index])

    # print(correspondings)
    return correspondings