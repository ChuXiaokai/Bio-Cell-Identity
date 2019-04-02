# encoding: utf8
"""将图片转为黑色图片"""

from skimage import io
import matplotlib.pyplot as plt

# 0->黑 100->亮
def load(img_path, threshold=150, pic_format='png'):
    if pic_format=='png':
        img = io.imread(img_path, as_gray=True)
    else:
        img = io.imread(img_path)
    # print(img)
    img[img < threshold] = 0
    return img