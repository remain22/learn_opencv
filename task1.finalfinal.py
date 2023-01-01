import cv2 as cv
import numpy as np
from skimage.exposure import rescale_intensity


def convolve(img, fil, mode='same'):  # 分别提取三个通道

    if mode == 'fill':
        h = fil.shape[0] // 2
        w = fil.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w), (0, 0)), 'constant')
    conv_b = _convolve(img[:, :, 0], fil)  # 然后去进行卷积操作
    conv_g = _convolve(img[:, :, 1], fil)
    conv_r = _convolve(img[:, :, 2], fil)

    dstack = np.dstack([conv_b, conv_g, conv_r])  # 将卷积后的三个通道合并
    return dstack  # 返回卷积后的结果


def _convolve(img, fil):
    fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
    fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

    conv_heigh = img.shape[0] - fil.shape[0] + 1  # 确定卷积结果的大小
    conv_width = img.shape[1] - fil.shape[1] + 1

    conv = np.zeros((conv_heigh, conv_width), dtype='uint8')

    for i in range(conv_heigh):
        for j in range(conv_width):  # 逐点相乘并求和得到每一个点
            conv[i][j] = wise_element_sum(img[i:i + fil_heigh, j:j + fil_width], fil)
    return conv


def wise_element_sum(img, fil):
    res = (img * fil).sum()
    # res = rescale_intensity(res, in_range=(0, 255))
    # res = (res * 255).astype("uint8")
    if (res < 0):
        res = 0
    elif res > 255:
        res = 255
    return res


img = cv.imread("C:/Users/alien/Desktop/learn/boat1.jpg")  # 在这里读取图片
cv.imshow('image window',img)  # 显示读取的图片


# 卷积核应该是奇数行，奇数列的
if __name__ == "__main__":
    fuilter1 = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    fuilter2 = [[-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    fuilter3 = [[-1, -1, -1, -1, -1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]]
    fuilter4 = [[-1, -1, -1, -1, -1], [0, 0, 2, 0, 0], [2, 2, 2, 2, 2]]
    fil = np.array(fuilter4)
    res = convolve(img, fil, 'fill')
    print("img shape :" + str(img.shape))
    cv.imshow('image window',res)  # 显示卷积后的图片
    print("res shape :" + str(res.shape))
    key = cv.waitKey()
    if key == ord("s"):
        cv.imwrite("boat_FILTER7.jpg", res)
    cv.destroyAllWindows()