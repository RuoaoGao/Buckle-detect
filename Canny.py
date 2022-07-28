
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:19:06 2022

@author: ASUS
"""
import cv2
import numpy as np
import math

# 定义 lxl 的高斯滤波器
def Gaussian_Filter(img,l):
    # 生成高斯滤波器
    r = (l+1)/2 #卷积核半径
    sigma1 = sigma2 = 1.4
    gau_sum = 0
    gaussian = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            gaussian[i, j] = math.exp((-1/(2*sigma1*sigma2))*(np.square(i-r)
                                + np.square(j-r)))/(2*math.pi*sigma1*sigma2)
            gau_sum =  gau_sum + gaussian[i, j]
    # 归一化处理
    gaussian = gaussian / gau_sum
    # 高斯滤波
    W, H = img.shape
    new_img = np.zeros([W, H])
    r1 = int((l-1)/2)
    for i in range(W):
        for j in range(H):
            if r1-1<i<W-r1 and r1-1<j<H-r1:
                new_img[i, j] = np.sum(img[i-r1:i+r1+1, j-r1:j+r1+1] * gaussian)
            else:
                new_img[i, j] = img[i,j]
    new_img=new_img.astype(np.uint8)
    return new_img

def Sobel(img,l):
    # 生成Sobelx滤波器
    sobelx =np.array([[1,0,-1],  #滤波器
                       [2,0,-2],
                       [1,0,-1]])/8
    
    W, H = img.shape
    r1 = int((l-1)/2)
    # 滤波
    new_img_x = np.zeros([W, H])
    for i in range(W):
        for j in range(H):
            if r1-1<i<W-r1 and r1-1<j<H-r1:
                new_img_x[i, j] = np.sum(np.array(img[i-r1:i+r1+1, j-r1:j+r1+1]) * sobelx)
            else:
                new_img_x[i, j] = img[i,j]  
    
    # 生成Sobely滤波器
    sobely =np.transpose(np.array([[1,0,-1],  #滤波器
                                      [2,0,-2],
                                      [1,0,-1]]))/8
    # 滤波
    new_img_y = np.zeros([W, H])
    for i in range(W):
        for j in range(H):
            if r1-1<i<W-r1 and r1-1<j<H-r1:
                new_img_y[i, j] = np.sum(np.array(img[i-r1:i+r1+1, j-r1:j+r1+1]) * sobely)
            else:
                new_img_y[i, j] = img[i,j]    
    #计算幅值及角度
    new_img = np.zeros([W, H])
    theta = np.zeros([W, H])
    for i in range(W):
        for j in range(H):
            new_img[i, j] = np.sqrt(np.square(new_img_x[i, j]) + np.square(new_img_y[i, j]))
            theta[i,j] = np.arctan2(new_img_x[i,j],new_img_y[i,j])
    new_img=new_img.astype(np.uint8)
    return new_img,theta,new_img_x,new_img_y

def NMS( M, dx, dy):
    d = np.copy(M)
    W, H = M.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W-1, :] = NMS[:, 0] = NMS[:, H-1] = 0

    for i in range(1, W-1):
        for j in range(1, H-1):

            # 如果当前梯度为0，该点就不是边缘点
            if M[i, j] == 0:
                NMS[i, j] = 0

            else:
                gradX = dx[i, j] # 当前点 x 方向导数
                gradY = dy[i, j] # 当前点 y 方向导数
                gradTemp = d[i, j] # 当前梯度点

                # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY) # 权重
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    # g1 g2
                    #    c
                    #    g4 g3
                    if gradX * gradY > 0:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #    g2 g1
                    #    c
                    # g3 g4
                    else:
                        grad1 = d[i-1, j+1]
                        grad3 = d[i+1, j-1]

                # 如果 x 方向梯度值比较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    # g1    
                    # g2 c g4
                    #      g3
                    if gradX * gradY > 0:

                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #      g3
                    # g2 c g4
                    # g1     
                    else:
                        grad1 = d[i+1, j-1]
                        grad3 = d[i-1, j+1]

                # 利用 grad1-grad4 对梯度进行插值
                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4

                # 当前像素的梯度是局部的最大值，可能是边缘点
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp

                else:
                    # 不可能是边缘点
                    NMS[i, j] = 0
    return NMS
#双阈值选取
def double_threshold(NMS,rl,rh):

    W, H = NMS.shape
    DT = np.zeros([W, H])

    # 定义高低阈值
    nms=[]
    for i in range(W):
        m=np.max(NMS[i,:])
        nms.append (m)
    M = np.sum(nms)/W
    TL = rl * M
    TH = rh * M

    for i in range(1, W-1):
        for j in range(1, H-1):
           # 双阈值选取
            if (NMS[i, j] < TL):
                DT[i, j] = 0

            elif (NMS[i, j] > TH):
                DT[i, j] = NMS[i,j]

            #连接
            elif (NMS[i-1, j-1:j+1].any() >TL or NMS[i+1, j-1:j+1].any() >TL
                    or NMS[i, [j-1, j+1]].any() >TL):
                DT[i, j] = TL
            else:
                DT[i, j] = 0
    return DT

#二值化
def binaryzation(img):
    W, H = img.shape
    DT = np.zeros([W, H])

    for i in range(1, W-1):
        for j in range(1, H-1):
            if img[i, j] ==0 :
                DT[i, j] = 0
            else :
                DT[i, j] = 255
    return DT

if __name__=='__main__':
    # 读入灰度图
    img = cv2.imread('lanes.png', 0)
    # cv2.imwrite('lanes_gray.png',img)
    # cv2.imshow('im',img)

    # 高斯滤波降噪
    img_g = Gaussian_Filter(img,3)
    # cv2.imwrite('lanes_gaussianblur.png',img_g)
    # cv2.imshow('img_g',img_g)

    # 梯度计算
    img_d,Theta,dx,dy = Sobel(img_g,3)
    # cv2.imshow('img_d',img_d)
    cv2.imwrite('lanes_sobel.png',img_d)

    #非极大值抑制
    img_n = NMS(img_d,dx,dy)
    # cv2.imshow('img_n',img_n)
    cv2.imwrite('lanes_NMS.png',img_n)

    #双阈值处理
    img_dt = double_threshold(img_n,0.2,0.3)
    # cv2.imshow('img_dt',img_dt)
    cv2.imwrite('lanes_doubel.png',img_dt)

    #二值化
    img_bi = binaryzation(img_dt)
    # cv2.imshow('img_bi',img_bi)
    cv2.imwrite('lanes_binaryzation.png',img_bi)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


