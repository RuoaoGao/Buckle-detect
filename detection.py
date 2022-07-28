# -*- coding: utf-8 -*-

import cv2
import dilate,erode,Canny
import numpy as np

def fillHole(img_dilate): #填充内部孔洞函数
    h, w = img_dilate.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=img_dilate.copy()
    #漫水填充从(0, 0)点开始
    cv2.floodFill(im_floodfill, mask, (0,0), 0)
    #反转漫水填充图像
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #将二值图与上一步图像求交集
    im_out = cv2.bitwise_and(img_dilate, im_floodfill_inv)
    return im_out

def canny(img):
    # 梯度计算
    img_d,Theta,dx,dy = Canny.Sobel(img,3)
    #非极大值抑制
    img_n = Canny.NMS(img_d,dx,dy)
    #双阈值处理
    img_dt = Canny.double_threshold(img_n,0.2,0.3)
    #二值化
    img_out = Canny.binaryzation(img_dt)
    img_out=img_out.astype(np.uint8)
    return img_out

def nms(dst,r):
    H,W = dst.shape
    dst1 = np.zeros((H,W))
    r=10
    corner = []
    for j in range(r,W-r):
        for i in range(r,H-r):
            if dst[i,j]< 0.01*dst.max():
                dst1[i,j]=0
            elif dst[i,j]<dst[i-r:i+r,j-r:j+r].max():
                dst1[i,j]=0
            else:
                dst1[i,j]=dst[i,j]
                corner=corner+[(j,i)]
    return dst1,corner

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param): # 标注程序，记录鼠标点击时的坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        inisial_location.append([x,y]) # 卡扣坐标
        cv2.circle(img2, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img2)

def lock_location(p1,p2,x1,y1,x2,y2,theta): # 卡扣定位函数
    '''
    :param p1: 标准件卡扣x
    :param p2: 标准件卡扣y
    :param x1: 标准件中心x
    :param y1: 标准件中心y
    :param x2: 待测件中心x
    :param y2: 待测件中心y
    :param theta: 旋转角度
    '''
    a = np.cos(theta)
    b = np.sin(theta)
    x = x1-x2
    y = y1-y2
    m = np.array([[a,-b,(1-a)*x+y*b],[b,a,(1-a)*y-x*b],[0,0,1]]) # 旋转矩阵，文献里的好像有问题

    after_rotate = np.dot([p1,p2,1],m.T)+ np.array([x,y,1]) # 旋转后的坐标
    after_rotate1 = np.trunc(after_rotate) # 取整
    after_rotate2 = after_rotate1.astype(int) # 取整
    return after_rotate2

def buckle_detect(corner,L,threshpic,originpic): # 卡扣检测
    '''
    :param corner: 卡扣坐标
    :param L: 检测范围大小
    :param threshpic: 待检测件的二值图
    :param originpic: 待检测件原图
    :return:
    '''

    n = 10
    for i in range(len(corner)):
        around_pixel = threshpic[corner[i][1]-L//2:corner[i][1]+L//2+1,corner[i][0]-L//2:corner[i][0]+L//2+1]
        S = around_pixel.sum() # 像素和
        if S >= n*255: # 如果像素和大于阈值
            result = cv2.rectangle(originpic,(corner[i][0]-L//2,corner[i][1]-L//2),(corner[i][0]+L//2,corner[i][1]+L//2),(0,255,0),2)
            #                                                                     矩形对角线上两点坐标
        else:
            result = cv2.rectangle(originpic,(corner[i][0]-L//2,corner[i][1]-L//2),(corner[i][0]+L//2,corner[i][1]+L//2),(0,0,255),2)

    return result

def detect(img):
    img_gau = cv2.GaussianBlur(img,(3,3), 1.5) #高斯滤波
    ret,img_thr = cv2.threshold(img_gau,185 , 255, cv2.THRESH_BINARY) #二值化
    img_dilate = dilate.Dilate(img_thr, 3) #膨胀
    img_out = fillHole(img_dilate) #空洞填充
    img_out = erode.Erode(img_out, 3) #膨胀
    img_edge = canny(img_out)
    im, contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #提取轮廓
    for c in range(len(contours)):
        rect = cv2.minAreaRect(contours[c])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    center=(int(cv2.moments(contours[0])['m10']/cv2.moments(contours[0])['m00']), #重心位置
            int(cv2.moments(contours[0])['m01']/cv2.moments(contours[0])['m00']))
    img_out = erode.Erode(img_out,21) #腐蚀小突起
    img_out = np.float32(img_out)
    dst = cv2.cornerHarris(img_out,3,15,0.04) #寻找角点
    dst1,corner = nms(dst, 6) #非极大值抑制

    xy=[0,0,0,0]  #[xl,yl,xr,yr]
    for i in range(len(corner)): #寻找水平线
        if corner[i][0]<center[0]:
            xy[0] = xy[0] +corner[i][0]/2 
            xy[1] = xy[1] +corner[i][1]/2
        else:
            xy[2] = xy[2] +corner[i][0]/2 
            xy[3] = xy[3] +corner[i][1]/2
    xy=np.round(np.int32(xy))
    k = (xy[0]-xy[1])/(xy[2]-xy[3]) #斜率
    theta = np.arctan(k)  #倾斜角度

    return box,center,corner,theta,img_thr  #外框、重心、角点、倾斜角、二值化图
if __name__=='__main__':
    image1 = cv2.imread('D:/mechine vision/finaltask/img/01parts.png') # 对两张图片同时进行以下过程
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #灰度图
    image2 = cv2.imread('D:/mechine vision/finaltask/img/0parts.png') # 标准件
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) #灰度图
    
    box,center1,corner,theta1,img_thr= detect(img1) 
    _,center,_,theta0,_ = detect(img2) #标准件
    theta = theta1 - theta0

    inisial_location=[] # 标准件卡扣坐标
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while(1):
        cv2.imshow("image", img2)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()
    
    buckle_corner = [] # 卡口坐标
    for i in range(len(inisial_location)): # 遍历所有卡扣坐标
        after_rotate_location = lock_location(inisial_location[i][0],inisial_location[i][1],center1[0],center1[0],center[0],center[0],theta) # 计算旋转后的卡扣坐标
        buckle_corner.append([after_rotate_location[0],after_rotate_location[1]])
    final_pic = buckle_detect(buckle_corner,20,img_thr,image1) # 判断是否有卡扣
    
    cv2.circle(final_pic,center, 2, (255, 0, 0), 2, 8, 0) #绘制重心
    cv2.drawContours(final_pic,[box],0,(0,255,0),2) #外框
    for i in range(len(corner)): #绘制角点
        cv2.circle(final_pic,corner[i], 2, (255, 255, 0), 2, 8, 0)
    cv2.imshow("detect_result", final_pic)
    cv2.imwrite('result.png',final_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

