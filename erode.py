# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 10:38:24 2022

@author: ASUS
"""
import cv2
import numpy


def Erode(image,r):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W =image.shape
    result = image.copy()
    r1=int((r-1)/2)
    for h in range(r1,H-1-r1):
        for w in range(r1,W-r1-1):
            window = image[h-r1:h+r1,w-r1:w+r1] #创建窗口
            result[h,w] =numpy.max(window)  #选取窗口中的最大值
    return result

if __name__=='__main__':
    im = cv2.imread('origin_image.png',cv2.IMREAD_GRAYSCALE)
    im_gn = cv2.imread('gaussian_noise.png',cv2.IMREAD_GRAYSCALE)
    im_pn = cv2.imread('pepper_noise.png',cv2.IMREAD_GRAYSCALE)
     
    im =Erode(im)
    im_gn = Erode(im_gn,3)
    im_pn = Erode(im_pn,3)
    
    cv2.imwrite('33_erode_none.png',im)      
    cv2.imwrite('33_erode_gaussian.png',im_gn)   
    cv2.imwrite('33_erode_pepper.png',im_pn)   
