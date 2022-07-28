# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:15:46 2022

@author: ASUS
"""
import cv2
import numpy


def Dilate(image,r):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W =image.shape
    result = image.copy()
    r1=int((r-1)/2)
    for h in range(r1,H-r1-1):
        for w in range(r1,W-r1-1):
            window = image[h-r1:h+r1,w-r1:w+r1] #创建窗口
            result[h,w] =numpy.min(window)  #选取最小值
    return result
if __name__ == '__main__':
    
    im = cv2.imread('origin_image.png',cv2.IMREAD_GRAYSCALE)
    im_gn = cv2.imread('gaussian_noise.png',cv2.IMREAD_GRAYSCALE)
    im_pn = cv2.imread('pepper_noise.png',cv2.IMREAD_GRAYSCALE)
     
    im =Dilate(im,3)
    im_gn = Dilate(im_gn,3)
    im_pn = Dilate(im_pn,3)
    
    cv2.imwrite('33_dilate_none.png',im)      
    cv2.imwrite('33_dilate_gaussian.png',im_gn)   
    cv2.imwrite('33_dilate_pepper.png',im_pn)   