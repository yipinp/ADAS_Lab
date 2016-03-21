# -*- coding: utf-8 -*-

"""
Created on Wed Mar 09 10:58:24 2016

@author: yipinp
"""

import cv2
import numpy as np
import os
from PIL import Image
import preprocessor as pp
from matplotlib import pyplot as plt 

class TNR_Model:
    def spatialFilterFrame(self,imageIn,mode):
        if mode == 0: 
            imageOut = cv2.GaussianBlur(imageIn,(3,3),0)
        elif mode == 1:
            imageOut = cv2.GaussianBlur(imageIn,(5,5),0)
        else:
            kernel = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],np.float32)/256
            imageOut = cv2.filter2D(imageIn,-1,kernel)
        return imageOut
    
    def SAD_block(self,array0,array1,x_center,y_center,isPictureBoundary,component,size=3):
        if isPictureBoundary:
            sad = np.abs(array0[y_center,x_center,component] - array1[y_center,x_center,component])
        else:
            sad = 0
            for i in range(size):
                for j in range(size):
                    sad += np.abs(array0[y_center-size//2 + i ,x_center-size//2 + j,component] - array1[y_center-size//2 + i ,x_center-size//2 + j,component])
            
        return sad
        
                             
    def setAlpha(self,sad,alpha,j,i,isBoundary,channel,light=1,ST=0,iir = 768):
        if light == 0 :
            alpha[i,j,channel] = 1024 - min(1024,4.5*sad)
        elif light == 1 and ST == 0:
            alpha[i,j,channel] = 1024 - min(1024,4*sad)
        elif light ==1 and ST ==1:
            alpha[i,j,channel] = 1024 - min(1024,3.0*sad)
        else:
            alpha[i,j,channel] = 1024 - min(1024,2.75*sad)
        
        #average alpha, top-left,top,top-right,left,current
        if light > 0 and not(isBoundary):
            alpha[i,j,channel] =( (alpha[i,j,channel]<<1) + (alpha[i,j-1,channel]<<1) +(alpha[i-1,j,channel]<<1) + alpha[i-1,j-1,channel] +alpha[i-1,j+1,channel])>> 3
            
            
        alpha[i,j,channel] = (iir * alpha[i,j,channel]) >> 10
        alpha[i,j,channel] = 512
        
        
 
    def getAlphaFromSAD(self,imageIn,imagePrev,height,width,channel,alpha):  
        sad = 0
        for i in range(height):
            for j in range(width):
                isBoundary = (j == 0) or (i == 0) or (j == width -1) or (i == height -1)
                sad = self.SAD_block(imageIn,imagePrev,j,i,isBoundary,channel) 
                self.setAlpha(sad,alpha,j,i,isBoundary,channel)

       
    def alphaBlending(self,imageIn,imagePrev,alpha,j,i,channel,imageOut):
         imageOut[i,j,channel] =  (alpha[i,j,channel]*imagePrev[i,j,channel]+(1024-alpha[i,j,channel])*imageIn[i,j,channel])>>10
        
        
    def betaBlending(self,imageSpatial,temporalFilter,j,i,channel,alpha,image3DOut):
        beta = 1024 - alpha[i,j,channel]
        beta = min(beta,400)
        image3DOut[i,j,channel] = (beta*imageSpatial[i,j,channel] + (1024-beta)*imageSpatial[i,j,channel]) >> 10
            
    def temporalFilterMA(self,imageSpatial,imageIn,imagePrev,height,width,channel):
         alpha=np.zeros([height,width,channel],np.uint32)
         imageTemporal = np.zeros([height,width,channel],np.uint8)
         image3DOut = np.zeros([height,width,channel],np.uint8)
         ST = 0
         for k in range(channel):       
             self.getAlphaFromSAD(imageSpatial,imagePrev,height,width,k,alpha)
             
         for i in range(height):
            for j in range(width):
                for k in range(channel):
                    self.alphaBlending(imageIn,imagePrev,alpha,j,i,k,imageTemporal)
                    self.betaBlending(imageSpatial,imageTemporal,j,i,k,alpha,image3DOut)
         if ST == 0:            
            image3DOut = imageTemporal
         #Beta blending
         return image3DOut
         




if __name__ == "__main__":
    filename = r"\Daylight_00000.jpeg"
    inputImage = os.getcwd() + filename
    inputT = "JPG"
    prep_jpg = pp.ADAS_Preprocess(inputImage,inputType=inputT) 
    TNR = TNR_Model()
    for nframes in range(10):
        print("frame is :%s"%nframes)
        img = prep_jpg.read2DImageFromSequence()
        height,width,channel = img.shape
        if nframes == 0 :
            imgPrev = np.zeros([height,width,channel],np.uint8)
        imgOut = TNR.spatialFilterFrame(img,2)    
        img3DOut = TNR.temporalFilterMA(imgOut,img,imgPrev,height,width,channel)
        imgPrev = img3DOut
        cv2.imwrite("TNR_test_" + str(nframes) + ".jpg",img3DOut)
       #plt.subplot(121)
       # plt.imshow(img)
       # plt.title('Origin')
       # plt.subplot(122)
       # plt.imshow(img3DOut)
       # plt.title('TNR out')
    del(prep_jpg)
    del(TNR)
    """   
    #filename = r"\akiyo_qcif.yuv"
    filename = r"\out.yuv"
    inputImage = os.getcwd() + filename
    width = 1920
    height = 1080
    inputType = "YUV420"
    outputType = "RGB"
    prep = pp.ADAS_Preprocess(inputImage,width,height,inputType,outputType)
    img = prep.read2DImageFromSequence()
    TNR = TNR_Model()
    imgOut = TNR.spatialFilterFrame(img,1)
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Origin')
    plt.subplot(122)
    plt.imshow(imgOut)
    plt.title('TNR out')
    cv2.imwrite("out.jpg",imgOut)
    cv2.imwrite("in.jpg",img)
    del(prep)
    del(TNR)
   """