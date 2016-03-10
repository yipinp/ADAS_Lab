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
        
    # def temporalFilter(self,imageIn,imgPrev,iir):
         
         




if __name__ == "__main__":
    filename = r"\Daylight_00000.jpeg"
    inputImage = os.getcwd() + filename
    inputT = "JPG"
    prep_jpg = pp.ADAS_Preprocess(inputImage,inputType=inputT)   
    img = prep_jpg.read2DImageFromSequence()
    TNR = TNR_Model()
    imgOut = TNR.spatialFilterFrame(img,2)
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Origin')
    plt.subplot(122)
    plt.imshow(imgOut)
    plt.title('TNR out')
    del(prep_jpg)
    del(TNR)
    
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