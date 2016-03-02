# -*- coding: utf-8 -*-

"""
Created on Wed Mar 02 10:58:24 2016

@author: yipinp
"""

import cv2
import numpy as np
import os
from PIL import Image

"""LDW : Implement many ADAS preprocess
     inputType : YUV420,YUV,GRAY,RGB,JPG
     outputType: RGB,Gray
     Support : YUV420/JPG/YUV->RGB   RGB->YUV, GRAY/YUV420/yuv/RGB->Gray, 
"""
class ADAS_Preprocess: 
    def __init__(self,filename,width=0,height=0,inputType="YUV420",outputType = "RGB"):    
        self.filename = filename
        self.inputType = inputType
        self.outputType = outputType
        self.width = width
        self.height = height

    
    
    def openImageSequence(self):
        if os.path.isfile(self.filename):
            self.fid = open(self.filename,"rb")
        else:
            print("Fail to open the file %s" %(self.filename))
            
    
    def YUV2RGB(self):
        if self.inputType == "YUV420":
            frameSize = int(self.width*self.height*3/2)
            imgByteArray = bytearray(self.fid.read(frameSize))
            imgYUV = np.zeros((int(self.height + self.height/2),self.width),np.uint8) 
            for i in range(frameSize):
                imgYUV[i/self.width][i%self.width] = imgByteArray[i]
            
            #[height,width,channel] from cv2.cvtColor, I420 YVU,YV12:YUV
            imgRGB = cv2.cvtColor(imgYUV,cv2.COLOR_YUV2BGR_I420)
                     
        else:
            frameSize = self.width*self.height*3
            imgByteArray = bytearray(self.fid.read(frameSize))
            imgYUV = np.zeros((self.height,self.width,3),np.uint8)  
            for i in range(frameSize):
                imgYUV[(i%(self.height*self.width))/self.width][i%self.width][i/(self.height*self.width)] = imgByteArray[i]
            
            #[height,width,channel] from cv2.cvtColor, I420 YVU,YV12:YUV          
            imgRGB = cv2.cvtColor(imgYUV,cv2.COLOR_YUV2RGB)
                   
        
        #return [Channel,width,height]
        return imgRGB  
        
        
    def RGB2YUV(self):
        frameSize = self.width*self.height*3
        imgByteArray = bytearray(self.fid.read(frameSize))
        imgRGB = np.zeros((self.height,self.width,3),np.uint8)  
        for i in range(frameSize):
            imgRGB[(i/3)/self.width][(i/3)%self.width][i%3] = imgByteArray[i]
     
        #[height,width,channel] from cv2.cvtColor, I420 YVU,YV12:YUV
        imgYUV = cv2.cvtColor(imgRGB,cv2.COLOR_BGR2YUV)
        return imgYUV            
        
            
    def readImageLumaOnly(self):
         frameSize = self.width * self.height
         if self.outputType != "Gray":
             pass
         elif self.inputType == "Gray":
             skipSize = 0;
         elif self.inputType == "YUV420":
             skipSize = int(self.width * self.height/2)
         elif self.inputType == "YUV":
             skipSize = int(self.width * self.height*2)
         elif self.inputType == "RGB":
             imgYUV = self.RGB2YUV()
             cv2.imwrite("test1.jpg",imgYUV[:,:,0])
             return imgYUV[:,:,0]                        
         
         imgByteArray = bytearray(self.fid.read(frameSize))
         self.fid.read(skipSize)
         imgY = np.zeros((self.height,self.width),np.uint8) 
         for i in range(frameSize):
             imgY[i/self.width][i%self.width] = imgByteArray[i]
         return imgY              
                       
                       
    def read2DImageFromSequence(self):  
        self.openImageSequence()
        if (self.inputType == "YUV420" or self.inputType == "YUV") and self.outputType == "RGB":            
            return self.YUV2RGB() 
        elif self.inputType == "JPG":
            return cv2.imread(self.filename)
        elif self.inputType == "RGB" and self.outputType == "YUV":
            return self.RGB2YUV()
        else:
            return self.readImageLumaOnly()  
            

    def __del__(self):
        if self.fid:
            self.fid.close()            
        

if __name__ == "__main__":
    
    #YUV420->RGB
    filename = r"\akiyo_qcif.yuv"
    inputImage = os.getcwd() + filename
    width = 176
    height = 144
    inputType = "YUV420"
    outputType = "RGB"
    prep = ADAS_Preprocess(inputImage,width,height,inputType,outputType)
    rgb = prep.read2DImageFromSequence()
    cv2.imwrite("rgb_out.png",rgb)
    f = open("rgb_raw.bin","wb+")
    f.write(rgb)
    f.close()
    #cv2.imshow("test",prep.read2DImageFromSequence())    
    del(prep)
    #JPG->RGB
    filename = r"\Daylight_00000.jpeg"
    inputImage = os.getcwd() + filename
    inputT = "JPG"
    prep_jpg = ADAS_Preprocess(inputImage,inputType=inputT)
    cv2.imwrite("test_jpg.jpg",prep_jpg.read2DImageFromSequence())
    del(prep_jpg)
    
    #iYUV420->Y
    filename = r"\akiyo_qcif.yuv"
    inputImage = os.getcwd() + filename
    width = 176
    height = 144
    inputType = "YUV420"
    outputType = "Gray"
    prep = ADAS_Preprocess(inputImage,width,height,inputType,outputType)
    cv2.imwrite("Luma.png",prep.read2DImageFromSequence())
    
    #RGB->Y
    filename = r"\rgb_raw.bin"
    inputImage = os.getcwd() + filename
    width = 176
    height = 144
    inputType = "RGB"
    outputType = "Gray"
    prep_rgb = ADAS_Preprocess(inputImage,width,height,inputType,outputType)
    cv2.imwrite("Luma_rgb.png",prep_rgb.read2DImageFromSequence())
    del(prep_rgb)
