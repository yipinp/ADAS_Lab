# -*- coding: utf-8 -*-

"""
Created on Wed Mar 02 10:58:24 2016

@author: yipinp
"""

import cv2
import numpy as np
import os

"""LDW : Implement many ADAS preprocess
     inputType : YUV420,YUV444,GRAY,RGB,JPG
     outputType: RGB,GRAY,Bypass
     Support : YUV420->RGB, YUV444->RGB, GRAY->Bypass, RGB->Bypass, JPG->Bypass
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
            imgYUV = np.zeros((3,self.height,self.width),np.uint8)  
            for i in range(frameSize):
                imgYUV[i/(self.height*self.width)][i/self.width][i%self.width] = imgByteArray[i]
            
            #[height,width,channel] from cv2.cvtColor, I420 YVU,YV12:YUV
            imgRGB = cv2.cvtColor(imgYUV,cv2.COLOR_YUV2RGB)
                   
        
        #return [Channel,width,height]
        return imgRGB    
        
            
    def readImageBypass2D(self):
         if self.outputType != "Bypass":
             pass
         elif self.inputType == "GRAY":
             
             frameSize = self.width * self.height
             imgByteArray = bytearray(self.fid.read(frameSize))
             imgY = np.zeros((self.height,self.width),np.uint8) 
             for i in range(frameSize):
                imgY[i/self.width][i%self.width] = imgByteArray[i]
             return imgY
         else:
             frameSize = self.width * self.height*3
             imgByteArray = bytearray(self.fid.read(frameSize))
             imgRGB = np.zeros((3,self.height,self.width),np.uint8) 
             for i in range(frameSize):
                imgRGB[i/(self.height*self.width)][(i%(self.height*self.width))/self.width][(i%(self.height*self.width))%self.width] = imgByteArray[i]
             return imgRGB                 
                       
    
    def read2DImageFromSequence(self):  
        self.openImageSequence()
        if (self.inputType == "YUV420" or self.inputType == "YUV444") and self.outputType == "RGB":            
            return self.YUV2RGB() 
        elif self.inputType == "JPG":
            return cv2.imread(self.filename)
        else:
            return self.readImageBypass2D()  
            

    def __del__(self):
        if self.fid:
            self.fid.close()            
        

if __name__ == "__main__":
    
    #YUV420
    filename = r"\akiyo_qcif.yuv"
    inputImage = os.getcwd() + filename
    width = 176
    height = 144
    inputType = "YUV420"
    outputType = "RGB"
    prep = ADAS_Preprocess(inputImage,width,height,inputType,outputType)
    cv2.imwrite("rgb_out.png",prep.read2DImageFromSequence())
   # cv2.imshow("test",prep.read2DImageFromSequence())    
    prep = ADAS_Preprocess(inputImage,width,height,inputType,outputType)
    #cv2.imshow("test1",prep.read2DImageFromSequence()) 
    del(prep)
    #JPG
    filename = r"\Daylight_00000.jpeg"
    inputImage = os.getcwd() + filename
    inputT = "JPG"
    prep_jpg = ADAS_Preprocess(inputImage,inputType=inputT)
    #cv2.imshow("test",prep_jpg.read2DImageFromSequence())
    del(prep_jpg)
  
    
