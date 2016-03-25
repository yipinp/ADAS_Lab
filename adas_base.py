# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:58:24 2016

@author: yipinp
"""

import cv2
import numpy as np
import os
from PIL import Image
import random

class Adas_base :
    def __init__(self,filename,width=0,height=0,inputType="YUV420",outputType = "RGB"):    
        self.filename = filename
        self.inputType = inputType
        self.outputType = outputType
        self.width = width
        self.height = height    
    
    
    def __del__(self):
        if self.fid:
            self.fid.close()
    
    
    
    """Color space convert
     inputType : YUV420,YUV,GRAY,RGB,JPG
     outputType: RGB,Gray
     Support : YUV420/JPG/YUV->RGB   RGB->YUV, GRAY/YUV420/yuv/RGB->Gray, 
    """ 
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
            
            
    """
        noise model
    """

    def saltAndPepper(self,img,percetage,width,height):
        imgsrc = bytearray(img)
        NoiseNumY = int(width*height*percetage)
        NoiseNumUV = int(width*height/4 *percetage)
        imgdst = imgsrc
        print("%s %s"% (width,height))

        #Y channel
        for i in range(NoiseNumY):
            randx    = random.randint(0,width-1)
            randy    = random.randint(0,height-1)
            if random.randint(0,1):
                imgdst[randy*width + randx] =  0
            else :
                imgdst[randy*width + randx] =  255

        #UV channel
        for i in range(NoiseNumUV):
            randx    = random.randint(0,width//2-1)
            randy    = random.randint(0,height//2-1)
            if random.randint(0,1):
                imgdst[int(randy*width//2 + randx + width*height)] =  0
                imgdst[int(randy*width//2 + randx + width*height + width*height/4)] =  0
            else :
                imgdst[int(randy*width//2 + randx + width*height)] =  255
                imgdst[int(randy*width//2 + randx + width*height +  width*height/4)] =  255

        #convert to numpy array
        imgArray = np.zeros((3*height//2,width),np.uint8)

        for i in range(len(imgdst)):
            imgArray[i/width][i%width] = imgdst[i]

        #Convert YUV to RGB,[height,width,channel]
        imgRGB = cv2.cvtColor(imgArray,cv2.COLOR_YUV2BGR_I420) 

        return imgRGB            
            
    
    
    
    """
       Temporal noise reduction algorithm 
    """
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
    
    #YUV420->RGB
    filename = r"\akiyo_qcif.yuv"
    inputImage = os.getcwd() + filename
    width = 176
    height = 144
    inputType = "YUV420"
    outputType = "RGB"
    prep = Adas_base(inputImage,width,height,inputType,outputType)
    rgb = prep.read2DImageFromSequence()
    cv2.imwrite("rgb_out.png",rgb)
    f = open("rgb_raw.bin","wb+")
    f.write(rgb)
    f.close()
    #cv2.imshow("test",prep.read2DImageFromSequence())    
    del(prep)       
    