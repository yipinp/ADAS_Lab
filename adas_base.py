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
import copy
import pylab as pl
from scipy import fftpack

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
    
    def saltAndPepperForRGB(self,img,percetage,width,height):
        NoiseNum = int(width*height*percetage)
        for i in range(NoiseNum):
            randx    = random.randint(0,width-1)
            randy    = random.randint(0,height-1)
            #print img.shape,randx,randy
            if random.randint(0,1):
                img[randy,randx,0] =  0
                img[randy,randx,1] =  0
                img[randy,randx,2] =  0
            else :
                img[randy,randx,0] =  255
                img[randy,randx,1] =  255
                img[randy,randx,2] =  255
                
        return img
        

    def saltAndPepperForYUV(self,img,percetage,width,height):
        imgsrc = bytearray(img)
        NoiseNumY = int(width*height*percetage)
        NoiseNumUV = int(width*height/4 *percetage)
        imgdst = imgsrc

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

    """ box-muller"""
    def GaussianWhiteNoiseForRGB(self,imgIn,width,height):
        img = imgIn
        level = 40
        gray = 255
        zu = []
        zv = []
        for i in xrange(0,height):
            for j in xrange(0,width,2):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                z1 = level*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
                z2 = level*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
                zu.append(z1)
                zv.append(z2)
                img[i,j,0] = np.clip(int(img[i,j,0] + z1),0,gray)
                img[i,j+1,0] = np.clip(int(img[i,j+1,0] + z2),0,gray)
                img[i,j,1] = np.clip(int(img[i,j,1] + z1),0,gray)
                img[i,j+1,1] = np.clip(int(img[i,j+1,1] + z2),0,gray)
                img[i,j,2] = np.clip(int(img[i,j,2] + z1),0,gray)
                img[i,j+1,2] = np.clip(int(img[i,j+1,2] + z2),0,gray)
            
       # pl.subplot(211)
       # pl.hist(zu+zv,bins=200,normed=True)
       # pl.subplot(212)
       # pl.psd(zu+zv)
       # pl.show()
        return img
        
    """Add LPF for white Gaussian Noise before adding to original img for real isp processing"""    
    def GaussianWhiteNoiseForRGB2(self,imgIn,width,height):
        noiseImg = np.zeros([height,width])
        level = 40
        for i in xrange(0,height):
            for j in xrange(0,width,2):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                z1 = level*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
                z2 = level*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
                noiseImg[i,j] = z1
                noiseImg[i,j+1] = z2
         
        """lpf"""
        cv2.imwrite("random_noise.png",noiseImg)
        noiseImg = cv2.GaussianBlur(noiseImg,(3,3),0)
        #pl.psd(noiseImg)
        #pl.show()
        """ Add noise to image"""
        imgIn[:,:,0] = np.clip(np.add(imgIn[:,:,0],noiseImg),0,255)
        imgIn[:,:,1] = np.clip(np.add(imgIn[:,:,1],noiseImg),0,255)
        imgIn[:,:,2] = np.clip(np.add(imgIn[:,:,2],noiseImg),0,255)
        
        
        """check frequency"""
        #zf = fftpack.fft2(noiseImg)   
        #F2 = fftpack.fftshift(zf)
        #psd = np.abs(F2)**2
        #pl.figure(1)
        #pl.clf()
        #pl.imshow(psd)
        #pl.show()
        return imgIn
    
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
    frames = 30
    test = Adas_base(inputImage,width,height,inputType,outputType)
    rgb = test.read2DImageFromSequence()
    salt_noise = copy.deepcopy(rgb)
    awg_noise = copy.deepcopy(rgb)
    awg_noise2 = copy.deepcopy(rgb)
    salt_noise = test.saltAndPepperForRGB(salt_noise,0.01,width,height)
    awg_noise = test.GaussianWhiteNoiseForRGB(awg_noise,width,height)
    awg_noise2 = test.GaussianWhiteNoiseForRGB2(awg_noise2,width,height); 
    cv2.imwrite("rgb_salt.png",salt_noise)
    cv2.imwrite("rgb.png",rgb)
    cv2.imwrite("rgb_awg.png",awg_noise)
    cv2.imwrite("rgb_awg2.png",awg_noise2)
    #cv2.imshow("test",prep.read2DImageFromSequence())   

    #test average frames to reduce noise with gaussian noise model
    result = np.zeros([height,width,3],np.uint32)    
    for i in xrange(frames):
        temp = copy.deepcopy(rgb)
        awg_noise3 = test.GaussianWhiteNoiseForRGB(temp,width,height)
        #cv2.imwrite("noise_"+str(i)+".png",awg_noise3)
        result = np.add(result,awg_noise3)
        print awg_noise3[0,0,0],result[0,0,0]
    result = np.divide(result,frames)
    out = np.zeros([height,width,3],np.uint8)
    out = np.asarray(result[:,:,:],np.uint8)
    print out[0,0,0]
    cv2.imwrite("rgb_result_out.png",out)
    
    del(test)       
    