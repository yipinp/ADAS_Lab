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
import pylab as plt
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
    
    
    #ffmpeg caller
    def  avi2YUV(self,videoName,number):
        #FFMPEG setting
        ffmpeg_exe =  os.getcwd() + r"\ffmpeg\ffmpeg.exe"
        cmdOptions  = ['%s '%(ffmpeg_exe)];      
        cmdOptions += ['-i %s -vframes %s -y -an %s.yuv'%(videoName+".avi", number, videoName)]; 
        cmd = "".join(cmdOptions)
        os.system(cmd)    
    
    
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
        level = 30
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
         elif mode == 2 :
             imageOut = cv2.bilateralFilter(imageIn,5,30,30)
         else:
             kernel = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],np.float32)/256
             imageOut = cv2.filter2D(imageIn,-1,kernel)
        
         return imageOut
    
    def SAD_block(self,array0,array1,x_center,y_center,isPictureBoundary,component,offsetx = 0,offsety = 0,size=3):
         if isPictureBoundary:
             sad = np.abs(np.substract(array0[y_center,x_center,component] - array1[y_center+offsety,x_center+offsetx,component],dtype=np.int32))
         else:
             sad = np.sum(
                     np.abs(
                         np.subtract(array0[y_center-size/2:y_center+size/2+1 ,x_center-size/2:x_center+size/2+1,component] 
                             ,array1[y_center-size/2 + offsety:y_center+size/2+1+offsety ,x_center-size/2+offsetx:x_center+size/2+1+offsetx,component],dtype=np.int32),dtype=np.int32))  
         return sad
         
    #Euclidean distancec     
    def Euc_distance_block(self,array0,array1,x_center,y_center,isPictureBoundary,component,offsetx = 0,offsety = 0,size=3):
        distance = np.sqrt(
                    np.sum(
                     np.power(
                         np.subtract(array0[y_center-size/2:y_center+size/2+1 ,x_center-size/2:x_center+size/2+1,component] 
                             ,array1[y_center-size/2 + offsety:y_center+size/2+1+offsety ,x_center-size/2+offsetx:x_center+size/2+1+offsetx,component],dtype=np.int32),2,dtype=np.int32)))  
        return distance
                             
    def setAlpha(self,sad,alpha,i,j,isBoundary,channel,light=1,ST=0,iir = 768):
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
        #alpha[i,j,channel] = 512
        
        
 
    def getAlphaFromSAD(self,imageIn,imagePrev,height,width,channel,alpha):  
        sad = 0
        for i in range(height):
            for j in range(width):
                isBoundary = (j == 0) or (i == 0) or (j == width -1) or (i == height -1)
                sad = self.SAD_block(imageIn,imagePrev,j,i,isBoundary,channel) 
                self.setAlpha(sad,alpha,i,j,isBoundary,channel)

       
    def alphaBlending(self,imageIn,imagePrev,alpha,j,i,channel,imageOut):
         imageOut[i,j,channel] =  (alpha[i,j,channel]*imagePrev[i,j,channel]+(1024-alpha[i,j,channel])*imageIn[i,j,channel])>>10
        
        
    def betaBlending(self,imageSpatial,temporalFilter,j,i,channel,alpha,image3DOut):
        beta = 1024 - alpha[i,j,channel]
        beta = min(beta,400)
        image3DOut[i,j,channel] = (beta*imageSpatial[i,j,channel] + (1024-beta)*imageSpatial[i,j,channel]) >> 10
            
    """ 
       R/G/B or Y/U/V seperate channel processing, motion detection not estimation, alpha blending and beta blending
    """
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
         
         
         
    def generateOpticalFlowView(self,p0,p1,imgIn,frame): 
        imgT  = copy.deepcopy(imgIn)
        good_new = p1
        good_old = p0
        mask =  np.zeros_like(imgIn)
        color = np.random.randint(0,255,(100,3))
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            print a,b,c,d

            cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i%100].tolist(), 2)
            cv2.circle(imgT,(a,b),5,color[i%100].tolist(),-1)

        img = cv2.add(imgT,mask)
        cv2.imwrite('opticalFlow'+str(frame)+'.png',img)  
        
        
        
    def callPixelOpticalFlowAnalysis(self,y,x,imgInSpatial,imgPrevSpatial,weight_sad,old_ps,new_ps,maxSAD):
        lk_params = dict(winSize = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,10,0.03)) 
        p0 = [[y,x]]
        p0 = np.float32(np.asarray(p0))
        p1,st,err = cv2.calcOpticalFlowPyrLK(imgPrevSpatial,imgInSpatial,p0,None,**lk_params)
        u,v = p1[0]
        sad = 0
                     
        #if detect the optical flow for the pixel, check if the motion vector is reasonable with SAD
        #if motion vector is out of the boundary, discard it
        if x + u - 1 > 0 and y+v+1<height and x + u + 1 <width and y + v - 1> 0 : 
            for c in xrange(3):
                sad +=(self.SAD_block(imgInSpatial,imgPrevSpatial,x,y,0,c,int(u),int(v))*weight_sad[c])
                print("Start optical flow analysis...")
                old_ps.append(p0)
                new_ps.append(p1)
        else:
            sad = maxSAD  #alpha is 0 for the pixel, not use prev buffer as blending
            
        return sad
        
        
    def betaCalculate(self,imgIn,beta_map,height,width,weight_sad):
         meanArray = np.zeros(3,np.uint32)
         stddevArray = np.zeros(3,np.uint32)
         for j in xrange(1,height-1):
             for i in xrange(1,width-1):
                 for c in xrange(3):
                     meanArray[c] = np.mean(imgIn[j-1:j+2,i-1:i+2,c],dtype=np.uint32)
                     stddevArray[c] = (np.sum(np.power(imgIn[j-1:j+2,i-1:i+2,c],2,dtype=np.uint32),dtype=np.uint32))/9.0
                                                         
                 mean = meanArray[0]*weight_sad[0] + meanArray[1]*weight_sad[1] + meanArray[2]*weight_sad[2]
                 stddev = stddevArray[0]*weight_sad[0] + stddevArray[1]*weight_sad[1] + stddevArray[2]*weight_sad[2]
                 stddev = np.sqrt(stddev - mean**2) 
                 if  mean < 0.01:
                     mean = 0.01
                 activity = stddev/mean
                 #print mean,stddev,activity
                 #call sigmod function for 0-1 normalization
                 beta = 1.0/np.exp(1.0/activity)
                 beta_map[j,i] = beta 
       
    """
        R/G/B,Y/U/V procecss together, not seperate channel for parameters, with simplied optical flow and motion detection
        alpha and beta blending
    """           
    def TNR3D(self,imgIn,imgOut,imgPrevIn,height,width,frame,alphaThres=600):
         #Spatial filter to imgPrevIn and imgIn
         imgPrevSpatial = self.spatialFilterFrame(imgPrevIn,2)
         imgInSpatial = self.spatialFilterFrame(imgIn,2)
         alpha=np.zeros([height,width,1],np.uint32)
         sads  = np.zeros([height,width,1],np.uint32)
         #R,G,B with equal weight
         weight_sad = (1/3.0,1/3.0,1/3.0)
         maxSAD = 1024.0
         minSAD = 0.0
         maxAlpha = 1024.0
         minAlpha = 0.0
         maxBeta = 1024.0
         minBeta = 0.0
         old_ps  =[]
         new_ps = []
         
         
         #calculate SAD and detect the static and moving pixels,all channels together
         #if SAD is too much, it means absolute moving region, call sparse LK for optical flow vector.
         #Please notice, if the alphaThreshold is too low, too much optical flow will be called and < 1 pixel mv will be get.
            
         for j in xrange(1,height-1):
             for i in xrange(1,width-1):
                 sad = 0
                 for c in xrange(3):
                     sad += (self.SAD_block(imgInSpatial,imgPrevSpatial,i,j,0,c)*weight_sad[c])
                       
                 #if sad is too high, it is moving region, try to call optical flow for the pixel                 
                 if sad > alphaThres: 
                     sad = test.callPixelOpticalFlowAnalysis(j,i,imgInSpatial,imgPrevSpatial,weight_sad,old_ps,new_ps,maxSAD)                                             
                 #set the alpha value for the pixel based on the sad
                 sads[j,i] = sad
                 self.setAlpha(sad,alpha,j,i,0,0)
            
         #plot sad vs alpha curve
         if len(old_ps) > 0 :
             test.generateOpticalFlowView(old_ps,new_ps,imgInSpatial,frame)
         plt.subplot(211)
         plt.plot(sads.flatten())
         plt.subplot(212)
         plt.plot(alpha.flatten())
                 
                 
                 
                 
         #calculate beta based on activity for input and spatial input blending
         #when the activity is high, it means the pixel variation in the block is high, may be edge
         # 3x3 fixed block
         beta_map = np.zeros([height,width],np.float)
         test.betaCalculate(imgIn,beta_map,height,width,weight_sad)
                    
                                                          
         """beta & alpha blending"""
         for j in xrange(height):
             for i in xrange(width):
                 #print j,i,beta_map[j,i],alpha[j,i,0]
                 for c in xrange(3):                     
                     imgOut[j,i,c] = (beta_map[j,i]*np.subtract(imgIn[j,i,c],imgInSpatial[j,i,c],dtype=np.int32)) + imgInSpatial[j,i,c]  
                     imgOut[j,i,c] = (alpha[j,i,0] *np.subtract(imgPrevIn[j,i,c],imgOut[j,i,c],dtype=np.int32)>>10) + imgOut[j,i,c]
        
         return imgOut         
                 
    """NLM for 3d"""
    def TNR3D_2(self,imgIn,imgOut,imgPrevIn,height,width,frame,sad_thres=600):
        imgPrevSpatial = self.spatialFilterFrame(imgPrevIn,2)
        imgInSpatial = self.spatialFilterFrame(imgIn,2)
        candiateNum = 18
        sad_candidate = np.zeros(candiateNum,np.uint32)
        distance_candidate = np.zeros(candiateNum,np.uint32)
        distance_weight = np.zeros(candiateNum,np.float)
        total_distance = np.zeros([height,width],np.float)
        offset = [(0,0),(0,-1),(0,1),(-1,0),(-1,-1),(-1,1),(1,0),(1,-1),(1,1)]
        #R,G,B with equal weight
        weight_sad = (1/3.0,1/3.0,1/3.0)
        #Gaussian variance
        variance = 1.0 

        #check 3x3 for current frame and prev 3x3 , 18 blocks for SAD
        for j in xrange(2,height-2):
             for i in xrange(2,width-2):
                 #idx 0-8  for prev picture, 9-17 for intra picture
                 for idx in xrange(18):
                     offsety,offsetx = offset[idx%9]
                     if idx > 8 :
                         imgIn2 = imgPrevSpatial                        
                     else:
                         imgIn2 = imgInSpatial
                     for c in xrange(3):
                         sad_candidate[idx] += (self.SAD_block(imgInSpatial,imgIn2,i,j,0,c,offsety,offsetx)*weight_sad[c])
                         distance_candidate[idx] += (self.Euc_distance_block(imgInSpatial,imgIn2,i,j,0,c,offsety,offsetx)*weight_sad[c])
                     total_distance[j,i] += distance_candidate[idx]
                     print total_distance[j,i]
                
                 #check sad threshold to select the reasonable candidate position and do NLM blending
                 for idx in xrange(18):
                     if sad_candidate[idx] > sad_thres:
                         distance_weight[idx] = 0.0
                     else:
                         #Gaussian weight based on euclidean distance                
                         distance_weight[idx] = np.exp(-1.0*distance_candidate[idx]/(total_distance[j,i]*(variance**2)))
                         #print idx,distance_weight[idx],total_distance[j,i],distance_candidate[idx],total_distance[j,i]*(variance**2),distance_candidate[idx]/(total_distance*(variance**2))
                    
                         
                 #normalization weight
                 for idx in xrange(18):
                     distance_weight[idx] /= total_distance[j,i]
                
                 for c in xrange(3):
                     pixel_v = 0.0
                     for idx in xrange(18):
                         if idx > 8 :
                             imgIn2 = imgPrevIn
                         else:
                             imgIn2 = imgIn                                    
                         pixel_v += distance_weight[idx]*imgIn2[j,i,c]
                     

                     imgOut[j,i,c] =int(pixel_v)  
        return imgOut
       
if __name__ == "__main__":
    
        
    #YUV420->RGB
    filename = r"\mobile_qcif.yuv"
    inputImage = os.getcwd() + filename
    width = 176
    height = 144
    inputType = "YUV420"
    outputType = "RGB"
    frames = 5
    test = Adas_base(inputImage,width,height,inputType,outputType)
    """
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
    """
    """
    #test average frames to reduce noise with gaussian noise model
    result = np.zeros([height,width,3],np.uint32)    
    for i in xrange(frames):
        temp = copy.deepcopy(rgb)
        awg_noise3 = test.GaussianWhiteNoiseForRGB(temp,width,height)
        #cv2.imwrite("noise_"+str(i)+".png",awg_noise3)
        result = np.add(result,awg_noise3)
    
    result = np.divide(result,frames)
    out = np.zeros([height,width,3],np.uint8)
    out = np.asarray(result[:,:,:],np.uint8)
    print out[0,0,0]
    cv2.imwrite("rgb_result_out.png",out)
    
    #test different image blending for ghost
    rgb1 = test.read2DImageFromSequence()
   # cv2.imwrite("rgb1.png",rgb1)
   # result = np.divide(np.substract(rgb1,rgb),2)
   # cv2.imwrite("ghost.png",result[:,:,0])
   # print rgb[0,0,0],rgb1[0,0,0],result[0,0,0]
   """  
   
    """
    #sparse LK 
    rgb1 = test.read2DImageFromSequence()
    oldGray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    newGray = cv2.cvtColor(rgb1,cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners = 100,qualityLevel=0.3,minDistance = 7, blockSize = 7)
    lk_params = dict(winSize = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,10,0.03))
    p0 = cv2.goodFeaturesToTrack(oldGray,mask = None, **feature_params)

    
    p1,st,err = cv2.calcOpticalFlowPyrLK(oldGray,newGray,p0,None,**lk_params)
    print len(p0),len(p1)

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    mask =  np.zeros_like(rgb1)
    color = np.random.randint(0,255,(100,3))
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)

        cv2.circle(rgb1,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(rgb1,mask)
    cv2.imwrite('frame.png',img)   
    
    #gradients
    oldGray = cv2.cvtColor(awg_noise,cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(oldGray,cv2.CV_64F)
    sobelx = cv2.Sobel(oldGray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(oldGray,cv2.CV_64F,0,1,ksize=5)
    cv2.imwrite("laplacian.png",laplacian)
    cv2.imwrite("sobelx.png",sobelx)
    cv2.imwrite("sobely.png",sobely)
    #plt.subplot(2,2,1),plt.imshow(oldGray,cmap = 'gray')
    #plt.title('Original'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    #plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    #plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    #plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    """

    imgOut = np.zeros([height,width,3],np.uint8)
    imgPrev = np.zeros([height,width,3],np.uint8)
    tnrInputName = "TnrIn"
    tnrOutName  = "TnrOut"

    fourcc = cv2.cv.FOURCC('M','J','P','G')
    videoW = cv2.VideoWriter(tnrInputName + ".avi",fourcc,1.0,(width,height),True)
    videoWR = cv2.VideoWriter(tnrOutName +".avi",fourcc,1.0,(width,height),True)
    for i in xrange(frames):
        print i
        rgbIn = test.read2DImageFromSequence()
        noisy_rgb = copy.deepcopy(rgbIn)    
        noisy_rgb = test.GaussianWhiteNoiseForRGB(rgbIn,width,height)
        #cv2.imwrite("frame"+str(i)+".png",noisy_rgb)
        videoW.write(noisy_rgb)
        if i == 0 :
            imgOut = test.spatialFilterFrame(noisy_rgb,2)
        else:
            imgOut = test.TNR3D_2(noisy_rgb,imgOut,imgPrev,height,width,i) 
        
        imgPrev = copy.deepcopy(imgOut)
        videoWR.write(imgOut)
        #cv2.imwrite("tnr_"+str(i)+".png",imgOut)
    videoW.release()
    videoWR.release()
    
    #convert to YUV with FFMPEG to compare them at the same time(yuvplayer)
    test.avi2YUV(tnrInputName,frames)
    test.avi2YUV(tnrOutName,frames)
    
    del(test)       
    