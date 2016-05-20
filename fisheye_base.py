# -*- coding: utf-8 -*-
"""
Created on Wed May 4 10:58:24 2016

@author: yipinp
"""
#encoding=utf-8
import cv2
import numpy as np
import os
#from PIL import Image
import random
import copy
import pylab as plt
#from scipy import fftpack
import matplotlib.pyplot as pltt
import math


class fisheye_base :
    def __init__(self):
        pass
    
    
    def __del__(self):
        pass


    def fisheye_radius(self,width,height):
         if width < height :
             r = width/2.0
         else:
             r = height/2.0
         return r


    #calculate the focus length based on the fisheye fov and fisheye image radius
    def focal_calc(self,mode,r,fov):
        if mode == 0 : # pinhole camera model
            f = r/math.tan(fov/2.0)
        elif mode == 1 : #r = fsin(theta)
            f = r/math.sin(fov/2.0)
        elif mode == 2 : # r = f(theta)
            f = r*2.0/fov
        elif mode == 3 : #r = 2f sin(theta/2)
            f = r/(2.0*math.sin(fov/4.0))
        else: #r = 2f tan(theta/2)
            f = r/(2.0*math.tan(fov/4.0))
        return f


    def theta_calc(self,mode,r,f):
        if mode == 0:
            theta = math.atan(r/f)
        elif mode == 1:
            theta = math.asin(r/f)
        elif mode == 2:
            theta = r/f
        elif mode == 3:
            theta = 2.0*math.asin(r/(f*2.0))
        else :
            theta = 2.0*math.atan(r/(f*2.0))
        return theta
        
    def phi_calc(self,dx,dy):
        if dy < 0.0 :
            if dx < 0.0 : #90~180
                phi = 2.0*math.pi - math.atan(dx/dy)
            else:  # 0~90
                phi = math.atan(-dx/dy)
        elif dy == 0.0:
            phi = math.pi
        else:
            if dx < 0.0: #180~270
                phi = math.pi + math.atan(-dx/dy)
            else: #270~360
                phi = math.pi - math.atan(dx/dy) 
        return phi

    def fisheye_gen(self,mode,width,height,fov):
        fisheye_img = np.zeros([height,width,3],np.uint8)
        max_r = self.fisheye_radius(width,height)
        fov = fov*math.pi/180.0
        f = self.focal_calc(mode,max_r,fov)  
        grid_spacing = 0.2
        line_width = 0.1
        edge_width = 0.03
        #macbeth color checker
        mb_r = np.zeros([4,6],np.uint8)
        mb_g = np.zeros([4,6],np.uint8)
        mb_b = np.zeros([4,6],np.uint8)
        mb_r = [ [115,194,98,87,133,103], [214,80,193, 94, 157, 224],[56,  70, 175, 231, 187,   8],[243, 200, 160, 122,  85,  52]]
        mb_g = [ [82, 150, 122, 108, 128, 189],[126,  91,  90,  60, 188, 163],[61, 148,  54, 199,  86, 133],[243, 200, 160, 122,  85,  52]    ]
        mb_b = [[68, 130, 157,  67, 177, 170],[44, 166,  99, 108,  64,  46],[150,  73,  60,  31, 149, 161],[242, 200, 160, 121,  85,  52] ]
        #set the image center as (0,0)
        top_left_x = -width/2.0
        top_left_y = -height/2.0
        
        #back projection for fisheye generation
        for y in xrange(height):
            for x in xrange(width):
                dy = y + top_left_y
                dx = x + top_left_x
                r = math.sqrt(dx*dx + dy*dy)
                #in fisheye circle 
                if (r < max_r):
                    theta = self.theta_calc(mode,r,f)
                    #calculate phi angle
                    phi = self.phi_calc(dx,dy)
                    #infinit plane assumption
                    px = math.sin(phi)*math.tan(theta)
                    py = math.cos(phi)*math.tan(theta)
                    z = 1.0
                    sx = np.mod(np.abs(px),grid_spacing)/grid_spacing
                    sy = np.mod(np.abs(py),grid_spacing)/grid_spacing
                    ix = int(3.0 + px/grid_spacing)
                    iy = int(2.0 - py/grid_spacing)
                    if ix < 0.0 :
                        ix = -1
                    if iy < 0.0:
                        iy = -1
                    #fill pattern
                    if ix >= 0 and ix < 6 and iy >=0 and iy < 4:
                        red = mb_r[iy][ix]
                        grn = mb_g[iy][ix]
                        blu = mb_b[iy][ix]
                    else:
                        red = 255.0
                        grn = 255.0
                        blu = 255.0
                    
                    #divide all plane as uniform grid network
                    # line->edge
                    if sx < line_width or sy < line_width:
                        p = 0.0
                    elif sx < (line_width + edge_width) or sy <(line_width+edge_width):
                        sx -= line_width
                        sx /= edge_width
                        sy -= line_width
                        sy /= edge_width
                        if sx < sy:
                            sy = np.mod(np.abs(math.cos(phi)*math.tan(theta)),grid_spacing)/grid_spacing
                            if sy > 1.0 - edge_width and sy -(1.0-edge_width)/edge_width > (1.0 - sx):
                                sy -= 1.0 - edge_width
                                sy /= edge_width
                                p = 0.5 + 0.5*math.cos(math.pi*sy)
                            else:
                                p = 0.5 - 0.5 *math.cos(math.pi*sy)
                        else:
                            sx = np.mod(np.abs(math.sin(phi)*math.tan(theta)),grid_spacing)/grid_spacing 
                            if sx > 1.0 - edge_width and sx -(1.0-edge_width)/edge_width > (1.0 - sy):
                                sx -= 1.0 - edge_width
                                sx /= edge_width
                                p = 0.5 + 0.5*math.cos(math.pi*sy)
                            else:
                                p = 0.5 - 0.5 *math.cos(math.pi*sy)
                    else:
                        if sx > 1.0-edge_width or sy > 1.0-edge_width:
                            sx -= (1.0 - edge_width);
                            sx /=edge_width
                            sy -= (1.0 - edge_width);
                            sy /=edge_width
                            if sx > sy:
                               p = 0.5 + 0.5*math.cos(math.pi*sx) 
                            else:
                               p = 0.5 + 0.5*math.cos(math.pi*sy)
                        else:
                              p = 1.0
                            
                    #draw circles for fov checker
                    theta = np.mod(theta,math.pi/18.0)/(math.pi/18.0)
                    #print theta
    
                    if theta < edge_width:
                        phi = 0.5 + 0.5*math.cos(math.pi*theta/edge_width)
                        red = (1.0-phi)*red*p + phi*255.0
                        grn = (1.0-phi)*grn*p + phi*16.0
                        blu = (1.0-phi)*blu*p + phi*16.0
                        p = 1.0
                    elif theta > 1.0 - edge_width:
                        phi = 0.5 + 0.5*math.cos(math.pi*(1.0 -theta)/edge_width)
                        red = (1.0-phi)*red*p + phi*255.0
                        grn = (1.0-phi)*grn*p + phi*16.0
                        blu = (1.0-phi)*blu*p + phi*16.0
                        p = 1.0                
                else:
                    red = 0.0
                    grn = 0.0
                    blu = 0.0
                    p = 0.0
        
                fisheye_img[y,x,0] = int(p*red)
                fisheye_img[y,x,1] = int(p*grn)
                fisheye_img[y,x,2] = int(p*blu)

        #write output frame
        fisheye_img = cv2.cvtColor(fisheye_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite('frame.png',fisheye_img)
        
                

if __name__ == "__main__":
    
    test =  fisheye_base()
    test.fisheye_gen(1,640,480,120)
    del(test)