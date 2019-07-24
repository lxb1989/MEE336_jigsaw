#!/usr/bin/env python
import cv2
import numpy as np

import sys
import os
import time
#get the root path for model inputing
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
sys.path.append(root_path)
from Driver.RealsenseController import RealsenseController


DEBUG = False
class Segment(object):
    def __init__(self,workspace = [380,100,1050,650]):
        self.ws = workspace

    def DiffGround(self,groundImg,currrentImg):
        groundImg_gray = cv2.cvtColor(groundImg,cv2.COLOR_BGR2GRAY)
        groundBlur = cv2.GaussianBlur(groundImg_gray,(5,5),1)
        groundBlur.dtype = 'int16'

        currrentImg_gray = cv2.cvtColor(currrentImg,cv2.COLOR_BGR2GRAY)
        currrentImgBlur = cv2.GaussianBlur(currrentImg_gray,(5,5),1)
        currrentImgBlur.dtype = 'int16'

        dGrayBlur = abs(groundBlur-currrentImgBlur)
        dGrayBlur.dtype = 'uint8'
        dGrayMidBlur=cv2.medianBlur(dGrayBlur,5)

        ret,thresh=cv2.threshold(dGrayMidBlur,10,255,cv2.THRESH_BINARY)
        print(thresh.dtype)
        # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rect = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 50*50 or area > 200*200:
                continue
            else:
                temp = cv2.boundingRect(contours[i])

                rect.append(temp)
        # x,y,w,h
        return rect

    def MOG2(self,groundImg,currrentImg,history = 1,kernel = 16):
        #TODO: complete this function for avi and a set of picture
        a = cv2.imread('image_C_0002.jpg',1)
        #histroy: use how many image to build the model
        #kernel: use how many gauss function to build the model
        fgbg = cv2.createBackgroundSubtractorMOG2(history,kernel,False)
        # 1 means add this picture to model
        fgmask = fgbg.apply(a)
        b = cv2.imread('image_C_0003.jpg',1)
        fgmask = fgbg.apply(b,1)

        b = cv2.imread('image_C_0001.jpg',1)
        # 0 means don't add this picture to model
        fgmask = fgbg.apply(b,fgmask,0)


    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    def ColorFilter(self,currrentImg,lower = np.array([10, 80, 0]),upper = np.array([80, 180, 80])):

        crop_img = currrentImg[self.ws[1]:self.ws[3],self.ws[0]:self.ws[2],:] #[ymin:ymax,xmin:xmax]
        mask = cv2.inRange(crop_img, lower, upper)
        img_medianBlur=cv2.medianBlur(mask,5)

        mask = cv2.bitwise_not(img_medianBlur)

        width = currrentImg.shape[1]
        heigh = currrentImg.shape[0]

        if(DEBUG):
            cv2.imshow('mask',mask)
            cv2.waitKey()

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = []

        # # contour: if hierarchy[0][i][3] == -1, it means there are contours inside
        # # len(contours[i] is the num of the contour
        # if (len(contours[i]) < self.min or len(contours[i]) > self.max or cv2.contourArea(contours[i]) < self.min_area): #or hierarchy[0][i][3] == -1
        #     continue
        rect = [] #ymin,xmin,ymax,xmax
        for i in range(len(contours)):
            area_temp = cv2.contourArea(contours[i])
            if area_temp < 20*20 or area_temp > 200*200:
                continue
            else:
                temp = cv2.boundingRect(contours[i])
                box_rate = np.zeros(4)
                box_rate[1] = (temp[0] + self.ws[0])/float(width)
                box_rate[0] = (temp[1] + self.ws[1])/float(heigh)
                box_rate[3] = (temp[2] + temp[0] + self.ws[0]) / float(width)
                box_rate[2] = (temp[3] + temp[1] + self.ws[1]) / float(heigh)
                rect.append(box_rate)
                area.append(area_temp)
        return rect,area

    def ColorFilter_minRect(self,currrentImg,lower = np.array([10, 80, 0]),upper = np.array([80, 180, 80])):
        crop_img = currrentImg[self.ws[1]:self.ws[3],self.ws[0]:self.ws[2],:] #[ymin:ymax,xmin:xmax]
        mask = cv2.inRange(crop_img, lower, upper)
        img_medianBlur=cv2.medianBlur(mask,5)

        mask = cv2.bitwise_not(img_medianBlur)

        width = currrentImg.shape[1]
        heigh = currrentImg.shape[0]

        if(DEBUG):
            cv2.imshow('mask',mask)
            cv2.waitKey()

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        rect = [] #4 points of a rectangle
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 20*20 or area > 200*200:
                continue
            else:
                min_Box = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(min_Box)
                box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                temp = box + [self.ws[0],self.ws[1]]
                # temp = temp/[float(width),float(heigh)]


                # temp = cv2.boundingRect(box)
                # box_rate = np.zeros(4)
                # box_rate[1] = temp[0]/float(width)
                # box_rate[0] = temp[1]/float(heigh)
                # box_rate[3] = (temp[2] + temp[0]) / float(width)
                # box_rate[2] = (temp[3] + temp[1]) / float(heigh)
                rect.append(temp)
        return rect






if __name__ == '__main__':
    camera_controller = RealsenseController()
    time.sleep(2)
    a,_,_,_ = camera_controller.getImage()

    # a = cv2.imread('../Data/BoundingBox1.png')
    # b = cv2.imread('../Data/BoundingBox2.png')
    # print('shape:', a.shape)
    # exit()

    rect,area = Segment().ColorFilter(a,np.array([80, 80, 80]),np.array([180, 180, 180]))
    # rect = Segment().ColorFilter_minRect(a,np.array([80, 80, 80]),np.array([180, 180, 180]))
    print(rect)
    showed_image = a.copy()
    for i in range(len(rect)):
        cv2.rectangle(showed_image,(int(rect[i][1]*1280),int(rect[i][0]*720)),(int(rect[i][3]*1280),int(rect[i][2]*720)),(0,255,0),2)
        # cv2.drawContours(showed_image, [rect[i]], -1, (255,0,0),2)
        # x,y = (rect[i][0]+rect[i][1]+rect[i][2]+rect[i][3])/4.0

        # cv2.circle(showed_image, (int(x), int(y)), 5, (255,0,0), 4)


    cv2.imshow('contours',showed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
