# Copyright (c) 2019 by liuxiaobo. All Rights Reserved.
# !/usr/bin/python
# coding=utf-8

'''
the workflow of the task
UR10e + robotiq haneE + realsenseD435
'''

import sys
import os
import time
#get the root path for model inputing
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
sys.path.append(root_path)


from ToolKit.BoundingBox import ComputeArea
from Calibration.Points_4 import calibration
from Driver.UrController import URController
from Driver.handE_controller.gripper_controller import HandEController
from Driver.RealsenseController import RealsenseController
from Config.OperateXml import OperateXml
from ToolKit.rpy2rotationVector import rpy2rotation
from ToolKit.saveData import SaveData
from Segmentation.DetectForeground import Segment


import datetime
import matplotlib.pyplot as plt
import copy
import cv2
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
DEBUG = True

def PickPlan(rect,color_image):
    width = color_image.shape[1]
    heigh = color_image.shape[0]
    ymin = int(rect[0]*heigh)
    xmin = int(rect[1]*width)
    ymax = int(rect[2]*heigh)
    xmax = int(rect[3]*width)

    crop_img = color_image[ymin-10:ymax+10,xmin-10:xmax+10]
    lower = np.array([80, 80, 80])
    upper = np.array([180, 180, 180])

    mask = cv2.inRange(crop_img, lower, upper)
    img_medianBlur=cv2.medianBlur(mask,5)
    mask = cv2.bitwise_not(img_medianBlur)
    if DEBUG:
        cv2.imshow('mask',mask)
        cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    min_Box = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(min_Box)
    box = np.intp(box)

    if DEBUG:
        cv2.drawContours(crop_img, [box], -1, (255,0,0),2)
        cv2.imshow('contours',crop_img)
        cv2.waitKey(1)

    x,y = min_Box[0]
    # x,y = (box[0]+box[1]+box[2]+box[3])/4.0
    x = x + xmin-10
    y = y + ymin-10
    # show the pick points in the image
    if(DEBUG):
        cv2.circle(color_image, (int(x), int(y)), 5, (255,0,0), 4)
        cv2.imshow('pickPoint',color_image)
        cv2.waitKey(1)

    # calibration
    hand_eye = calibration()
    x_p,y_p = hand_eye.cvt(x,y)
    angle = min_Box[2]

    #save the middle result
    cv2.putText(color_image,'angle:%.2f'%(angle*3.14/180.0),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.35, (209, 80, 0, 255), 1)
    recorder.SavePicture('process_%04d'%loop_num + '.jpg',color_image)
    return x_p,y_p,angle*3.14/180.0

def Exeture(pose,rclass):
    # class 0: pick target
    # class 1: place target
    pick_pos = []
    place_pos = []
    for i in range (len(rclass)):
        if (rclass[i]==0):
            pick_pos.append(pose[i])
        elif(rclass[i]==1):
            place_pos.append(pose[i])

    for i in range(len(pick_pos)):
        pick_loop(pick_pos[i],place_pos[i])

def pick_loop(pick_pos,place_pos):
    up_z = 40 #mm
    pick_z = 170
    place_z = 170
    init_rpy = [3.14,0,-0.0*3.14/180.0]
    pick_rpy = init_rpy
    pick_rpy[2] = init_rpy[2]-pick_pos[2]
    rotationVector = rpy2rotation(pick_rpy[0],pick_rpy[1],pick_rpy[2])

    # go above the pick position
    robot_controller.movej(pick_pos[0], pick_pos[1], pick_z+up_z, rotationVector[0], rotationVector[1], rotationVector[2], 0.7, 1.6)
    targtePosition = [pick_pos[0],pick_pos[1],pick_z+up_z,rotationVector[0], rotationVector[1], rotationVector[2]]
    robot_controller.verifyPostion(targtePosition)
    # go down and pick
    robot_controller.movej(pick_pos[0], pick_pos[1], pick_z, rotationVector[0], rotationVector[1], rotationVector[2], 0.5, 0.5)
    targtePosition = [pick_pos[0],pick_pos[1],pick_z,rotationVector[0], rotationVector[1], rotationVector[2]]
    robot_controller.verifyPostion(targtePosition)

    gripper_controller.closeGripper()
    time.sleep(1.5)
    # go above the pick position
    robot_controller.movej(pick_pos[0], pick_pos[1], pick_z+up_z, rotationVector[0], rotationVector[1], rotationVector[2], 0.7, 1.6)
    targtePosition = [pick_pos[0],pick_pos[1],pick_z+up_z,rotationVector[0], rotationVector[1], rotationVector[2]]
    robot_controller.verifyPostion(targtePosition)

    # go above the place position
    place_rpy = [3.14,0,-0.0*3.14/180.0]
    place_rpy[2] = place_rpy[2]-place_pos[2]
    rv_place = rpy2rotation(place_rpy[0],place_rpy[1],place_rpy[2])

    robot_controller.movej(place_pos[0], place_pos[1], place_z+up_z, rv_place[0], rv_place[1], rv_place[2], 0.7, 1.6)
    targtePosition = [place_pos[0],place_pos[1],place_z+up_z,rv_place[0], rv_place[1], rv_place[2]]
    robot_controller.verifyPostion(targtePosition)

    # go down and place
    robot_controller.movej(place_pos[0], place_pos[1], place_z, rv_place[0], rv_place[1], rv_place[2], 0.7, 1.6)
    targtePosition = [place_pos[0],place_pos[1],place_z,rv_place[0], rv_place[1], rv_place[2]]
    robot_controller.verifyPostion(targtePosition)

    gripper_controller.openGripper()
    time.sleep(1.5)

    # go above the place position
    robot_controller.movej(place_pos[0], place_pos[1], place_z+up_z, rv_place[0], rv_place[1], rv_place[2], 0.7, 1.6)
    targtePosition = [place_pos[0],place_pos[1],place_z+up_z,rv_place[0], rv_place[1], rv_place[2]]
    robot_controller.verifyPostion(targtePosition)

    # go home
    robot_controller.movej(home_joints[0], home_joints[1], home_joints[2], home_joints[3], home_joints[4], home_joints[5], 0.8, 1.8,True)
    robot_controller.verifyPostion(home_pose)


def Result():
    str_complete = raw_input("Enter your input: ")
    return str_complete

if __name__ =='__main__':
    #set the robot and camera if needed
    robot_ip = '192.168.31.10'
    robot_port = 30003
    #make sure the home_pose and home_joints are in same pose
    home_pose = [434.02, -16.08, 325.44, 3.14, 0.0, 0.0] # 6D pose, mm/radian
    home_joints = [21.35,-71.45,-142.63,-55.97,89.84,-67.13] #joint, degree

    works_box = [650,150,1000,680] # [xmin,ymin,xmax,ymax] , in image coordinate
    place_box = [350,150,650,680] # [xmin,ymin,xmax,ymax] , in image coordinate
    # # calibration
    # hand_eye = calibration()
    # active camera
    camera_controller = RealsenseController()
    # the main function
    execution_condition = True

    recorder = SaveData()
    loop_num = -1
    finished = False
    angle_time = 0
    while(execution_condition):
        loop_num = loop_num + 1
        place_position = [163.75, 535.28, 175.0, 3.14, 0.0, 0.0]
        # go to home position, please make sure the robot do not occupy the objects when capture images
        robot_controller = URController(robot_ip,robot_port)
        robot_controller.movej(home_joints[0], home_joints[1], home_joints[2], home_joints[3], home_joints[4], home_joints[5], 0.8, 1.8,True)
        robot_controller.verifyPostion(home_pose)
        # active the gripper
        gripper_controller = HandEController(robot_ip,robot_port)
        gripper_controller.openGripper()
        time.sleep(1)

        ''' task begin'''
        task_start=time.time()
        # capture images
        color_image,_,_,_ = camera_controller.getImage()
        # save picture
        recorder.SavePicture('start_%04d'%loop_num + '.jpg',color_image)

        target_index = []
        #segmentation
        seg_start=time.time()
        seg = Segment([380,100,1050,650])
        rect_seg, area_seg = seg.ColorFilter(color_image,np.array([80, 80, 80]),np.array([180, 180, 180]))
        class_reg = len(rect_seg)*[-1]
        seg_end=time.time()
        seg_static =(seg_end-seg_start)

        #Recognition
        rec_start=time.time()
        extend_range = 10
        width = color_image.shape[1]
        heigh = color_image.shape[0]
        for i in range(len(rect_seg)):
            # ymin = int(rect_seg[i][0]*heigh)
            # xmin = int(rect_seg[i][1]*width)
            # ymax = int(rect_seg[i][2]*heigh)
            # xmax = int(rect_seg[i][3]*width)
            # #workspace + placespace
            # bool_ws = ymin >= works_box[1] and xmin >= works_box[0]  and ymax <= works_box[3] and xmax <= works_box[2]
            # bool_ps = ymin >= place_box[1] and xmin >= place_box[0]  and ymax <= place_box[3] and xmax <= place_box[2]
            target_index.append(i)
            area_temp = area_seg[i]
            print(area_temp)
            if(area_temp > 500 and area_temp < 1000):
                class_reg[i] = 0 # cube
            else:
                class_reg[i] = 1 # block area

        rec_end=time.time()
        rec_static = (rec_end-rec_start)

        #pick plan
        pp_start=time.time()
        pick_pose = [0]*len(target_index) # [[u,v,angle],...]
        pick_class = [0]*len(target_index) # [class,...]
        for i in range(len(target_index)):
            index = target_index[i]
            pick_pose[i] = PickPlan(rect_seg[index],color_image)
            pick_class[i] = class_reg[index]

        pp_end=time.time()
        pp_static = (pp_end-pp_start)

        #Execute
        exe_start=time.time()
        Exeture(pick_pose,pick_class)
        exe_end=time.time()
        exe_static = (exe_end-exe_start)
        #result
        ''' task end'''
        task_end=time.time()
        task_static = (task_end-task_start)
        # save picture
        color_image,_,_,_ = camera_controller.getImage()
        cv2.putText(color_image,'Segmentation time: %02.2f'%seg_static, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=2)
        cv2.putText(color_image,'Recognition time: %02.2f'%rec_static, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=2)
        cv2.putText(color_image,'PickPlan time: %02.2f'%pp_static, org=(10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=2)
        cv2.putText(color_image,'Exeture time: %02.2f'%exe_static, org=(10, 110), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=2)
        cv2.putText(color_image,'Task time: %03.2f'%task_static, org=(10, 140), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=2)

        task_result = Result()
        cv2.putText(color_image,'Task score: ' + task_result, org=(10, 170), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 255), thickness=2)
        recorder.SavePicture('resudt_%04d'%loop_num + '.jpg',color_image)
        exit()
