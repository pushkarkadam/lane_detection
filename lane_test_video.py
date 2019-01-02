#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:13:03 2019

@author: pushkarkadam
"""


import numpy as np 
import cv2


def lane_detection(image):
    def make_coordinate(image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        
        return np.array([x1, y1, x2, y2])
    
    def average_slope_intercept(image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
                
        left_fit_average = np.average(left_fit, axis = 0)
        right_fit_average = np.average(right_fit, axis = 0)
        
        left_line = make_coordinate(image, left_fit_average)
        right_line = make_coordinate(image, right_fit_average)
        
        return np.array([left_line, right_line])
    
    def edge_detection(image):
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_image = cv2.GaussianBlur(grey_image,(5,5),0)
        canny_image = cv2.Canny(blur_image, 50, 150) # 1:3 ratio
        return canny_image
    
    def region_of_interest(image):
        height = image.shape[0]
        polygon = np.array([
                [(200,height),(1100, height),(550,250)]
                ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(image, mask) #Bitwise operation
        return masked_image
    
    def display_lines(image, lines):
        #line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(image, 
                         (x1,y1), 
                         (x2, y2), 
                         color = (0, 255, 0), 
                         thickness = 10)
        return image
    
    def center_line(image, lines):
        x1_1, y1_1, x2_1, y2_1 = lines[0].reshape(4)
        x1_2, y1_2, x2_2, y2_2 = lines[1].reshape(4)
        
        ym1 = y1_1
        ym2 = y2_1
        
        xm1 = int(x1_1 - (x1_1 - x1_2)/2)
        xm2 = int(x2_1 - (x2_1 - x2_2)/2)
        
        cv2.line(image,
                 (xm1,ym1),
                 (xm2,ym2),
                 color = (255,0,0),
                 thickness = 10)
        
        return image
        
    # Image Processing
    lane_image = np.copy(image)
    edge_image = edge_detection(lane_image)
    roi_image = region_of_interest(edge_image) 
    
    # Hough transforms
    lines = cv2.HoughLinesP(roi_image,
                                 rho = 2,
                                 theta = np.pi/180,
                                 threshold = 100,
                                 lines = np.array([]),
                                 minLineLength = 40,
                                 maxLineGap = 4)
    averaged_lines = average_slope_intercept(lane_image, lines)
    center_image = center_line(lane_image, averaged_lines)
    lane_image = display_lines(center_image, averaged_lines)
    return lane_image


cap = cv2.VideoCapture('test2.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('test2_output.mp4', 
                      cv2.VideoWriter_fourcc('P','I','M','1'), 
                      fps = 15, 
                      frameSize = (frame_width,frame_height),
                      isColor = 1)

current_frame = 1
while(current_frame<length):
    _, frame = cap.read()
    try:
        image = lane_detection(frame)
        out.write(image)
        print('Frame: ' + str(current_frame) + '/' + str(length))
    except ValueError:
        print('Value Error')
        out.write(frame)
    except TypeError:
        print('Type Error')
        out.write(frame)
    current_frame += 1
    
cap.release()
out.release()

cv2.destroyAllWindows()
    