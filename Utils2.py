import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_lines_on_img(points1, points2,F,rect_img1, rect_img2):
    line1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    line1 = line1.reshape(-1,3)

    line2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 2, F)
    line2 = line2.reshape(-1,3)

    img1 = cv2.cvtColor(rect_img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(rect_img2, cv2.COLOR_GRAY2BGR)
    h1,w1,_ = img1.shape
    h2,w2,_ = img2.shape

    for line,point in zip(line1,points1):
        a1,b1,c1 = line
        p1 = (0,-int(c1/b1))
        p2 = (w1, -int((a1*w1 + c1)/b1))
        cv2.line(img1, p1, p2, (0,255,0),2)
        cv2.circle(img1,(int(point[0]), int(point[1])),2,(0,0,255),10)
    
    for line,point in zip(line2,points2):
        a1,b1,c1 = line
        p1 = (0,-int(c1/b1))
        p2 = (w1, -int((a1*w1 + c1)/b1))
        cv2.line(img2, p1, p2, (0,255,0),2)
        cv2.circle(img2,(int(point[0]), int(point[1])),2,(0,0,255),10)
    return img1, img2
    

    

    




