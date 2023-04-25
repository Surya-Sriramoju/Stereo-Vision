import cv2
import numpy as np
from Utils import *

def main():
    img1 = cv2.imread('dataset/artroom/im0.png')
    img2 = cv2.imread('dataset/artroom/im1.png')

    key1,key2 = get_Keypoints(img1, img2)
    F = ransacF(key1, key2, 0.01)
    # plot1(img1, img2, F, key1, key2)
    K = get_K('dataset/artroom/')
    E1, E2 = get_E(F,K)
    E = E2
    R,t = CameraPose(E, key1)
    H1, H2, new_p1, new_p2, rect_img1, rect_img2 = rectify(img1,img2,key1, key2, F)
    plot1(rect_img1, rect_img2, F, three_point_form(new_p1), three_point_form(new_p2))
    # cv2.imshow('side by side', cv2.resize(np.hstack([rect_img1, rect_img2]), None, fx = 0.3, fy = 0.3))
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()