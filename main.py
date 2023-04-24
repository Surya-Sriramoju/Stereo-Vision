import cv2
import numpy as np
from Utils import *

def main():
    img1 = cv2.imread('dataset/artroom/im0.png')
    img2 = cv2.imread('dataset/artroom/im1.png')

    key1,key2 = get_Keypoints(img1, img2)
    F = ransacF(key1, key2, 0.01)
    plot1(img1, img2, F, key1, key2)
    K = get_K('dataset/artroom/')
    print(K)
    E1, E2 = get_E(F,K)
    E = E2
    print(E)

if __name__ == '__main__':
    main()