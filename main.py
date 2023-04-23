import cv2
import numpy as np
from keypoints import get_Keypoints
from get_F_E import plot1, ransacF, get_E
from read_params import get_K

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