import cv2
import numpy as np
from Utils import *

def main():
    while True:
      print('Enter 1,2,3 based on the options below:\n')
      print("1.artroom\n")
      print("2.chess\n")
      print("3.ladder\n")
      option = int(input())
      if option == 1:
        path = 'dataset/artroom/'
        img1, img2, K, baseline, f = get_params(path)
        img1 = cv2.resize(img1, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        break
      elif option == 2:
        path = 'dataset/chess/'
        img1, img2, K, baseline, f = get_params(path)
        img1 = cv2.resize(img1, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        break
      elif option == 3:
        path = 'dataset/ladder/'
        img1, img2, K, baseline, f = get_params(path)
        img1 = cv2.resize(img1, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        break
      else:
        print("incorrect option, try again\n")
        continue
    while True:
        try:
            key1,key2 = get_Keypoints(img1, img2)
            F = ransacF(key1, key2, 0.02)
            E = get_E(F,K)
            R,t = CameraPose(E, key1)
            
            H1, H2, new_p1, new_p2, rect_img1, rect_img2 = rectify(img1,img2,key1, key2, F)
            line_img1, line_img2 = get_lines_on_img(new_p1, new_p2,F,rect_img1, rect_img2)
            break
        except Exception:
           continue
    cv2.imshow('Epipolar lines', np.hstack([line_img1, line_img2]))
    cv2.waitKey(0)
    print("Fundamental Matrix: ")
    print(F)
    print("Essential Matrix: ")
    print(E)
    print('Camera Pose: ')
    camera_pose = np.hstack([R,t.reshape(3,-1)])
    print(camera_pose)

    print("Homography matrix of image 1:")
    print(H1)

    print("Homography matrix of image 2:")
    print(H2)

    print('Calculating Disparity Map')
    disparity_map_unscaled, disparity_map_scaled = correspondence(rect_img1, rect_img2)
    print('Calculating Depth Map')
    depth_map, depth_array = depth_calc(baseline, f, disparity_map_unscaled)
    plt.figure(1)
    plt.title('Disparity Map Grayscale')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.figure(2)
    plt.title('Disparity Map Hot')
    plt.imshow(disparity_map_scaled, cmap='hot')
    plt.figure(3)
    plt.title('Depth Map Grayscale')
    plt.imshow(depth_map, cmap='gray')
    plt.figure(4)
    plt.title('Depth Map Hot')
    plt.imshow(depth_map, cmap='hot')
    plt.show()

if __name__ == '__main__':
    main()