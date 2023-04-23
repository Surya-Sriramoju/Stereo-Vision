import cv2
import numpy as np

def three_point_form(points):
  points1 = np.zeros((points.shape[0],3))
  
  for i in range(points.shape[0]):
    points1[i][0] = points[i][0]
    points1[i][1] = points[i][1]
    points1[i][2] = 1
  return points1


def get_Keypoints(img1, img2):
    img1_Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_Gray,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_Gray,None)

    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    good_matches = []
    i = 0
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            good_matches.append(m)
            i += 1

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,matches,None,**draw_params)
    
    source_points = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    key_pts1 = np.zeros((source_points.shape[0],2))
    key_pts2 = np.zeros((destination_points.shape[0],2))

    for i in range(source_points.shape[0]):
      key_pts1[i] = source_points[i]
      key_pts2[i] = destination_points[i]
    
    key_pts1 = three_point_form(key_pts1)
    key_pts2 = three_point_form(key_pts2)

    return key_pts1, key_pts2