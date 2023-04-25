import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def getInliers(pt1, pt2, F):
  value = np.dot(pt1.T,np.dot(F,pt2))
  return abs(value)

def normalize(points1):
  '''
  ref: https://www.youtube.com/watch?v=zX5NeY-GTO0
  '''
  mean = np.mean(points1, axis=0)
  std = np.std(points1, axis=0)
  S = np.array([[np.sqrt(2)/std[0], 0 , 0],
                [0, np.sqrt(2)/std[1], 0],
                [0, 0, 1]])
  mean_mat = np.array([[1, 0, -mean[0]],
                      [0, 1, -mean[1]],
                      [0, 0, 1]])
  T = np.matmul(S, mean_mat)
  return points1,T

def computeF(pts1, pts2):
  n = pts1.shape[0]
  A = np.zeros((n, 9))
  for i in range(n):
    A[i][0] = pts1[i][0]*pts2[i][0]
    A[i][1] = pts1[i][0]*pts2[i][1]
    A[i][2] = pts1[i][0]
    A[i][3] = pts1[i][1]*pts2[i][0]
    A[i][4] = pts1[i][1]*pts2[i][1]
    A[i][5] = pts1[i][1]
    A[i][6] = pts2[i][0]
    A[i][7] = pts2[i][1]
    A[i][8] = 1

  U, S, V = np.linalg.svd(A)
  F = V[-1].reshape(3, 3)
  U, S, V = np.linalg.svd(F)
  S[2] = 0
  F = np.dot(U, np.dot(np.diag(S), V))
  return F

def ransac_alg(points1, points2, thresh):
  max_it = 1000
  j = 0
  best_F = np.zeros((3,3))
  num_of_inliers = 0
  while j<max_it:
    pts1 = []
    pts2 = []
    random_points = random.sample(range(0, points1.shape[0]), 8)
    for i in random_points:
      pts1.append(points1[i])
      pts2.append(points2[i])
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    F = computeF(pts1,pts2)
    values = []
    
    for i in range(points1.shape[0]):
      value = getInliers(points1[i], points2[i], F)
      #print(value)
      if value<thresh:
        values.append(value)
        #print('hi')
    if (len(values)) > num_of_inliers:
      num_of_inliers = len(values)
      best_F = F
      #print(len(values))
    j += 1
  return best_F

def transform_points(P,T):
  x = np.dot(T,P.T)
  return x.T

def ransacF(points1, points2, thresh):
  # Find normalization matrix
  P1,T1 = normalize(points1)
  P2,T2 = normalize(points2)
  # Transform point set 1 and 2
  P1_trans = transform_points(P1,T1)
  P2_trans = transform_points(P2,T2)

  # RANSAC based 8-point algorithm
  F = ransac_alg(P1_trans,P2_trans, thresh)
  f_mat = np.dot(np.transpose(T2), np.dot(F, T1))
  return f_mat


def get_E(F, K):
   E1 = np.dot(K.T, np.dot(F,K))
   U,S,V = np.linalg.svd(E1)
   S = [1,1,0]
   E2 = np.dot(U, np.dot(np.diag(S), V))
   return E1,E2
   

def plot1(img1,img2, F_mat, P1, P2):
    random_points = random.sample(range(0, P1.shape[0]), 10)
    w = img2.shape[1]
    num = 1
    for i in random_points:
        a1,b1,c1 = np.matmul(P1[i].transpose(), F_mat)
        
        p1 = (0,-int(c1/b1))
        p2 = (w, -int((a1*w + c1)/b1))
        cv2.line(img2, p1, p2, (0,255,0),2)
        cv2.circle(img2,(int(P2[i][0]), int(P2[i][1])),2,(255,0,0),5)
        num+=1
    num = 1
    for i in random_points:
        a1,b1,c1 = np.matmul(P2[i].transpose(), F_mat)
        p1 = (0,-int(c1/b1))
        p2 = (w, -int((a1*w + c1)/b1))
        cv2.line(img1, p1, p2, (0,255,0),2)
        cv2.circle(img1,(int(P1[i][0]), int(P1[i][1])),2,(255,0,0),5)
        num+=1
    imgs = []
    imgs.append(img1)
    imgs.append(img2)
    _, axs = plt.subplots(1, 2, figsize=(15, 15))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()

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

    # draw_params = dict(matchColor = (0,255,0),
    #                singlePointColor = (255,0,0),
    #                matchesMask = matchesMask,
    #                flags = cv2.DrawMatchesFlags_DEFAULT)
    # img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,matches,None,**draw_params)
    # cv2.imshow('matches', img3)
    # cv2.waitKey(0)
    
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

def get_K(path):
    K = []
    with open(path+'/calib.txt') as f:
        lines = f.readlines()
        k_string = lines[0].split(';')
        for i in k_string:
            if ('cam0' in i) or ('cam1' in i):
                x = i.split(' ')
                x1,x2,x3 = x
                x1 = float(x1[6:])
                x2 = float(x2)
                x3 = float(x3)
                K.append([x1,x2,x3])
            else:
                if "]\n" in i:
                    x = i.split()
                    x1,x2,x3 = x
                    x1 = float(x1)
                    x2 = float(x2)
                    x3 = float(x3[:-1])
                    K.append([x1,x2,x3])
                else:
                    x = i.split(' ')
                    _,x1,x2,x3 = x
                    x1 = float(x1)
                    x2 = float(x2)
                    x3 = float(x3)
                    K.append([x1,x2,x3])
    K = np.array(K).reshape(3,-1)
    return K

def CameraPose(E, key1):
    U,S,V = np.linalg.svd(E)
    W = np.array([[0, -1, 0],[1,0,0],[0,0,1]])

    R1 = U.dot(W.dot(V))
    R2 = U.dot(np.transpose(W).dot(V))
    C1 = U[:,2]
    C2 = -U[:,2]
    Poses = [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]
    p_len = 0
    for pose in Poses:
        points = []
        for i in range(key1.shape[0]):
            point = key1[i]
            V = point-pose[1]
            z = np.dot(pose[0][2], V)
            if z>0:
               points.append(point)
        if len(points)>p_len:
           p_len = len(points)
           best_pose = pose
    return best_pose[0], best_pose[1]

def rectify(img1,img2,key1, key2, F):
    pt1 = np.float32(np.delete(key1,2,1))
    pt2 = np.float32(np.delete(key2,2,1))
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1,w1 = gray_1.shape
    h2,w2 = gray_2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pt1, pt2,F, (w1,h1))
    new_p1 = np.zeros((pt1.shape))
    new_p2 = np.zeros((pt2.shape))
    for i in range(key1.shape[0]):
        temp_new = np.dot(H1, key1[i])
        x = temp_new[0]/temp_new[2]
        y = temp_new[1]/temp_new[2]
        new_p1[i][0] = x
        new_p1[i][1] = y

        temp_new = np.dot(H2, key2[i])
        x = temp_new[0]/temp_new[2]
        y = temp_new[1]/temp_new[2]
        new_p2[i][0] = x
        new_p2[i][1] = y
    rect_img1 = cv2.warpPerspective(gray_1, H1, (w1,h1))
    rect_img2 = cv2.warpPerspective(gray_2, H2, (w2,h2))

    return H1, H2, new_p1, new_p2, rect_img1, rect_img2



   

      



