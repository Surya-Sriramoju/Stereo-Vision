import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def getInliers(pt1, pt2, F):
  value = np.dot(pt1.T,np.dot(F,pt2))
  return abs(value)

def normalize(points1):
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