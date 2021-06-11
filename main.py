# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:31:34 2021

@author: Ulis

"""

import cv2
import os
import matplotlib.pyplot as plt

def main():
    directory = "images"
    img1 = cv2.imread(os.path.join(directory,'img1.jpg'))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    sift = cv2.SIFT_create()
    keypoint, description = sift.detectAndCompute(img1,None)
    img_keypoints=cv2.drawKeypoints(img1,keypoint,img1)
    
    plt.figure(figsize=(64 , 64))    
    plt.imshow(img_keypoints); plt.show()
    cv2.imwrite(os.path.join(directory,'img1_keypoints.jpg'),img_keypoints)
    print(description)
    
if __name__ == '__main__':
    main()

