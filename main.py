# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:31:34 2021

@author: 

"""
from os import walk
import cv2
import os
import matplotlib.pyplot as plt

import numpy as np
#Import library k-Means
from sklearn.cluster import KMeans

"""
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
    
if __name__ == '__main__':
    main()
"""    
    
#directory = "images"

fileName=[]
for(dirpath,dirnames,filenames) in walk("images"):
    fileName = filenames
    path = dirpath
    
    
koordinat = [] #Variabel untuk menyimpan koordniant dari masing-masing key point
desc= []
for i in fileName:
    path = str(path + '/')
    img1 = cv2.imread(path + i)
    
    # Convert the image to gray scale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    

    sift = cv2.SIFT_create()
    keypoint, description = sift.detectAndCompute(img1,None)
    
    for d in description:
        desc.append(d) 
        
    #lihat key point (koordinat x dan y) dengan atribut pt
    for k in keypoint:
        koordinat.append(k.pt)
        #print(k.pt)
        
X = np.array(desc)

#Lakukan clustering terhadap X 
kmeans_model = KMeans(n_clusters=150, random_state=0).fit(X)

# Simpan hasil clustering berupa nomor klaster tiap objek/rekord di
# varialbel klaster_objek
klaster_objek = kmeans_model.labels_

# Simpan hasil clustering berupa centroid (titik pusat) tiap kelompok
# di variabel centroids
centroids = kmeans_model.cluster_centers_ 


img_keypoints=cv2.drawKeypoints(img1,keypoint,img1)

plt.figure(figsize=(64 , 64))    
plt.imshow(img_keypoints); plt.show()
cv2.imwrite((path + 'img1_keypoints.jpg'),img_keypoints)