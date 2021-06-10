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
import pandas as pd
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

################################################################################

#Lakukan clustering terhadap X 
kmeans_model = KMeans(n_clusters=150, random_state=0).fit(X)

# Simpan hasil clustering berupa nomor klaster tiap objek/rekord di
# varialbel labels_single
labels_single = kmeans_model.labels_

# Simpan hasil clustering berupa centroid (titik pusat) tiap kelompok
# di variabel centroids
centroids = kmeans_model.cluster_centers_ 

###############################################################################

#Data frame untuk menyimpan hasil cluster dan 
#nilai jarak masing2 description ke centroidnya
df_result = pd.DataFrame([])

for ca, x in zip(labels_single, X):
    # Menghitung Euclidean distance menggunakan linalg.norm()
    dist = np.linalg.norm(x - centroids[ca])
    row = pd.Series([ca, x, dist])      
    row_df = pd.DataFrame([row])  
    #Insert baris baru ke data frame
    df_result = pd.concat([row_df, df_result], ignore_index=True)
    
#Rename kolom
df_result.rename(columns = {0:'Cluster',1:'Description', 2:'Jarak ke Centroid'}, inplace = True)

###############################################################################

#Data frame untuk menyimpan rata-rata jarak setiap cluster ke centroid
df_result2 = pd.DataFrame([])

for m in range (max(labels_single)+1):
    df = df_result[df_result['Cluster'] == m]
    sumJarak = sum(df['Jarak ke Centroid'])  
    rataRata = sumJarak / len(df)
    row2 = pd.Series([m, rataRata])      
    row_df2 = pd.DataFrame([row2])  
    #Insert baris baru ke data frame
    df_result2 = pd.concat([row_df2, df_result2], ignore_index=True)  
 
#Rename kolom
df_result2.rename(columns = {0:'Cluster',1:'Rata-rata'}, inplace = True)

###############################################################################
#Histogram Rata-rata jarak setiap cluster ke centroid
count, bin_edges = np.histogram(df_result2['Rata-rata'])
df_result2['Rata-rata'].plot(kind = 'hist', xticks = bin_edges)
plt.title('Histogram Rata-rata Jarak Setiap Cluster ke Centroid')
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('Frekuensi')  
plt.xlabel('Rata-rata')                      
plt.show()

###############################################################################

img_keypoints=cv2.drawKeypoints(img1,keypoint,img1)

plt.figure(figsize=(64 , 64))    
plt.imshow(img_keypoints); plt.show()
cv2.imwrite((path + 'img1_keypoints.jpg'),img_keypoints)