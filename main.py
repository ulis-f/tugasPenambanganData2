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
import pickle
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def getKeypoints():
    
    fileName=[]
    for(dirpath,dirnames,filenames) in walk("images_to_guess"):
        fileName = filenames
        path = dirpath
        
    sift = cv2.SIFT_create()
    desc=[]
    kp =[]
    img=[]
    for i in fileName:
        img1 = cv2.imread(os.path.join(path,i))
        
        # Convert the image to gray scale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
    
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        
        keypoint, description = sift.detectAndCompute(img1,None)
        
        desc.append(description)
        kp.append(keypoint)
        img.append(img1)
    return kp,desc, fileName,img
    
    
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

###############################################################################

# wcss=[]
# for i in range(140,151): 
#     kmeans_model = KMeans(n_clusters=i).fit(X)
    
#     #Cari nilai WCSS 
#     wcss_iter = kmeans_model.inertia_
#     wcss.append(wcss_iter)

# number_clusters = range(140,151)
# plt.plot(number_clusters,wcss)
# plt.title('Method Elbow')
# plt.xlabel('Nilai k')
# plt.ylabel('WCSS') 

# ###############################################################################
# score_silhouette = []
# for i in range(150,201):
#     kmeans_model = KMeans(n_clusters=i).fit(X)
    
#     #Simpan hasil clustering berupa nomor klaster tiap objek/rekord di varialbel labels
#     labels = kmeans_model.labels_
    
#     #Hitung score sillhoutte 
#     silhouette_avg = silhouette_score(X,labels)
#     score_silhouette.append(silhouette_avg)

#k terbaik adalah 178            
###############################################################################
#Lakukan clustering terhadap X 

kmeans_model = None

if os.path.exists('./model178.pkl'):
    with open("model178.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    
else:
    kmeans_model = KMeans(n_clusters=178, random_state=0).fit(X)
    with open("model178.pkl", "wb") as f:
        pickle.dump(kmeans_model, f)

# Simpan hasil clustering berupa nomor klaster tiap objek/rekord di
# varialbel labels_single
labels_single = kmeans_model.labels_

# Simpan hasil clustering berupa centroid (titik pusat) tiap kelompok
# di variabel centroids
centroids = kmeans_model.cluster_centers_ 
print(centroids)

###############################################################################

#Data frame untuk menyimpan hasil cluster dan 
#nilai jarak masing2 description ke centroidnya
df_result = pd.DataFrame([])

for ca, x in zip(labels_single, X):
    # Menghitung Euclidean distance menggunakan linalg.norm()
    dist = np.linalg.norm(x - centroids[ca])
    row = pd.Series([ca, x, dist,centroids[ca]])      
    row_df = pd.DataFrame([row])  
    #Insert baris baru ke data frame
    df_result = pd.concat([row_df, df_result], ignore_index=True)
    
#Rename kolom
df_result.rename(columns = {0:'Cluster',1:'Description', 2:'Jarak ke Centroid',3:'Centroid'}, inplace = True)

###############################################################################

#Data frame untuk menyimpan rata-rata jarak setiap cluster ke centroid
df_result2 = pd.DataFrame([])

for m in range (max(labels_single)+1):
    df = df_result[df_result['Cluster'] == m]
    sumJarak = sum(df['Jarak ke Centroid'])  
    rataRata = sumJarak / len(df)
    row2 = pd.Series([m, rataRata,df['Centroid'].iloc[0]])      
    row_df2 = pd.DataFrame([row2])  
    #Insert baris baru ke data frame
    df_result2 = pd.concat([row_df2, df_result2], ignore_index=True)  
 
#Rename kolom
df_result2.rename(columns = {0:'Cluster',1:'Rata-rata',2:'Centroid'}, inplace = True)
print(df_result2)

###############################################################################
#Histogram Rata-rata jarak setiap cluster ke centroid
count, bin_edges = np.histogram(df_result2['Rata-rata'])
df_result2['Rata-rata'].plot(kind = 'hist', xticks = bin_edges)
plt.title('Histogram Rata-rata Jarak Setiap Cluster ke Centroid')
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('Frekuensi')  
plt.xlabel('Rata-rata')                      
plt.show()
    
#Buat boxplot
plt.boxplot(df_result2['Rata-rata'])
plt.show()

###############################################################################

new_df = df_result2
print(new_df)
count, bin_edges = np.histogram(new_df['Rata-rata'])
new_df['Rata-rata'].plot(kind = 'hist', xticks = bin_edges)
plt.title('Histogram Rata-rata Jarak Setiap Cluster ke Centroid')
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('Frekuensi')  
plt.xlabel('Rata-rata')                      
plt.show()

kp,desc,fileName,img = getKeypoints()
idx=1
for key,description,myFile,myImage in zip(kp,desc,fileName,img):
    keyWords=[]
    kNonWords=[]
    for k,d in zip(key,description):
        minim = sys.maxsize
        avg = 288.5
        for index, row in new_df.iterrows():
            
            result = np.linalg.norm(row["Centroid"]-d)
            if minim > result and result <= avg:
                minim = result                
        
        if minim < avg:
            keyWords.append(k)
        else:
            kNonWords.append(k)
    img_keypoints = cv2.drawKeypoints(myImage,keyWords,0,color= (0,128,0))
    img_keypoints = cv2.drawKeypoints(img_keypoints,kNonWords,0,color=(255,0,0))
    fileOut = 'img'+str(idx)+'.jpg'
    idx+=1
    cv2.imwrite(fileOut,img_keypoints)
    


