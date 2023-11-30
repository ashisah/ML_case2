#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:33:38 2023

@author: Ashley Sah
"""

#%%import libraries
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from utils import loadData, plotVesselTracks
#%%import data

data = loadData('set1.csv') #here we need to instead import the modified features
features = data[:,2:]
labels = data[:,1]

new_features = np.loadtxt('set1_transformed_features.csv', delimiter = ',')


#%%perform basic agglomerative clustering

Z = linkage(features)
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

threshold = 20  # Adjust this value based on the dendrogram
agg_labels_old = fcluster(Z, t=threshold, criterion='maxclust')

plotVesselTracks(features[:,[2,1]], labels)
plotVesselTracks(features[:,[2,1]], agg_labels_old)


#%%perform agglomerative clustering with a custom metric

def custom_distance(pt1, pt2):
    time_diff = pt2[0]-pt1[0]
    speed_diff = pt2[3]-pt1[3]
    dir_diff = pt2[4]-pt1[4]
    if(time_diff == 0):
        return 100000000
    else:
        if(time_diff<0):
            new_long = pt2[5]*time_diff+pt2[2]
            new_lat = pt2[6]*time_diff+pt2[1]
            projected_pos = np.array([new_lat, new_long])
            actual_pos = np.array([pt1[1], pt1[2]])
            return np.linalg.norm(projected_pos-actual_pos) #+ 20*time_diff + 10000*abs(speed_diff*dir_diff)
        else:
            new_long = pt1[5]*time_diff+pt1[2]
            new_lat = pt1[6]*time_diff+pt1[1]
            projected_pos = np.array([new_lat, new_long])
            actual_pos = np.array([pt2[1], pt2[2]])
            return np.linalg.norm(projected_pos-actual_pos)

Z = linkage(new_features, method = 'complete', metric = custom_distance)

plt.title('Hierarchical Clustering Dendrogram')
plt.show()

threshold = 20  # Adjust this value based on the dendrogram
agg_labels = fcluster(Z, t=threshold, criterion='maxclust')

plotVesselTracks(features[:,[2,1]], labels)
plotVesselTracks(features[:,[2,1]], agg_labels)

print(adjusted_rand_score(labels, agg_labels))

#%%perform spectral clustering 
dendrogram(Z)26
#%%perform DBSCAN clustering