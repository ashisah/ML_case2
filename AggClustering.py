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

new_features = np.loadtxt('set1_transformed_features_plus_speed.csv', delimiter = ',')


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
    dir_diff = abs(pt2[4]-pt1[4])
    dir_diff = np.min([2*np.pi-dir_diff, dir_diff])
    if(time_diff == 0):
        return 100000000
    else:
        if(time_diff<0):
            if(abs(time_diff)<100):
               new_long = pt2[5]*time_diff*-1+pt2[2]
               new_lat = pt2[6]*time_diff*-1+pt2[1]
               projected_pos = np.array([new_lat, new_long])
               actual_pos = np.array([pt1[1], pt1[2]])
               return np.linalg.norm(projected_pos-actual_pos)+pt2[3]*dir_diff #+ 20*time_diff + 10000*abs(speed_diff*dir_diff)
            else:
                new_long = pt2[5]*100+pt2[2]
                new_lat = pt2[6]*100+pt2[1]
                projected_pos = np.array([new_lat, new_long])
                actual_pos = np.array([pt1[1], pt1[2]])
                return np.linalg.norm(projected_pos-actual_pos)+pt2[3]*dir_diff 
        else:
            if(abs(time_diff)<100):
                new_long = pt1[5]*time_diff+pt1[2]
                new_lat = pt1[6]*time_diff+pt1[1]
                projected_pos = np.array([new_lat, new_long])
                actual_pos = np.array([pt2[1], pt2[2]])
                return np.linalg.norm(projected_pos-actual_pos)+pt1[3]*dir_diff
            else:
                new_long = pt1[5]*100+pt1[2]
                new_lat = pt1[6]*100+pt1[1]
                projected_pos = np.array([new_lat, new_long])
                actual_pos = np.array([pt2[1], pt2[2]])
                return np.linalg.norm(projected_pos-actual_pos)+pt1[3]*dir_diff

Z = linkage(new_features, method = 'average', metric = custom_distance)

plt.title('Hierarchical Clustering Dendrogram')
plt.show()

threshold = 20  # Adjust this value based on the dendrogram
agg_labels = fcluster(Z, t=threshold, criterion='maxclust')

plotVesselTracks(features[:,[2,1]], labels)
plotVesselTracks(features[:,[2,1]], agg_labels)

print(adjusted_rand_score(labels, agg_labels))

#%%perform spectral clustering 
dendrogram(Z)
#%%Set 3 test

import matplotlib.pyplot as plt
from matplotlib import markers,colors

data = loadData('set3noVID.csv')
features = data[:,2:]
labels = data[:,1]

speed_col = 3
dir_col = 4

speeds = features[:, speed_col]
degrees = features[:, dir_col]

degrees = degrees/10 #converting to degrees
degrees = (90-degrees)%360 #converting to correct orientation
speeds_knots = speeds/10 #converting to nm/hr
speeds_knots_per_sec = speeds_knots/3600 #converting to nm/sec

radian_dir = degrees*np.pi/180


#degrees of longitude will vary in distance depending on how far you are from the equator
#at 36.99 degree latitude the distance between a degree of longitude is 48 nautical miles
x_speed_per_sec = 1/48*speeds_knots_per_sec*np.cos(radian_dir)
#degrees of latitiude are 60 nautical miles apart
y_speed_per_sec = 1/60*speeds_knots_per_sec*np.sin(radian_dir)

speeds_knots_per_sec = speeds_knots_per_sec[..., np.newaxis]
x_speed_per_sec = x_speed_per_sec[...,np.newaxis]
y_speed_per_sec = y_speed_per_sec[...,np.newaxis]
radian_dir = radian_dir[...,np.newaxis]

new_features = np.concatenate((features[:, 0:3], speeds_knots_per_sec, radian_dir, x_speed_per_sec, y_speed_per_sec),axis = 1)

def custom_distance(pt1, pt2):
    time_diff = pt2[0]-pt1[0]
    speed_diff = pt2[3]-pt1[3]
    dir_diff = abs(pt2[4]-pt1[4])
    dir_diff = np.min([2*np.pi-dir_diff, dir_diff])
    if(time_diff == 0):
        return 100000000
    else:
        if(time_diff<0):
            new_long = pt2[5]*time_diff*-1+pt2[2]
            new_lat = pt2[6]*time_diff*-1+pt2[1]
            projected_pos = np.array([new_lat, new_long])
            actual_pos = np.array([pt1[1], pt1[2]])
            return np.linalg.norm(projected_pos-actual_pos) +10*pt2[3]*dir_diff#+ 20*time_diff + 10000*abs(speed_diff*dir_diff)
        else:
            new_long = pt1[5]*time_diff+pt1[2]
            new_lat = pt1[6]*time_diff+pt1[1]
            projected_pos = np.array([new_lat, new_long])
            actual_pos = np.array([pt2[1], pt2[2]])
            return np.linalg.norm(projected_pos-actual_pos) +10*pt1[3]*dir_diff
            
Z = linkage(new_features, method = 'average', metric = custom_distance)


threshold = 12  # Adjust this value based on the dendrogram
agg_labels_12 = fcluster(Z, t=threshold, criterion='maxclust')

plotVesselTracks(features[:, [2,1]], agg_labels_12)

unique_labels = np.unique(agg_labels_12)
print(unique_labels)
for label in unique_labels:
    indices = np.where(agg_labels_12 == label)[0].tolist()
    plotVesselTracks(features[np.ix_(indices, [2,1])])
  
