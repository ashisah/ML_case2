# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    speed_col = 3
    dir_col = 4
   
    speeds = testFeatures[:, speed_col]
    degrees = testFeatures[:, dir_col]
    
    degrees = degrees/10 #converting to degrees
    degrees = (90-degrees)%360 #converting to correct orientation
    speeds_knots = speeds/10 #converting to nm/hr
    speeds_knots_per_sec = speeds_knots/3600 #converting to nm/sec
    
    radian_dir = degrees*np.pi/180
    
    x_speed_per_sec = 1/48*speeds_knots_per_sec*np.cos(radian_dir)*-1
    
    y_speed_per_sec = 1/60*speeds_knots_per_sec*np.sin(radian_dir)
    
    speeds_knots_per_sec = speeds_knots_per_sec[..., np.newaxis]
    x_speed_per_sec = x_speed_per_sec[...,np.newaxis]
    y_speed_per_sec = y_speed_per_sec[...,np.newaxis]
    radian_dir = radian_dir[...,np.newaxis]

    new_features = np.concatenate((testFeatures[:, 0:3], speeds_knots_per_sec, radian_dir, x_speed_per_sec, y_speed_per_sec),axis = 1)

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
    
    
    Z = linkage(new_features, method = 'average', metric = custom_distance)
    
    agg_labels = fcluster(Z, t=numVessels, criterion='maxclust')
    
    """
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
                random_state=100)
    predVessels = km.fit_predict(testFeatures)
    """
    
    return agg_labels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    