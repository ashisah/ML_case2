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
    dir_col = 4
    speed_col = 3
    
    radians = ((90 - (testFeatures[:, dir_col]/10))%360)*np.pi/180
    speeds = testFeatures[:, speed_col]/10
    
    x_speed = speeds*np.cos(radians)
    y_speed = speeds*np.sin(radians)
    
    new_features = np.concatenate((testFeatures[:, 0:3], x_speed[..., np.newaxis], y_speed[...,np.newaxis]), axis =1)
    
    def custom_distance(pt1, pt2):
        def compute(x1,y1,x2,y2,oldlat,oldlon):
            newlat =   oldlat + -1.19744794e-07*x1 +\
                    y1 * 3.30944844e-06 + x2* 1.42758346e-07 +\
                    y2* 1.28076506e-06 + 1.15731537e-06
            newlon =   oldlon +  3.64863143e-06 *x1 +\
                y1 * -1.21328244e-06 + x2* 2.11551448e-06 +\
                y2* 1.18718185e-06+ 3.55361593e-07

            return newlat,newlon

        dt = pt2[0]-pt1[0]
        x1,x2 = pt1[3],pt2[3]
        y1,y2 = pt1[4],pt2[4]
        lat1,lat2 = pt1[1],pt2[1]
        lon1,lon2 = pt1[2],pt2[2]
        
        if(dt == 0):
            return 100000000
        if(dt>0):
            new_lat,new_long = compute(dt*x1,dt*y1,dt*x2,dt*y2,lat1,lon1)
            projected_pos = np.array([new_lat, new_long])
            actual_pos = np.array([pt2[1], pt2[2]])
            return np.linalg.norm(projected_pos-actual_pos) #+ 20*time_diff + 10000*abs(speed_diff*dir_diff)
        else:
            dt *=-1
            new_lat,new_long = compute(dt*x2,dt*y2,dt*x1,dt*y1,lat2,lon2)
            projected_pos = np.array([new_lat, new_long])
            actual_pos = np.array([pt1[1], pt1[2]])
            return np.linalg.norm(projected_pos-actual_pos)
    #((90 - (data['COURSE_OVER_GROUND'] /10 ))%360 )* np.pi/180
    
    Z = linkage(new_features, method = 'average', metric = custom_distance)
    
    predVessels = fcluster(Z, t=numVessels, criterion='maxclust')
    
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 12, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set2.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK= predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK= predictWithoutK(features)
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
    