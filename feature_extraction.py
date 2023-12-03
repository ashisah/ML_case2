#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:02:07 2023

@author: Ashley Sah
"""

#%%Import libraries
from utils import loadData, plotVesselTracks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%Import data
data = loadData('set1.csv')
features = data[:,2:]
labels = data[:,1]

#%%Extract x-speed (change in longitude/sec) and y-speed (change in latitude/sec)
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

new_features = np.concatenate((features[:, 0:3], speeds_knots_per_sec, radian_dir, x_speed_per_sec, y_speed_per_sec, features[:, 4][..., np.newaxis]) ,axis = 1)

#%%write out new features to csv
new_feats = pd.DataFrame(new_features)
new_feats.to_csv('set1_transformed_features.csv', header = False, index = False)

