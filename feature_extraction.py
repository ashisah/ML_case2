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

#%%Import data
data = loadData('set1.csv')
features = data[:,2:]
labels = data[:,1]

#%%Extract x-speed (change in longitude/sec) and y-speed (change in latitude/sec)
speed_col = 3
dir_col = 4

speeds = features[:, speed_col]
degrees = features[:, dir_col]

degrees = degrees/10
speeds_knots = speeds/10


radian_dir = degrees*np.pi/180

x_speed = 7
y_speed = 10



