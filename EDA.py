#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:04:09 2023

@author: Ashley Sah
"""
#%%import libraries
from utils import loadData, plotVesselTracks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import plotly.express as px
import pandas as pd
#%%load data

data = loadData('set1.csv')
features = data[:,2:]
labels = data[:,1]
#%%plot the individual tracks:
plotVesselTracks(features[:,[2,1]], labels)

indices = np.where(labels == 100002)[0].tolist()
    
for i in range(100001, 100021):
    indices = np.where(labels == i)[0].tolist()
    
    times = features[indices, 0]
    latitude = features[indices, 1]
    longitude = features[indices, 2]
    speeds = features[indices, 3]
    
    time_range = np.ptp(times)
    lat_range = np.ptp(latitude)
    long_range = np.ptp(longitude)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)
    speed_range = np.ptp(speeds)
    
    report = """time range: {0} \nlatitiude range {1}\nlongitude range: {2}\nmax speed: {3}\nmin speed: {4}\nspeed range: {5}""".format(time_range, lat_range, long_range, max_speed, min_speed, speed_range)
    
    plotVesselTracks(features[np.ix_(indices, [2,1])])
    plt.title("Vessel {0} Track".format(i))
    
    plt.text(0.35, -0.15, report, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))

indices = np.where((labels == 100006) | (labels==100005) | (labels==100008))[0].tolist() #why doesn't this work?
plotVesselTracks(features[np.ix_(indices, [2,1])], labels[indices])
plt.title("Vessel 5,6,and 8 Tracks".format(i))

#%%find range of time reports and range of motion generated from each vessel

for i in range(100001, 100021):
    indices = np.where(labels == i)[0].tolist()
    times = features[indices, 0]
    latitude = features[indices, 1]
    longitude = features[indices, 2]
    speeds = features[indices, 3]
    directions = features[indices, 4]
    print("Vessel {0} time range: {1}".format(i, np.ptp(times)))
    print("Vessel {0} latitude range: {1}".format(i, np.ptp(latitude)))
    print("Vessel {0} longitude range: {1}".format(i, np.ptp(longitude)))
    print("Vessel {0} max speed: {1}".format(i, np.max(speeds)))
    print("Vessel {0} min speed: {1}".format(i, np.min(speeds)))
    print("Vessel {0} speed range: {1}".format(i, np.ptp(speeds)))
    print("Vessel {0} max direction: {1}".format(i, np.max(directions)))
    print("Vessel {0} min direction: {1}".format(i, np.min(directions)))
    print("Vessel {0} direction range: {1}".format(i, np.ptp(directions)))
    
    direction_ranges = np.zeros(len(indices)-1)
    dir_change_per_time = np.zeros(len(indices)-1)
    
    filtered_features = features[indices, :]
    sorted_indices = np.argsort(filtered_features[:,0])
    sorted_by_time = filtered_features[sorted_indices, :]
    
    for d in range(0, len(direction_ranges)):
        direction_ranges[d] = abs(sorted_by_time[d+1, 4]-sorted_by_time[d, 4])
        dir_change_per_time = abs(direction_ranges[d]/(sorted_by_time[d+1,0]-sorted_by_time[d,0]))
    
    print("max direction change: {0}".format(np.max(direction_ranges)))
    print("max direction change/time: {0}".format(np.max(dir_change_per_time)))
    
    max_ind_dir_change = np.argmax(direction_ranges)
    print("time change of max direction change: {0}".format(sorted_by_time[max_ind_dir_change+1, 0]-sorted_by_time[max_ind_dir_change, 0]))
    
    print()
    


#%%find how change in direction varies with speed and amt of time passed between points

time_col = 0
dir_col = 4
speed_col = 3
for i in range(100001, 100021):
    
    indices = np.where(labels == i)[0].tolist()
    vessel_reports = features[indices, :]
    
    sorted_indices = np.argsort(vessel_reports[:, time_col])
    reports_sorted_by_time = vessel_reports[sorted_indices, :]
    
    #calculate direction changes and direction change/second
    direction_ranges = np.zeros(len(indices)-1)
    dir_change_per_time = np.zeros(len(indices)-1)
    
    #calculate average speed and time change
    speed_chnge = np.zeros(len(indices)-1)
    time_chnge = np.zeros(len(indices)-1)
    initital_speed = np.zeros(len(indices)-1)
    
    for j in range(0, len(indices)-1):
        direction_ranges[j] = abs(reports_sorted_by_time[j+1, dir_col]-reports_sorted_by_time[j, dir_col])
        time_chnge[j] = reports_sorted_by_time[j+1, time_col]-reports_sorted_by_time[j, time_col]
        
        dir_change_per_time[j] = abs(direction_ranges[j]/time_chnge[j])
        speed_chnge[j] = (reports_sorted_by_time[j+1, speed_col]-reports_sorted_by_time[j, speed_col])/time_chnge[j]
        initital_speed[j] = reports_sorted_by_time[j, speed_col]
    
    
    plt.figure()
    plt.scatter(time_chnge, dir_change_per_time)
    plt.title("Vessel {0}: Direction Change/second vs. Time Change".format(i))
    
    
    """
    plt.figure()
    plt.scatter(speed_chnge, dir_change_per_time)
    plt.title("Vessel {0}: Direction Change/second vs. Speed Change".format(i))
    """
    
    """
    plt.figure()
    plt.scatter(initital_speed, dir_change_per_time)
    plt.title("Vessel {0}: Direction Change/second vs. Initial Speed".format(i))
    """

#%%Try plotting a track over time

features_raw_csv = pd.read_csv('set1.csv', delimiter = ',')
features_raw_csv['SEQUENCE_DTTM'] = pd.to_datetime(features_raw_csv['SEQUENCE_DTTM'], format='%H:%M:%S')
features_raw_csv['time_string'] = features_raw_csv['SEQUENCE_DTTM'].dt.strftime('%H:%M:%S')

pd_one_vessel = features_raw_csv.loc[features_raw_csv['VID'] == 100001]

"""
px.scatter_mapbox(
    features_raw_csv,
    lat = 'LAT',
    lon = 'LON',
    animation_frame = 'time_string'
).update_layout(
    mapbox = {"style": "carto-positron"}
)
   
"""

#%%

fig = plt.figure()
l, = plt.scatter([], [])

plt.xlim(np.min(features_raw_csv['LAT'])-0.05, np.max(features_raw_csv['LAT'])+0.05)
plt.ylim(np.min(features_raw_csv['LON'])-0.05, np.max(features_raw_csv['LON'])+0.05)
writer = PillowWriter(fps = 15, metadata=None)

xlist = []
ylist = []

unique_times = np.unique(features_raw_csv['SEQUENCE_DTTM'])

with writer.saving(fig, "/Users/monugoel/Desktop/CSDS_340/case2/tryone.gif", 100):
    for time in unique_times:
        result_df = features_raw_csv.loc[features_raw_csv['SEQUENCE_DTTM']==time]
        for i in range(0, len(result_df)):
            xlist.append(result_df.iloc[i]['LAT'])
            ylist.append(result_df.iloc[i]['LON'])
        
        l.set_data(xlist, ylist)
        writer.grab_frame()

