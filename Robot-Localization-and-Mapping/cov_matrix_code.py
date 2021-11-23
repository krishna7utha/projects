# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 08:25:31 2021

@author: Krishna Nuthalapati
"""

import numpy as np
import pickle


# Loading data here
p = open('data.pickle', 'rb')
data = np.array(pickle.load(p))
p.close()

# ---------------------Computing covariance matrices--------------------------------# 

def dist(pt):
    return np.sqrt(pt[0]**2 + pt[1]**2)

def bearing(pt1, pt2):
    return np.arctan2(-pt1, pt2)*(180/np.pi)

# Converting from Cartesian to Polar
data_temp = data
for i in range(data_temp.shape[0]):
    for a in data_temp[i]['radar']:
        data_temp[i]['radar'][a][0] -= 7.5 #Shifting landmarks left (on X-axis)
        temp = data_temp[i]['radar'][a][0]
        data_temp[i]['radar'][a][0] = dist(data_temp[i]['radar'][a])
        data_temp[i]['radar'][a][1] = bearing(temp,data_temp[i]['radar'][a][1])

# Measurement covariance
temp = np.zeros((5000, 6 ,2)) # 5000 for 50 seconds, 6 for initial landmarks, 2 for (x,y)
for i in range(5000):
    idx = 0
    for a in data_temp[i]['radar']:
        temp[i, idx, :] = data_temp[i]['radar'][a]
        idx += 1

measurement_cov = np.zeros((2,2), dtype=float)
for i in range(6):
    cov = np.cov(temp[:,i,0],temp[:,i,1])
    measurement_cov += cov
    
measurement_cov /= 6
print('\nMeasurement Covariance matrix')
print(measurement_cov)

# Control covariance
temp = np.zeros((5000,2))
for i in range(5000):
    temp[i,0] = data[i]['odometry']
    temp[i,1] = data[i]['gyroscope']

control_cov = np.cov(temp[:,0], temp[:,1])
print('\nControl Covariance matrix')
print(control_cov)


