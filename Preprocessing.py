# -*- coding: utf-8 -*-

"""
@Original author: Umair Ahmed (Thu Nov 17 23:53:32 2016)

@Modified by: Yingying Zhou (2023)
"""

#important imports
import pandas as pd
import math
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
import csv


#Functions for calculating the Magnitude of any axis of Raw sensor Data
def magnitude(user_id):
    x2 = user_id['xAxis'] * user_id['xAxis']
    y2 = user_id['yAxis'] * user_id['yAxis']
    z2 = user_id['zAxis'] * user_id['zAxis']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

# Understanding: Applies the magnitude computation to each segment. 
# The resulting 'magnitude' axis is treated as a fourth sensor dimension 
# and included in feature extraction alongside x, y, and z.
def calc_magnitudes():
    for i in range (1,19):
        user_list[i-1]['magnitude'] = magnitude(user_list[i-1])
        

#Function for defining the window on data
# Understanding: Splits the time series into overlapping windows of fixed size (default: 100 samples).
# The window advances by 50% of its length (dx/2), ensuring each point can appear in multiple windows.
# This increases sample count and improves coverage of behavior patterns.
def window(axis,dx=100):
    start = 0;
    size = axis.count();

    while (start < size):
        end = start + dx
        yield start,end
        start = start+int (dx/2)
        


#Features which are extracted from Raw sensor data
# Understanding: Computes statistical summaries (e.g., mean, std, skewness) 
# for a single sensor axis over a specific time window.
def window_summary(axis, start, end):
    acf = stattools.acf(axis[start:end])
    acv = stattools.acovf(axis[start:end])
    sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
    return [
        axis[start:end].mean(),
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        acf.mean(), # mean auto correlation
        acf.std(), # standard deviation auto correlation
        acv.mean(), # mean auto covariance
        acv.std(), # standard deviation auto covariance
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
        math.sqrt(sqd_error.mean())
    ]

# Understanding: This function performs feature extraction for each segment.
# It applies overlapping windowing (defined in window()) and computes 
# multi-axis statistics per window. Each window becomes one training example.
def features(user_id):
    for (start, end) in window(user_id['timestamp']):
        features = []
        for axis in ['xAxis', 'yAxis', 'zAxis', 'magnitude']:
            features += window_summary(user_id[axis], start, end)
        yield features        

   
     

#Main code for Pre-processing of the Data
# Understanding: This defines the column names for accelerometer data.
# Only x/y/z axes are used for feature computation. Timestamp is used for window segmentation but not as a feature.
COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']

# Understanding: Initializes a list to store 18 segments of walking data.
# Each segment is treated as an independent instance for feature extraction and classification.
user_list = []

titles_list=[]
user_to_auth = 0

# Understanding: Each CSV file is loaded and trimmed to the first 1100 rows.
# This standardizes segment length across samples, likely to ensure uniformity during statistical analysis.
# Alternatively, all CSV files could be read dynamically from the directory,
# but fixed index filenames ensure consistent order and reproducibility.
for i in range (1,19):
    file_path = 'd:\桌面\code\Dataset/'+str(i)+'.csv'
    user_list.append((pd.read_csv(file_path,header=None,names=COLUMNS))[:1100])

#Add an additional axis of magnitude of the sensor data
calc_magnitudes() 

#Write the feature vectors to a separate excel file
# Understanding: Opens the output file and writes each extracted feature vector to CSV.
# Each segment produces multiple vectors - one per time window - 
# all rows are prefixed with that segment's index to preserve source identity.
with open('d:\桌面\code\Features\Features.csv', 'w') as out:
    rows = csv.writer(out)
    for i in range(0, len(user_list)):
        for f in features(user_list[i]):
            rows.writerow([i]+f)
                
# Understanding: The script writes extracted features into a CSV file. 
# Each segment produces multiple feature vectors - one per time window - 
# and each row is prefixed with the segment index.

# Understanding: For each segment (user_list[i]), generate a sequence of feature vectors. 
# These are collected from all time windows defined within the segment.