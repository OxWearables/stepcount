import pywear
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import datetime
import glob
import os

# Step 1 - Create CSV from .cwa file
acc_data, info = pywear.read_device("elena.cwa", lowpass_hz=20, calibrate_gravity=True, detect_nonwear=True, resample_hz=100)
acc_data['time'] = acc_data.index
acc_data = acc_data.reset_index(drop=True)

#Step 2 - Import epoch data from BAAT

step_epochs = pd.read_csv("elena-timeSeries.csv")
poi = "elena"

# Define data type to analyze
epoch_length = 15 # Number of seconds in each epoch -- defined
steps_per_epoch = 5 # Number of steps per epoch in defined walking classification
samplerate = 100 #Hz

# Setting peak parameters
distance_list = [35*(samplerate/100)] # a distance of 0.35 seconds
prominence_list = [.175] # gravitational units


# Setting up data collection dataframe
d = []

# Import acceleration data and calculate unfiltered ENMO Trunc
acc_data['ENMO'] = ((acc_data['x']**2 + acc_data['y']**2 + acc_data['z']**2)**0.5)-1
acc_data.loc[acc_data['ENMO'] < 0, 'ENMO'] = 0
print("This is data for participant:", poi)

    
## Make timestamps workable
step_epochs['time'] = step_epochs['time'].str[:23]
step_epochs['time'] = pd.to_datetime(step_epochs['time'], format= "%Y-%m-%d %H:%M:%S.%f")
acc_data['time'] = pd.to_datetime(acc_data['time'], format= "%Y-%m-%d %H:%M:%S.%f")

   
    

print("Distance is:", distance_list[0])
print("Prominence is:", prominence_list[0])
           
# Step Counting across all accelerometer data, to give a rough step count without RF/HMM walking classification
steps, _ = find_peaks(acc_data['ENMO'], distance = distance_list[0], prominence = prominence_list[0])
print("Peak-detection-only step count:", len(steps), "steps")
whole_file_steps = len(steps)


## Separate classified walking epochs
walking_epochs = step_epochs[(step_epochs['walk'] >0.5)].reset_index(drop = True)
time_walking = len(walking_epochs)*epoch_length/60 #minutes
print("Total walking time:", time_walking, "min")

# For loop to count steps over each epoch
       

total = 0
for count in range(1,len(walking_epochs)): #Ignore the first epoch, because sometimes it starts/ends too early
    epoch_window_start = walking_epochs['time'][count]
    epoch_window_end = epoch_window_start + datetime.timedelta(seconds = epoch_length)
    window = acc_data['time'][(acc_data['time']>= epoch_window_start) & (acc_data['time'] < epoch_window_end)]
    window_rows = window.index.tolist()
    epoch_steps = steps[(steps >= min(window_rows)) & (steps < max(window_rows))]
    #print("Epoch Start:", epoch_window_start, "Epoch End:", epoch_window_end)
    #print("The total number of steps in this epoch is:", len(epoch_steps))
    total = total +len(epoch_steps)
print("RF/HMM + Peak Detection Calculated Steps:", total, "steps", "\n")



d.append({'participant': poi,'Peak_Only_Steps': len(steps),'Walking_Time': time_walking,'RFF/HMM+Peak_Steps':total})
            
            
final_output = pd.DataFrame(d)
output_path = "{}_total_steps.csv".format(poi)
final_output.to_csv(output_path, index= False)
