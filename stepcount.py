import pywear
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import datetime
import glob
import os
import argparse
import time

# Computational timing
start = time.time()

# Step 1 - Create CSV from .cwa file
parser = argparse.ArgumentParser()
parser.add_argument("cwa", help = "Enter location of .cwa filename to be processed")
parser.add_argument("predictions", help = "Enter location of .csv timeseries file from Biobank Accelerometer Analysis Tool")
parser.add_argument('--sampleRate', default = 100, type = int, help = "Enter sample rate of accelerometer data")
args = parser.parse_args()

acc_data, info = pywear.read_device(args.cwa, lowpass_hz=20, calibrate_gravity=True, detect_nonwear=True, resample_hz=args.sampleRate)
acc_data['time'] = acc_data.index
acc_data = acc_data.reset_index(drop=True)



#Step 2 - Import epoch data from BAAT

step_epochs = pd.read_csv(args.predictions)
poi = args.predictions[:-15]

# Define data type to analyze
epoch_length = 15 # Number of seconds in each epoch -- defined
steps_per_epoch = 5 # Number of steps per epoch in defined walking classification
samplerate = args.sampleRate #Hz

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

#print("Distance is:", distance_list[0])
#print("Prominence is:", prominence_list[0])
           
# Step Counting across all accelerometer data, to give a rough step count without RF/HMM walking classification
steps, _ = find_peaks(acc_data['ENMO'], distance = distance_list[0], prominence = prominence_list[0])
print("Peak-detection-only step count:", len(steps), "steps")
whole_file_steps = len(steps)


## Separate classified walking epochs
walking_epochs = step_epochs[(step_epochs['walk'] >0.5)].reset_index(drop = True)
time_walking = len(walking_epochs)*epoch_length/60 #minutes
print("Total classified step time:", time_walking, "min")


merged = pd.merge_asof(acc_data, step_epochs, on = 'time')
all_steps=merged.iloc[steps,]
counted_steps = all_steps[all_steps['walk'] == 1]
print("Total steps in file:", len(counted_steps), "steps")


# Sum steps by hour
hourly = counted_steps.resample('H', on = 'time').walk.sum()
#print(hourly)

# Output hourly stepcounts
hourly_output_path = "{}_HourlySteps.csv".format(poi)
hourly.to_csv(hourly_output_path)

#Sum steps by day
daily = counted_steps.resample('D', on = 'time').walk.sum()
#print(daily)

# Output daily stepcounts
daily_output_path = "{}_DailySteps.csv".format(poi)
daily.to_csv(daily_output_path)

# Computational Timing
end = time.time()
print(f"Done! ({round(end - start,2)}s)")
