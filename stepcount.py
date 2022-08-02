import actipy
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import datetime
import argparse
import time
import pytz
import re


def main(args):

    # Computational timing
    start = time.time()

    # Step 1 - Create CSV from .cwa file
    acc_data, info = actipy.read_device(args.cwa, lowpass_hz=args.lowpass_hz, calibrate_gravity=True, detect_nonwear=True)
    acc_data.index = machine2utc(acc_data.index)
    acc_data['time'] = acc_data.index
    acc_data = acc_data.reset_index(drop=True)

    # Step 2 - Import epoch data from BAAT

    step_epochs = pd.read_csv(args.predictions, parse_dates=['time'], date_parser=date_parser)
    step_epochs['time'] = step_epochs['time'].dt.tz_convert('UTC')
    poi = args.predictions[:-15]

    # Define data type to analyze
    epoch_length = 5  # Number of seconds in each epoch, optimized in iterative parameter testing
    steps_per_epoch = 2  # Number of steps per epoch in defined walking classification, optimized in iterative parameter testing
    samplerate = info['SampleRate']  # Hz

    # Setting peak parameters
    # a distance of 0.32 seconds, optimized in iterative parameter testing for 100 Hz AX3
    distance_list = [32 * (samplerate / 100)]
    prominence_list = [.140]  # gravitational units, optimized in iterative parameter testing for 100 Hz AX3

    # Setting up data collection dataframe
    d = []

    # Import acceleration data and calculate unfiltered ENMO Trunc
    acc_data['ENMO'] = ((acc_data['x']**2 + acc_data['y']** 2 + acc_data['z']**2)**0.5)-1
    acc_data.loc[acc_data['ENMO'] < 0, 'ENMO'] = 0
    print("This is data for participant:", poi)

    # Step Counting across all accelerometer data, to give a rough step count without RF/HMM walking classification
    steps, _ = find_peaks(acc_data['ENMO'], distance=distance_list[0], prominence=prominence_list[0])
    print("Peak-detection-only step count:", len(steps), "steps")
    whole_file_steps = len(steps)

    # Separate classified walking epochs
    walking_epochs = step_epochs[(step_epochs['walk'] > 0.5)].reset_index(drop=True)
    time_walking = len(walking_epochs) * epoch_length / 60  # minutes
    print("Total classified step time:", time_walking, "min")

    merged = pd.merge_asof(acc_data, step_epochs, on='time')
    all_steps = merged.iloc[steps, ]
    counted_steps = all_steps[all_steps['walk'] == 1]
    print("Total steps in file:", len(counted_steps), "steps")

    # Sum steps by hour
    hourly = counted_steps.resample('H', on='time').walk.sum()

    # Output hourly stepcounts
    hourly_output_path = "{}_HourlySteps.csv".format(poi)
    hourly.to_csv(hourly_output_path)

    # Sum steps by day
    daily = counted_steps.resample('D', on='time').walk.sum()

    # Output daily stepcounts
    daily_output_path = "{}_DailySteps.csv".format(poi)
    daily.to_csv(daily_output_path)

    # Output step info for all counted steps
    step_output_path = "{}_StepInfo.csv".format(poi)
    counted_steps.to_csv(step_output_path)

    # Computational Timing
    end = time.time()
    print(f"Done! ({round(end - start,2)}s)")


def machine2utc(t, tz='Europe/London'):
    """ Convert machine time to UTC time, fixing for DST crossovers if any.

    Notes:
        - Non-existent times are due to a push forward, so it is in DST
        - Ambiguous times are assumed to be in DST, i.e. before the pull back
        - Only fixes one DST crossover

    """

    tz = pytz.timezone(tz)

    t_start = t[0].tz_localize(tz, ambiguous=True)
    t_end = t[-1].tz_localize(tz, ambiguous=True, nonexistent='shift_forward')
    dst_shift = t_end.dst() - t_start.dst()

    if not (abs(dst_shift) > datetime.timedelta(0)):
        # Nothing to do. Just convert to UTC
        return t.tz_localize(tz, ambiguous=True).tz_convert("UTC")

    # Convert to UTC. This is only correct up till the DST transition
    t_utc = (t.tz_localize(tz, ambiguous=True, nonexistent='NaT')
             .tz_convert("UTC")
             .tz_convert(None))

    # Find when transition happens
    t_trans = tz._utc_transition_times[np.searchsorted(tz._utc_transition_times, t_utc[0])]

    # Now correct the local times after transition
    t = t.to_series(t.name)
    whr_befor_trans = ~(t_utc < t_trans)  # not the same as t_utc >= t_trans due to NaTs
    t[whr_befor_trans] += dst_shift

    # Finally convert to UTC
    start_is_dst = t_start.dst() > t_end.dst()
    end_is_dst = not start_is_dst
    t_before = t[~whr_befor_trans].dt.tz_localize(tz, ambiguous=start_is_dst).dt.tz_convert("UTC")
    t_after = t[whr_befor_trans].dt.tz_localize(tz, ambiguous=end_is_dst).dt.tz_convert("UTC")

    t = pd.concat((t_before, t_after))

    return t


def date_parser(t):
    '''
    Parse date a date string of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    '''
    tz = re.search(r'(?<=\[).+?(?=\])', t)
    if tz is not None:
        tz = tz.group()
    t = re.sub(r'\[(.*?)\]', '', t)
    return pd.to_datetime(t, utc=True).tz_convert(tz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cwa", help="Enter location of .cwa filename to be processed")
    parser.add_argument("predictions", help="Enter location of .csv timeseries file from Biobank Accelerometer Analysis Tool")
    parser.add_argument('--lowpass_hz', default=False, type=int, help="Enter lowpass filter Hz if one is desired")
    args = parser.parse_args()

    main(args)
