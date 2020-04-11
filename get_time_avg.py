# USAGE
# python get_time_avg.py --exp_num number_in_filenames

import pandas as pd 
import numpy as np 
import csv 
import datetime
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-ex", "--exp_num", required=True,
	help="for naming files")

args = vars(ap.parse_args())


def get_sec(row):
    split_time = row['time_dif'].split(':')
    sec = float(split_time[-1])
    minute = float(split_time[-2])
    total = sec + 60*minute 
    if total >3599:
        total=0
    return total

times_df = pd.read_csv('times_'+args['exp_num']+'.csv')

first_200 = times_df.head(200)
first_200['time_dif_sec'] = first_200.apply(lambda row: get_sec(row), axis=1)


print(first_200)
print(first_200.mean())
