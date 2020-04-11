# USAGE
# python get_times.py --exp_num number_in_filenames

import pandas as pd 
import numpy as np 
import csv 
import datetime
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-ex", "--exp_num", required=True,
	help="for naming files")

args = vars(ap.parse_args())

client_df = pd.read_csv('client_time_data_'+args['exp_num']+'.csv')
server_df = pd.read_csv('server_time_data_'+args['exp_num']+'.csv')
filename = 'times_'+args['exp_num']+'.csv'

print(client_df.shape)
print(server_df.shape)

def calc_time_dif(row):
    # x is client, y is server
    split_time_x = row['time_x'].split(':')
    sec_x = float(split_time_x[-1])
    minute_x = float(split_time_x[-2])
    hour_x = float(split_time_x[-3])
    split_time_y = row['time_y'].split(':')
    sec_y = float(split_time_y[-1])
    minute_y = float(split_time_y[-2])
    hour_y = float(split_time_y[-3])
    return(datetime.timedelta(hours=hour_y, minutes=minute_y, seconds=sec_y)
    -datetime.timedelta(hours=hour_x, minutes=minute_x, seconds=sec_x))


df_inner = pd.merge(client_df, server_df, on='frame', how='inner')

if df_inner.shape[0]==0:
    print('No frames matched between client and server csvs')
    exit()

df_inner['time_dif'] = df_inner.apply(lambda row: calc_time_dif(row), axis=1)

print(df_inner)


df_inner.to_csv(filename)