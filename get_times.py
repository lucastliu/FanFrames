import pandas as pd 
import numpy as np 
import csv 
import datetime


client_df = pd.read_csv('client_time_data_1.csv')
server_df = pd.read_csv('server_time_data_1.csv')
filename = 'times_1.csv'

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