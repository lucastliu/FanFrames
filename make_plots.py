import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv 
import datetime
import argparse


## BAR CHART OF MEAN AND STD LATENCIES

x = ['No Computation', 'Blur and Lighting', 'Blur, Lighting, and Object Detection']
## my laptop as client, Ryan's as server
no = [9.264329, 11.734076,  9.289587]
bl = [23.340849, 11.54291, 8.921167,  8.929306]
fd = [12.508818, 15.049769, 5.735671]
# ## both client and server on same laptop:
# no = [0.002728, 0.001299,  0.001848]
# bl = [0.014195, 0.014034, 0.015488]
# fd = [9.561148, 10.882906, 10.533174]

no_m = np.mean(no)
bl_m = np.mean(bl)
fd_m = np.mean(fd)

no_s = np.std(no)
bl_s = np.std(bl)
fd_s = np.std(fd)

y = [no_m, bl_m, fd_m]
print(y)
err = [no_s, bl_s, fd_s]

plt.bar(x, y, yerr=err, capsize=10)
plt.ylabel('Mean Latency (s)')
plt.title('Mean Latencies When Transmitting All Frames')
plt.show()


## LINE GRAPH OF VARIOUS TRIALS ON POTTS WIFI 

def get_sec(row):
        split_time = row['time_dif'].split(':')
        sec = float(split_time[-1])
        minute = float(split_time[-2])
        total = sec + 60*minute 
        if total >3599:
            total=0
        return total

def get_200(num):
    times_df = pd.read_csv('times_'+str(num)+'.csv')
    first_200 = times_df.head(200)
    first_200['time_dif_sec'] = first_200.apply(lambda row: get_sec(row), axis=1)
    return first_200['time_dif_sec']


for trial in [4, 6, 8, 11, 5, 7, 12, 3, 9, 10]:
    if trial in [4,6,8,11]:
        col='#6924AC'
    elif trial in [5,7,12]:
        col='c'
    elif trial in [3,9,10]:
        col='#F78232'
        
    y = get_200(trial)
    plt.plot(range(1,201), y, col)
h1 = mpatches.Patch(color='#F78232', label = 'No Processing')
h2 = mpatches.Patch(color='#6924AC', label = 'Blur and Lighting')
h3 = mpatches.Patch(color='c', label = 'Blur, Lighting, and Object Detection')
plt.legend(handles=[h1, h2, h3])
plt.title('Latency of First 200 Frames')
plt.ylabel('Latency (s)')
plt.xlabel('Frame Number')
plt.show()


## LINE GRAPH OF FIRST 200 FRAMES BOTH CLIENT AND SERVER ON SAME COMPUTER 
for trial in [13, 14, 15, 16, 17, 18, 19, 20, 21]:
    if trial in [16, 17, 18]:
        col='#6924AC'
    elif trial in [19, 20, 21]:
        col='c'
    elif trial in [13, 14, 15]:
        col='#F78232'
        
    y = get_200(trial)
    plt.plot(range(1,201), y, col)
h1 = mpatches.Patch(color='#F78232', label = 'No Processing')
h2 = mpatches.Patch(color='#6924AC', label = 'Blur and Lighting')
h3 = mpatches.Patch(color='c', label = 'Blur, Lighting, and Object Detection')
plt.legend(handles=[h1, h2, h3])
plt.title('Latency of First 200 Frames')
plt.ylabel('Latency (s)')
plt.xlabel('Frame Number')
plt.show()
