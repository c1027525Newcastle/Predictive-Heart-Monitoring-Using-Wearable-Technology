import os
import sys
import json
import csv
import ast
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math

sport = 'run'

# Load the data
df = pd.read_json('2_5000_corrected_endomondoHR.json', lines=True)

# Remove Things from data
df = df.dropna()
df = df[df['gender'] != 'female']
df = df.drop(columns=['url', 'id', 'userId'])

# Convert string columns to lists
def convert_to_list(x):
    if isinstance(x, list):
        return x  # Already a list
    elif isinstance(x, str):
        return list(map(float, x.strip('[]').split(',')))
    else:
        return [x]

df['speed'] = df['speed'].apply(convert_to_list)
df['altitude'] = df['altitude'].apply(convert_to_list)
df['heart_rate'] = df['heart_rate'].apply(convert_to_list)
df['timestamp'] = df['timestamp'].apply(convert_to_list)
df['longitude'] = df['longitude'].apply(convert_to_list)
df['latitude'] = df['latitude'].apply(convert_to_list)

# Find the first row with the sport bike
bike_row = df[df['sport'] == sport].iloc[0]


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    r = 6371  # Radius of Earth in kilometers. Use 3956 for miles
    return c * r

def calculate_total_distance(latitudes, longitudes):
    """
    Calculate the total distance covered by a series of lat/long points.
    latitudes and longitudes are lists of coordinates.
    """
    total_distance = 0.0
    for i in range(1, len(latitudes)):
        total_distance += haversine_distance(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
    return total_distance

# Choose which record to use
latitude_list = bike_row['latitude']
longitude_list = bike_row['longitude']

# Calculate cumulative distance
total_distance = calculate_total_distance(latitude_list, longitude_list)
print(f"Total distance covered: {total_distance} km")

# Calculate the cumulative distance covered by the user during the workout
cumulative_distances = [0]  # Starting point has 0 distance
for i in range(1, len(latitude_list)):
    distance = haversine_distance(latitude_list[i-1], longitude_list[i-1], latitude_list[i], longitude_list[i])
    cumulative_distances.append(cumulative_distances[-1] + distance)

# print(f"Cumulative distances: {cumulative_distances}")
# print(f"Length of cumulative distances: {len(cumulative_distances)}")

fig, ax1 = plt.subplots()
fig.set_size_inches(14, 8)

# Plot Speed vs. Cumulative Distance on the left Y-axis
ax1.plot(cumulative_distances[:100], bike_row['speed'][:100], label='Speed', color='green', linestyle='-')
ax1.set_xlabel('Cumulative Distance (km)')
ax1.set_ylabel('Speed (km/h)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Create a second y-axis for heart rate
ax2 = ax1.twinx()
ax2.plot(cumulative_distances[:100], bike_row['heart_rate'][:100], label='Heart Rate', color='red', linestyle='-')
ax2.set_ylabel('Heart Rate (bpm)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, title='Type')

ax1.set_xlim(left=0)
ax2.set_xlim(left=0)

plt.title(f'{sport.upper()}: Speed, Heart Rate vs. Distance')
fig.tight_layout()
plt.grid(True)
plt.savefig(f'Plots/{sport}_heart_rate_speed_distance.png')
plt.show()
