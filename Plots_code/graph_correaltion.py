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


# Load the data
df = pd.read_json('2_10000_corrected_endomondoHR.json', lines=True)

# Remove Things from data
df = df.dropna()
df = df.drop(columns=['url', 'id', 'userId', 'gender'])


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

# Convert sport to categorical and encode it as numbers
df['sport'] = df['sport'].astype('category').cat.codes


max_size = {
    'speed': df['speed'].apply(len).max(),
    'altitude': df['altitude'].apply(len).max(),
    'heart_rate': df['heart_rate'].apply(len).max(),
    'timestamp': df['timestamp'].apply(len).max(),
    'longitude': df['longitude'].apply(len).max(),
    'latitude': df['latitude'].apply(len).max(),
}

# Step 2: Filter the DataFrame
df_filtered = df[
    (df['speed'].apply(len) == max_size['speed']) &
    (df['altitude'].apply(len) == max_size['altitude']) &
    (df['heart_rate'].apply(len) == max_size['heart_rate']) &
    (df['timestamp'].apply(len) == max_size['timestamp']) &
    (df['longitude'].apply(len) == max_size['longitude']) &
    (df['latitude'].apply(len) == max_size['latitude'])
]

# Step 3: Print the number of records that are taken into account
print(f"Number of records considered for analysis: {len(df_filtered)}")

# Exploding the lists into individual rows
df_exploded = df_filtered.explode(['speed', 'altitude', 'heart_rate', 'timestamp', 'longitude', 'latitude'])

# Convert the columns to numeric after exploding to ensure proper correlation calculation
df_exploded['speed'] = pd.to_numeric(df_exploded['speed'], errors='coerce')
df_exploded['altitude'] = pd.to_numeric(df_exploded['altitude'], errors='coerce')
df_exploded['heart_rate'] = pd.to_numeric(df_exploded['heart_rate'], errors='coerce')
df_exploded['longitude'] = pd.to_numeric(df_exploded['longitude'], errors='coerce')
df_exploded['latitude'] = pd.to_numeric(df_exploded['latitude'], errors='coerce')

# Dropping any NaN values that might have been introduced
df_exploded = df_exploded.dropna()

# Step 3: Now calculate the correlation matrix on the exploded data
plt.figure(figsize=(10, 8))
correlation_matrix = df_exploded[['speed', 'heart_rate', 'altitude', 'longitude', 'latitude']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap (Exploded Data)')
plt.savefig('Plots/2_correlation_heatmap_exploded.png')
plt.show()

