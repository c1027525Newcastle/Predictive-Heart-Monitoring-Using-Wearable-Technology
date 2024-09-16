import pandas as pd
import random
import json
import math

# Usage
SEQUENCE_LENGTH = 5
# Path to the input JSON file
input_filename = 'corrected_endomondoHR.json'
# Number of records to select
num_records = 5000
# Target sports to include in the dataset
target_sports = ['bike', 'bike (transport)', 'run', 'hiking', 'rowing', 'core stability training', 'kayaking', 'indoor cycling', 'mountain bike',
                 'walk', 'orienteering']


def count_lines_in_file(filename):
    """Counts the number of lines (records) in the file."""
    with open(filename, 'r') as file:
        return sum(1 for _ in file)
    
    
def select_random_lines_to_df(input_filename, num_records, target_sports):
    """Selects 10 records for each sport in target_sports and additional random lines from the input JSON and stores them in a DataFrame."""

    total_lines = 253020  # This should match the actual number of lines if known.
    
    print(f'Total lines in the file: {total_lines}\n')
    
    selected_records = []
    # Step 1: Ensure we have 10 records for each sport in target_sports
    for sport in target_sports:
        print(f'Adding records for sport: {sport}')
        
        # Select 10 records from the original file matching the sport
        with open(input_filename, 'r') as infile:
            added_count = 0
            for line in infile:
                record = json.loads(line.strip())
                
                if record['sport'] == sport:
                    selected_records.append(record)
                    added_count += 1
                    if added_count >= 10:
                        break  # Stop once we have 10 instances for this sport

    # Step 2: Now, randomly select additional lines to complete num_records
    remaining_records = num_records - len(selected_records)  # How many more records we need
    
    if remaining_records > 0:
        print(f'Adding {remaining_records} additional random records.\n')

        # Randomly select remaining lines to extract
        selected_lines = sorted(random.sample(range(1, total_lines + 1), remaining_records))
    
        line_number = 0
        selected_index = 0
    
        with open(input_filename, 'r') as infile:
            for line in infile:
                line_number += 1
                
                if selected_index < remaining_records and line_number == selected_lines[selected_index]:
                    selected_records.append(json.loads(line.strip()))
                    selected_index += 1
    
    # Convert the selected records to a DataFrame
    df = pd.DataFrame(selected_records)
    
    return df



df = select_random_lines_to_df(input_filename, num_records, target_sports)

print(f'Original df info:\n')
print(f'{df.info()}\n')

df = df[df['sport'].isin(target_sports)]

print(f'After filtering by target sports, Info:\n')
print(f'{df.info()}\n')

df.dropna(inplace=True)
df = df.drop(columns=['gender', 'id', 'userId', 'url'])

print(f'After dropping N/A values Info:')
print(f'{df.info()}\n')


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


# TODOL: Convert lat and longitude to cummulative distance
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


def calculate_cumulative_distances(latitudes, longitudes):
    """
    Calculate the cumulative distances for a series of lat/long points.
    latitudes and longitudes are lists of coordinates.
    """
    cumulative_distances = [0]  # Starting point has 0 distance
    for i in range(1, len(latitudes)):
        distance = haversine_distance(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
        cumulative_distances.append(cumulative_distances[-1] + distance)
    return cumulative_distances


# Create a new column for cumulative distances
df['cumulative_distance'] = df.apply(lambda row: calculate_cumulative_distances(row['latitude'], row['longitude']), axis=1)

# Drop the latitude and longitude columns
df = df.drop(columns=['latitude', 'longitude'])
print('5. Longitude and Latitude columns converted to cumulative_distance\n')


# TODOL: Convert timestamp to real time tracked from 0
# Normalize the 'timestamp' column so that each list starts from 0
def normalize_timestamps(timestamps):
    if len(timestamps) > 0:
        first_value = timestamps[0]
        return [t - first_value for t in timestamps]
    else:
        return timestamps

df['timestamp'] = df['timestamp'].apply(normalize_timestamps)
print('6. Timestamp column normalised to start from 0\n')


# Function to split sequences into chunks of 5
def split_into_chunks(df, sequence_columns, chunk_size):
    new_rows = []
    
    for _, row in df.iterrows():
        min_length = min([len(row[col]) for col in sequence_columns])  # Find the minimum length across all columns for that row
        
        # Only process rows where sequences are divisible by chunk size
        if min_length >= chunk_size:
            num_chunks = min_length // chunk_size  # Calculate number of full chunks
            for i in range(num_chunks):
                new_row = row.copy()
                for col in sequence_columns:
                    new_row[col] = row[col][i*chunk_size:(i+1)*chunk_size]  # Split each sequence into chunks of 5
                new_rows.append(new_row)
    
    # Create a new DataFrame with the split rows
    return pd.DataFrame(new_rows)


# Columns to split into chunks of 5
sequence_columns = ['speed', 'altitude', 'timestamp', 'cumulative_distance', 'heart_rate']

# Split the sequences into chunks of 5
df = split_into_chunks(df, sequence_columns, SEQUENCE_LENGTH)
print(f"7. Sequences split into chunks of {SEQUENCE_LENGTH}\n")


print(f'Final Dataset:\n')
print(f'{df.info()}\n')

df.drop(columns=['sport'], inplace=True) ### Added this line to drop the sport column

# Save the DataFrame to a new JSON file
output_filename = f'Data_Samples/4_random_sample_{num_records}_ml_corrected_endomondoHR.json'
df.to_json(output_filename, orient='records', lines=True)
print(f'Data saved to {output_filename}')
