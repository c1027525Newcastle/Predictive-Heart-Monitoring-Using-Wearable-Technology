import pandas as pd
import random
import json


# Usage
input_filename = 'corrected_endomondoHR.json'
num_records = 5000
target_sports = [
    'bike',
    'bike (transport)',
    'run',
    'mountain bike',
    'rowing',
    'hiking',
    'orienteering',
    'tennis',
    'core stability training',
    'kayaking',
    'walk',
    'indoor cycling',
    'skate',
    'cross-country skiing'
]


def count_lines_in_file(filename):
    """Counts the number of lines (records) in the file."""
    with open(filename, 'r') as file:
        return sum(1 for _ in file)

def select_random_lines_to_df(input_filename, num_records):
    """Selects random lines from the input JSON and stores them in a DataFrame."""

    total_lines = 253020
    print(f'Total lines in the file: {total_lines}\n')
    
    # Randomly select line numbers to extract
    selected_lines = sorted(random.sample(range(1, total_lines + 1), num_records))
    
    selected_records = []
    line_number = 0
    selected_index = 0
    
    with open(input_filename, 'r') as infile:
        for line in infile:
            line_number += 1
            if selected_index < num_records and line_number == selected_lines[selected_index]:
                selected_records.append(json.loads(line.strip()))
                selected_index += 1
    
    # Convert the selected records to a DataFrame
    df = pd.DataFrame(selected_records)
    
    return df


df = select_random_lines_to_df(input_filename, num_records)

print(f'Original df info:\n{df.info()}\n')

df = df[df['sport'].isin(target_sports)]
print(f'After filtering by target sports, Info:\n{df.info()}\n')

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

# Now pad all sequences to the same length
max_len = df['speed'].apply(len).max()

# Drop rows where heart_rate length is not equal to max_len
df = df[df['heart_rate'].apply(len) == max_len]

def pad_sequences(df, columns, max_len):
    for column in columns:
        df[column] = df[column].apply(lambda x: x + [0] * (max_len - len(x)) if len(x) < max_len else x)  # Padding with zeros
    return df

# Apply padding
sequence_columns = ['speed', 'altitude', 'timestamp', 'longitude', 'latitude', 'heart_rate']
df = pad_sequences(df, sequence_columns, max_len)

# One-hot encode the 'sport' column (explode it into multiple columns)
df = pd.get_dummies(df, columns=['sport'])

# Expand the one-hot encoded 'sport' columns to match the sequence length
def expand_one_hot(df, max_len):
    sport_columns = [col for col in df.columns if col.startswith('sport_')]
    for column in sport_columns:
        df[column] = df[column].apply(lambda x: [x] * max_len)
    return df

# Apply the expansion to the one-hot encoded columns
df = expand_one_hot(df, max_len)

# Drop unnecessary columns
df.dropna(inplace=True)
df = df[df['gender'] != 'female']
df = df.drop(columns=['gender', 'id', 'userId', 'url'])

print(f'After dropping N/A values and female, Info:\n{df.info()}\n')

# Save the DataFrame to a new JSON file
output_filename = f'Data_Samples/3_random_sample_{num_records}_ml_corrected_endomondoHR.json'
df.to_json(output_filename, orient='records', lines=True)
print(f'Data saved to {output_filename}')
