import pandas as pd

# Load the data
df = pd.read_json('4_random_sample_5000_ml_corrected_endomondoHR.json', lines=True)

# Print all unique values in the 'sport' column
df.dropna(inplace=True)
unique_sports = df['sport'].unique()

print("Unique sports in the dataset:\n")
i = 1
for sport in unique_sports:
    print(i, sport)
    i += 1

