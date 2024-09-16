import pandas as pd
import numpy as np
import math
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# Choose the number of data points and epochs
DATA_NUM = 7500
EPOCH_NUM = 500
NORMALISATION = 'TRUE' # Choose 'TRUE' if normalisation is wanted

# Set the length of the sequences used as input to the model
SEQUENCE_LENGTH = 5

#ML_Models/gru_model_3_data7500_epoch500_sequence5_NomalisationTRUE.pth
#ML_Models/gru_model_3_data5000_epoch250_sequence5_NomalisationTRUE.pth

# Paths to save the model dictionary and full model
DICTIONARY_PATH = f'ML_Models/gru_model_3_data{DATA_NUM}_epoch{EPOCH_NUM}_sequence{SEQUENCE_LENGTH}_Nomalisation{NORMALISATION}.pth'
FULL_MODEL_PATH = f'ML_Models/gru_model_3_data{DATA_NUM}_epoch{EPOCH_NUM}_sequence{SEQUENCE_LENGTH}_Normalisation{NORMALISATION}_full.pth'


# Load the data
df = pd.read_json(f'2_{DATA_NUM}_corrected_endomondoHR.json', lines=True)
print('1. Data loaded\n')

# Drop the 'gender', 'id', 'userId', 'url' columns as they are not needed
df = df.drop(columns=['gender', 'id', 'userId', 'url'])
df = df.dropna()  # Drop rows with missing values
print('2. Un-useful columns and rows dropped\n')

# Function to convert columns with lists of values to actual lists
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
print('3. Columns converted to lists\n')


# Print unique sports before encoding
unique_sports = df['sport'].unique()
print(f'Unique sports before encoding: {unique_sports}\n')


# Convert the sport column to categorical codes
df['sport'] = df['sport'].astype('category').cat.codes
print('4. Sport column converted to categorical codes\n')


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


# TODO: Maybe normalise the numbers?
# Normalize selected columns
def normalize_column(column):
    scaler = MinMaxScaler()
    column_values = np.concatenate(df[column].values).reshape(-1, 1)  # Flatten the sequence column
    scaler.fit(column_values)
    
    # Apply normalization to each sequence in the column
    df[column] = df[column].apply(lambda x: scaler.transform(np.array(x).reshape(-1, 1)).flatten())
    
    return scaler

# Normalize columns that need normalization
if NORMALISATION == 'TRUE':
    scalers = {}
    scalers['speed'] = normalize_column('speed')
    scalers['altitude'] = normalize_column('altitude')
    scalers['cumulative_distance'] = normalize_column('cumulative_distance')
    scalers['timestamp'] = normalize_column('timestamp')
    scalers['heart_rate'] = normalize_column('heart_rate')

    print('Columns normalised for model training\n')


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
print(f'Dataframe info\n')
print(df.info())


############################################################################################################


# Combine the features into a single tensor
X = df[['speed', 'altitude', 'timestamp', 'cumulative_distance']]  # Select the features
X = np.stack([np.array(X[col].tolist()) for col in X.columns], axis=-1)  # Shape: (batch_size, sequence_length, num_features)
y = np.array(df['heart_rate'].apply(lambda x: x[-1]).tolist())  # Shape: (batch_size,) - Only last heart rate value
sport = df['sport'].values  # Extract sport as a separate array
print('6. Features and target combined\n')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, sport_train, sport_test = train_test_split(X, y, sport, test_size=0.2, random_state=42)
print('7. Data split into training and testing sets\n')

# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # Make y a column vector
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
sport_train = torch.tensor(sport_train, dtype=torch.long)
sport_test = torch.tensor(sport_test, dtype=torch.long)

# Define the GRU-based neural network with embedding
class HeartRateGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_sports, embedding_dim):
        super(HeartRateGRUModel, self).__init__()
        self.embedding = nn.Embedding(num_sports, embedding_dim)
        self.gru = nn.GRU(input_size + embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sport):
        sport_embedded = self.embedding(sport)
        sport_embedded = sport_embedded.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, sport_embedded), dim=2)
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)  # GRU forward pass
        out = self.fc(out[:, -1, :])  # Take the last time step for each sequence
        return out

# Model parameters
input_size = X_train.shape[2]  # Number of features in each time step
hidden_size = 64
num_layers = 2
output_size = 1  # Predicting a single value (heart rate)
num_sports = df['sport'].nunique()  # Number of unique sports
embedding_dim = 5  # Embedding dimension for the sport

# Instantiate the model, loss function, and optimizer
model = HeartRateGRUModel(input_size, hidden_size, num_layers, output_size, num_sports, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate of 0.001

# Training loop
for epoch in range(EPOCH_NUM):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train, sport_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{EPOCH_NUM}, Loss: {loss.item()}')
print('8. Model trained\n')


# Make predictions
print('9. Making predictions\n')
model.eval()
with torch.no_grad():
    y_pred = model(X_test, sport_test)

# Convert to numpy arrays for evaluation
y_pred = y_pred.numpy()

# Inverse transform heart rate predictions and actual values if normalized
if NORMALISATION == 'TRUE':
    y_pred = scalers['heart_rate'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test = scalers['heart_rate'].inverse_transform(y_test).flatten()

print('10. Calculating metrics\n')
# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Calculate the R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.2f}\n')

# Convert regression predictions to binary classification
threshold = 100
y_pred_class = (y_pred > threshold).astype(int)
y_test_class = (y_test > threshold).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class, zero_division=1)
recall = recall_score(y_test_class, y_pred_class, zero_division=1)
f1 = f1_score(y_test_class, y_pred_class)

print('Classification Metrics:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# Save the model as a .pth file
torch.save(model.state_dict(), DICTIONARY_PATH)
print(f'Saved the model successfully as {DICTIONARY_PATH}')

torch.save(model, FULL_MODEL_PATH)
print(f'Saved the whole model successfully as {FULL_MODEL_PATH}')
