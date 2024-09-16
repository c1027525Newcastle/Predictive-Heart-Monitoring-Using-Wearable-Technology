import pandas as pd
import numpy as np
import math
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# Choose the number of data points and epochs
DATA_NUM = 7500
EPOCH_NUM = 2000
NORMALISATION = 'TRUE'  # Choose 'TRUE' if normalisation is wanted
PATIENCE = 50  # Number of epochs to wait for improvement before stopping early

# Set the length of the sequences used as input to the model
SEQUENCE_LENGTH = 5

# Paths to save the model dictionary and full model
DICTIONARY_PATH = f'ML_Models/gru_model_4_data{DATA_NUM}_epoch{EPOCH_NUM}_sequence{SEQUENCE_LENGTH}_Nomalisation{NORMALISATION}.pth'
FULL_MODEL_PATH = f'ML_Models/gru_model_4_data{DATA_NUM}_epoch{EPOCH_NUM}_sequence{SEQUENCE_LENGTH}_Normalisation{NORMALISATION}_full.pth'

# Load and preprocess data (this part remains the same)
df = pd.read_json(f'Data_Samples/2_{DATA_NUM}_corrected_endomondoHR.json', lines=True)
df = df.drop(columns=['gender', 'id', 'userId', 'url', 'sport']).dropna()  # Drop unnecessary columns and rows with missing values

# Convert columns to lists
def convert_to_list(x):
    if isinstance(x, list):
        return x
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

# Convert lat/long to cumulative distance
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of Earth in km
    return c * r

def calculate_cumulative_distances(latitudes, longitudes):
    cumulative_distances = [0]
    for i in range(1, len(latitudes)):
        distance = haversine_distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
        cumulative_distances.append(cumulative_distances[-1] + distance)
    return cumulative_distances

df['cumulative_distance'] = df.apply(lambda row: calculate_cumulative_distances(row['latitude'], row['longitude']), axis=1)
df = df.drop(columns=['latitude', 'longitude'])

# Normalize the 'timestamp' column so that each list starts from 0
def normalize_timestamps(timestamps):
    return [t - timestamps[0] for t in timestamps] if len(timestamps) > 0 else timestamps

df['timestamp'] = df['timestamp'].apply(normalize_timestamps)

# Normalize columns
def normalize_column(column):
    scaler = MinMaxScaler()
    column_values = np.concatenate(df[column].values).reshape(-1, 1)
    scaler.fit(column_values)
    df[column] = df[column].apply(lambda x: scaler.transform(np.array(x).reshape(-1, 1)).flatten())
    return scaler

if NORMALISATION == 'TRUE':
    scalers = {}
    scalers['speed'] = normalize_column('speed')
    scalers['altitude'] = normalize_column('altitude')
    scalers['cumulative_distance'] = normalize_column('cumulative_distance')
    scalers['timestamp'] = normalize_column('timestamp')
    scalers['heart_rate'] = normalize_column('heart_rate')

# Function to split sequences into chunks of 5
def split_into_chunks(df, sequence_columns, chunk_size):
    new_rows = []
    for _, row in df.iterrows():
        min_length = min([len(row[col]) for col in sequence_columns])
        if min_length >= chunk_size:
            num_chunks = min_length // chunk_size
            for i in range(num_chunks):
                new_row = row.copy()
                for col in sequence_columns:
                    new_row[col] = row[col][i * chunk_size:(i + 1) * chunk_size]
                new_rows.append(new_row)
    return pd.DataFrame(new_rows)

# Split sequences into chunks
sequence_columns = ['speed', 'altitude', 'timestamp', 'cumulative_distance', 'heart_rate']
df = split_into_chunks(df, sequence_columns, SEQUENCE_LENGTH)

# Combine the features into a single tensor
X = df[['speed', 'altitude', 'timestamp', 'cumulative_distance']]
X = np.stack([np.array(X[col].tolist()) for col in X.columns], axis=-1)  # Shape: (batch_size, sequence_length, num_features)

# Prepare the target
y = np.array(df['heart_rate'].apply(lambda x: x[-1]).tolist())

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# Define the GRU-based neural network without sport feature
class HeartRateGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HeartRateGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_size = X_train.shape[2]  # Number of features in each time step (now without sport)
hidden_size = 64
num_layers = 2
output_size = 1  # Predicting a single value (heart rate)

# Instantiate the model, loss function, and optimizer
model = HeartRateGRUModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop with early stopping
for epoch in range(EPOCH_NUM):
    # Training phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f'Epoch {epoch+1}/{EPOCH_NUM}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the best model so far
        torch.save(model.state_dict(), DICTIONARY_PATH)
        torch.save(model, FULL_MODEL_PATH)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break

print('Training complete\n')

# Load the best model
model.load_state_dict(torch.load(DICTIONARY_PATH))

# Make predictions on the validation set
model.eval()
with torch.no_grad():
    y_pred = model(X_val)

# Evaluate the model
y_pred = y_pred.numpy()
if NORMALISATION == 'TRUE':
    y_pred = scalers['heart_rate'].inverse_transform(y_pred)
    y_val = scalers['heart_rate'].inverse_transform(y_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}\n')
