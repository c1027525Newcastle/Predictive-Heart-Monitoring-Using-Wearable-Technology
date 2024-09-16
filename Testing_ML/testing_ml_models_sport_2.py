import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Path to the model and data
model_name = 'gru_model_data7500_epoch500_full'
ml_model = f'ML_Models/gru_model_data7500_epcoh500_full.pth'
json_file = '4_random_sample_10000_ml_corrected_endomondoHR.json'


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
    

# Load the entire model (with architecture)
model = torch.load(ml_model)

# Set the model to evaluation mode
model.eval()

# Load the data
df = pd.read_json(json_file, lines=True)

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

# Determine the maximum sequence length
max_len = df['speed'].apply(len).max()

# Pad all sequences to the same length
def pad_sequences(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: x + [0] * (max_len - len(x)))  # Padding with zeros
    return df

# Apply padding
sequence_columns = ['speed', 'altitude', 'timestamp', 'longitude', 'latitude', 'heart_rate']
df = pad_sequences(df, sequence_columns)

# Combine the features into a single tensor
X = df[['speed', 'altitude', 'timestamp', 'longitude', 'latitude']]
X = np.stack([np.array(X[col].tolist()) for col in X.columns], axis=-1)  # Shape: (batch_size, sequence_length, num_features)
y = np.array(df['heart_rate'].apply(lambda x: x[-1]).tolist())  # Shape: (batch_size,) - Only last heart rate value
sport = df['sport'].values 

print('1. Model loaded\n2. Data loaded, formatted and un-useful columns dropped\n')

# Test the model
# Convert data to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
sport = torch.tensor(sport, dtype=torch.long)

# Make predictions
with torch.no_grad():
    y_pred = model(X, sport)

# Convert predictions to numpy arrays for evaluation
y_pred = y_pred.numpy()
y = y.numpy()

# Save actual and predicted heart rates to CSV
results_df = pd.DataFrame({
    'Actual_Heart_Rate': y,
    'Predicted_Heart_Rate': y_pred.flatten()
})

results_csv_file = f'Metrics/{model_name}_predicted_vs_actual_heart_rate_.csv'
results_df.to_csv(results_csv_file, index=False)
print(f'Results saved to {results_csv_file}')

# Evaluate the model
print('3. Evaluating the model\n')

# Calculate the Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Calculate the R² score
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.2f}\n')

# Experiment with different thresholds
thresholds = range(60, 140, 5)  # Adjust the range and step size as needed
metrics = []

for threshold in thresholds:
    y_pred_class = (y_pred > threshold).astype(int)
    y_class = (y > threshold).astype(int)

    accuracy = accuracy_score(y_class, y_pred_class)
    precision = precision_score(y_class, y_pred_class, zero_division=1)
    recall = recall_score(y_class, y_pred_class, zero_division=1)
    f1 = f1_score(y_class, y_pred_class, zero_division=1)

    metrics.append({
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Save the metrics to a CSV file
metrics_csv_file = f'Metrics/{model_name}_metrics_by_threshold.csv'
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(metrics_csv_file, index=False)
print(f'Metrics by threshold saved to {metrics_csv_file}')

# Print the metrics DataFrame for review
print(metrics_df)
