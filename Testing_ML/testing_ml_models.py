import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Path to the model and data
data_name = '5000_old'
ml_model = 'ML_Models/gru_model_5000.pth' 
json_file = '3_random_sample_5000_ml_corrected_endomondoHR.json'


# Define the GRU-based neural network
class HeartRateGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HeartRateGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)  # GRU forward pass
        out = self.fc(out[:, -1, :])  # Take the last time step for each sequence
        return out
    

# Model parameters (These should match what was used during training)
input_size = 15  # Number of features in each time step
hidden_size = 64
num_layers = 2
output_size = 1  # Predicting a single value (heart rate)

# Instantiate the model
model = HeartRateGRUModel(input_size, hidden_size, num_layers, output_size)

# Load the model's state dictionary
model.load_state_dict(torch.load(ml_model))

# Set the model to evaluation mode
model.eval()

# Load the data
df = pd.read_json(json_file, lines=True)
df.dropna(inplace=True)  # Drop rows with any NA values

def convert_to_list(x):
    if isinstance(x, list):
        return x  # Already a list
    elif isinstance(x, str):
        return list(map(float, x.strip('[]').split(',')))
    elif x is None:
        return []  # Return an empty list if None
    else:
        return [float(x)]  # Ensure it's a list of floats

df['speed'] = df['speed'].apply(convert_to_list)
df['altitude'] = df['altitude'].apply(convert_to_list)
df['heart_rate'] = df['heart_rate'].apply(convert_to_list)
df['timestamp'] = df['timestamp'].apply(convert_to_list)
df['longitude'] = df['longitude'].apply(convert_to_list)
df['latitude'] = df['latitude'].apply(convert_to_list)

for col in df.columns:
    if col.startswith('sport_'):
        df[col] = df[col].apply(convert_to_list)

df.drop(columns=['sport_orienteering', 'sport_cross-country skiing', 'sport_core stability training'], inplace=True)

# Drop any rows where list columns still contain empty lists after conversion
df = df[df['speed'].apply(len) > 0]
df = df[df['altitude'].apply(len) > 0]
df = df[df['heart_rate'].apply(len) > 0]
df = df[df['timestamp'].apply(len) > 0]
df = df[df['longitude'].apply(len) > 0]
df = df[df['latitude'].apply(len) > 0]

for col in df.columns:
    if col.startswith('sport_'):
        df = df[df[col].apply(len) > 0]

# Combine the features into a single tensor
X = df[['speed', 'altitude', 'timestamp', 'longitude', 'latitude'] + [col for col in df.columns if col.startswith('sport_')]]

# Ensure all elements are float
X = X.applymap(lambda x: [float(i) for i in x])

# Now create the numpy array
X = np.stack([np.array(X[col].tolist()) for col in X.columns], axis=-1)  # Shape: (batch_size, sequence_length, num_features)

y = np.array(df['heart_rate'].apply(lambda x: x[-1]).tolist())  # Shape: (batch_size,) - Only last heart rate value

print('1. Model loaded\n2. Data loaded, formatted and un-useful columns dropped\n')

# Test the model
# Convert data to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    y_pred = model(X)

# Convert predictions to numpy arrays for evaluation
y_pred = y_pred.numpy()
y = y.numpy()

# Evaluate the model
print('3. Evaluating the model\n')

# Save actual and predicted heart rates to CSV
results_df = pd.DataFrame({
    'Actual_Heart_Rate': y,
    'Predicted_Heart_Rate': y_pred.flatten()
})

results_csv_file = f'Metrics/{data_name}_predicted_vs_actual_heart_rate_.csv'
results_df.to_csv(results_csv_file, index=False)
print(f'Results saved to {results_csv_file}')

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
metrics_csv_file = f'Metrics/{data_name}_metrics_by_threshold.csv'
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(metrics_csv_file, index=False)
print(f'Metrics by threshold saved to {metrics_csv_file}')

# Print the metrics DataFrame for review
print(metrics_df)
