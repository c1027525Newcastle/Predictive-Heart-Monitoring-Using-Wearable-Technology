import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# Path to the model and data
data_name = 'gru_model_4_5000_250_5'
ml_model = 'ML_Models/gru_model_4_data7500_epoch2000_sequence5_NormalisationTRUE_full.pth' 
json_file = '4_random_sample_5000_ml_corrected_endomondoHR.json'
NORMALISATION = 'TRUE'

# Define the GRU-based neural network with embedding (from your training code)
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

# Correct the column name for cumulative distance (fixing typo)
df['cumulative_distance'] = df['cumulative_distance'].apply(convert_to_list)

# Normalize selected columns (if needed)
def normalize_column(column):
    scaler = MinMaxScaler()
    column_values = np.concatenate(df[column].values).reshape(-1, 1)  # Flatten the sequence column
    scaler.fit(column_values)
    
    # Apply normalization to each sequence in the column
    df[column] = df[column].apply(lambda x: scaler.transform(np.array(x).reshape(-1, 1)).flatten())
    
    return scaler

if NORMALISATION == 'TRUE':
    scalers = {}
    scalers['speed'] = normalize_column('speed')
    scalers['altitude'] = normalize_column('altitude')
    scalers['cumulative_distance'] = normalize_column('cumulative_distance')
    scalers['timestamp'] = normalize_column('timestamp')
    scalers['heart_rate'] = normalize_column('heart_rate')

# Combine the features into a single tensor
X = df[['speed', 'altitude', 'timestamp', 'cumulative_distance']]
X = np.stack([np.array(X[col].tolist()) for col in X.columns], axis=-1)  # Shape: (batch_size, sequence_length, num_features)

# Prepare the heart rate values
y = np.array(df['heart_rate'].apply(lambda x: x[-1]).tolist())  # Shape: (batch_size,) - Only last heart rate value

# # Define the model structure
# num_sports = df['sport'].nunique()
# input_size = X.shape[2]  # Number of features in each time step
# hidden_size = 64
# num_layers = 2
# output_size = 1  # Predicting a single value (heart rate)
# embedding_dim = 5

# model = HeartRateGRUModel(input_size, hidden_size, num_layers, output_size, num_sports, embedding_dim)
model = torch.load(ml_model)
# # Load the model's state dictionary
# model.load_state_dict(torch.load(ml_model, weights_only=True))#, strict=False) # strict=False allows loading partial state dict meaning if i have less sports all is gucci

# Set the model to evaluation mode
model.eval()

print('1. Model loaded\n2. Data loaded, formatted, and un-useful columns dropped\n')

# Test the model
# Convert data to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    y_pred = model(X)

# Convert predictions to numpy arrays for evaluation
y_pred = y_pred.numpy()

# Inverse-transform the heart rate if normalization was applied
if NORMALISATION == 'TRUE':
    y_pred = scalers['heart_rate'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y = scalers['heart_rate'].inverse_transform(y.reshape(-1, 1)).flatten()

# Evaluate the model
print('3. Evaluating the model\n')

# Save actual and predicted heart rates to CSV
results_df = pd.DataFrame({
    'Actual_Heart_Rate': np.round(y, 2),
    'Predicted_Heart_Rate': np.round(y_pred.flatten(), 2)
})

results_csv_file = f'Metrics/{data_name}_predicted_vs_actual_heart_rate.csv'
results_df.to_csv(results_csv_file, index=False)
print(f'Results saved to {results_csv_file}')

# Calculate the Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Calculate the R² score
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.2f}\n')

# Experiment with different thresholds
thresholds = range(60, 200, 10)  # Adjust the range and step size as needed
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
        'Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1 Score': round(f1, 2)
    })

# Save the metrics to a CSV file
metrics_csv_file = f'Metrics/{data_name}_metrics_by_threshold.csv'
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(metrics_csv_file, index=False)
print(f'Metrics by threshold saved to {metrics_csv_file}')

# Print the metrics DataFrame for review
print(metrics_df)
