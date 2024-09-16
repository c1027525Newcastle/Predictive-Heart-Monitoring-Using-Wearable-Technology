import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim

data_number = 10000
# Load the data
df = pd.read_json(f'2_{data_number}_corrected_endomondoHR.json', lines=True)
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

# One-hot encode the 'sport' column
df = pd.get_dummies(df, columns=['sport'])
print('4. Sports column one-hot encoded\n')

# Expand the one-hot encoded 'sport' columns to match the sequence length
def expand_one_hot(df, max_len):
    sport_columns = [col for col in df.columns if col.startswith('sport_')]
    for column in sport_columns:
        df[column] = df[column].apply(lambda x: [x] * max_len)
    return df

# Determine the maximum sequence length
max_len = df['speed'].apply(len).max()

# Apply the expansion
df = expand_one_hot(df, max_len)

# Now pad all sequences to the same length
def pad_sequences(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: x + [0] * (max_len - len(x)))  # Padding with zeros
    return df

# Apply padding
sequence_columns = ['speed', 'altitude', 'timestamp', 'longitude', 'latitude', 'heart_rate'] + [col for col in df.columns if col.startswith('sport_')]
df = pad_sequences(df, sequence_columns)
print('5. Sequences padded\n')

# Combine the features into a single tensor
X = df[['speed', 'altitude', 'timestamp', 'longitude', 'latitude'] + [col for col in df.columns if col.startswith('sport_')]]
X = np.stack([np.array(X[col].tolist()) for col in X.columns], axis=-1)  # Shape: (batch_size, sequence_length, num_features)
y = np.array(df['heart_rate'].apply(lambda x: x[-1]).tolist())  # Shape: (batch_size,) - Only last heart rate value
print('6. Features and target combined\n')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('7. Data split into training and testing sets\n')

# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # Make y a column vector
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

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

# Model parameters
input_size = X_train.shape[2]  # Number of features in each time step
hidden_size = 64
num_layers = 2
output_size = 1  # Predicting a single value (heart rate)

# Instantiate the model, loss function, and optimizer
model = HeartRateGRUModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate of 0.001

# Training loop
num_epochs = 4000  # Number of epochs
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
print('8. Model trained\n')

# Make predictions
print('9. Making predictions\n')
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Converte to numpy arrays for evaluation
y_pred = y_pred.numpy()
y_test = y_test.numpy()

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
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)

print('Classification Metrics:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# Save the model as a .pth file
torch.save(model.state_dict(), f'ML_Models/gru_model_{data_number}.pth')
torch.save(model, f'ML_Models/gru_model_{data_number}_full.pth')
print(f'Saved the model successfully as "ML_Models/gru_model_{data_number}.pth"')
print(f'Saved the whole model successfully as "ML_Models/gru_model_{data_number}_full.pth"')