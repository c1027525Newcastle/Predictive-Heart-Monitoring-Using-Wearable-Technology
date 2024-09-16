import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import ast

# Load the data
df = pd.read_json('2_5000_corrected_endomondoHR.json', lines=True)
print('1 Data loaded\n')

# Removing the rows with missing values
df = df.dropna()
print('2 Missing values removed\n')

# Convert string representations of lists to actual lists
def convert_to_list_of_floats(x):
    try:
        x = ast.literal_eval(x)
        return list(map(float, x))
    except (ValueError, SyntaxError):
        return x

def convert_to_list_of_ints(x):
    try:
        x = ast.literal_eval(x)
        return list(map(int, x))
    except (ValueError, SyntaxError):
        return x

df['speed'] = df['speed'].apply(convert_to_list_of_floats)
df['heart_rate'] = df['heart_rate'].apply(convert_to_list_of_ints)
df['timestamp'] = df['timestamp'].apply(convert_to_list_of_ints)
df['longitude'] = df['longitude'].apply(convert_to_list_of_floats)
df['latitude'] = df['latitude'].apply(convert_to_list_of_floats)
df['altitude'] = df['altitude'].apply(convert_to_list_of_floats)

print('3 Columns converted to lists\n')

# Columns to average
array_columns = ['speed', 'altitude', 'timestamp', 'longitude', 'latitude', 'heart_rate']

# Function to average array columns
def average_array_columns(df, columns):
    for column in columns:
        df[column] = df[column].apply(np.mean)
    return df

# Apply the averaging
df = average_array_columns(df, array_columns)
print('4 Columns averaged\n')

# Drop non-useful columns (id, url, userId)
df = df.drop(columns=['id', 'url', 'userId'])
print('5 Non-useful columns dropped\n')

# Convert categorical columns to numerical and add new columns for each sub-category (One-Hot Encoding)
df = pd.get_dummies(df, columns=['gender', 'sport'])
print('6 Categorical columns converted to numerical\n')

# Define your features (X) and target (y)
X = df.drop(columns=['heart_rate'])
y = df['heart_rate']
print('7 Features and target defined\n')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('8 Data split into training and testing sets\n')

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define a simple neural network
class HeartRateModel(nn.Module):
    def __init__(self, input_size):
        super(HeartRateModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
print("INPUT SIZE:", input_size, "\n")
model = HeartRateModel(input_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
print('9 Training the model\n')
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
print('10 Making predictions\n')
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Convert to numpy arrays for evaluation
y_pred = y_pred.numpy()
y_test = y_test.numpy()

print('11 Calculating metrics\n')
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

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Save the model as a .pth file
torch.save(model.state_dict(), 'hr_model.pth')
print('Saved the model successfully as "hr_model.pth"')