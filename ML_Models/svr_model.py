import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import torch #### NEW ####
import ast

# Load the data
df = pd.read_json('2_medium_corrected_endomondoHR.json', lines=True)
print('1 Data loaded\n')


# Removing the rows with missing values
df = df.dropna()
print('2 Missing values removed\n')


####
def convert_to_list_of_floats(x):
    try:
        # Convert string representation of list to an actual list
        x = ast.literal_eval(x)
        # Convert all elements to float
        return list(map(float, x))
    except (ValueError, SyntaxError):
        return x
    

def convert_to_list_of_ints(x):
    try:
        # Convert string representation of list to an actual list
        x = ast.literal_eval(x)
        # Convert all elements to int
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
####

# Columns to average
# TODO: Included heart here for now but i am not happy about it
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
# TODO: For now also drop gender? as it is probably unknown without asking the user also sport
df = df.drop(columns=['id', 'url', 'userId'])
print('5 Non-useful columns dropped\n')


# Convert categorical columns to numerical and ads new columns for each sub-category (One-Hot Encoding)
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

# Define the SVR model and the Radial Basis Function kernel
svr = SVR(kernel='rbf')

# Create the pipeline
pipeline = Pipeline([('scaler', scaler), ('svr', svr)])

# Train the model
print('9 Training the model\n')
pipeline.fit(X_train, y_train)


# Make predictions
print('10 Making predictions\n')
y_pred = pipeline.predict(X_test)


print('11 Calculating metrics\n')
# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Calculate the R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.2f}\n')


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

# To save the model as a pth file
torch.save(svr, 'svr_model.pth')
print('Saved the model successfully as "svr_model.pth"')