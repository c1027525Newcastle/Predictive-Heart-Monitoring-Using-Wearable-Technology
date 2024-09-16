# Enhancing Heathcare for Older Adults: Predictive Heart Monitoring Using Wearable Technology
This repository was used during my (Luca Albinescu) internship at Newcastle University in researching how to enhance healthcare for older adults using wearable technology. This project aims to leverage wearable technology like smart watches to provide continuous heart rate monitoring for early detection of potential cardiac issues in older adults. This project tackles the growing challenge in the UK, where more elderly individuals are faced with having to live alone. This raises healthcare concerns, particularly around heart health, as sudden health crises can occur without immediate assistance.
The primary aim of this project is to develop a proactive system using smartwatches for real-time heart rate monitoring.
The objectives of my research include:
1. Finding the right dataset that contains enough metrics for heart-rate predictive analytics;
2. Training a machine learning model for accurate heart rate prediction using the metrics given by the wearable device;
3. Deploying this model on wearable devices.
This repositories will tackle the first 2 objectives.

# Usage
This software was developed and run on an Asus Rog Strix G G531, 2019, with the following specifications:
• OS: Ubuntu 20.04.6 LTS
• Processor (CPU): Intel® Core™ i7-9750H CPU @ 2.60GHz × 12
• Graphics (GPU): NV168 / Mesa Intel® UHD Graphics 630 (CFL GT2)
• Memory (RAM): 16GB
• Storage: 512GB SSD

## Installation
After cloning the git repository on the machine the following commands need to be run in the terminal:

- Create and activate virtual environment:
```
python3 -m venv venv 
```

```
source venv/bin/activate 
```

- Install required packages:
```
pip install -r requirements.txt
```

Before running anything in the project the [dataset](https://www.kaggle.com/datasets/pypiahmad/endomondo-fitness-trajectories?resource=download) should be first downloaded and placed in the directory of the project.
 
## Data Exploration
- The main file for exploring the Endomondo dataset is `data_exploration.ipynb`;
- This files offer a comprehensive analysis and explanation of the dataset, starting from how to access and format the file if need, for transforming it into a pandas dataframe, to creating graphs based on small samples from the dataset;
- One cell of the Jupyter Notebook offers a way to create a smaller sample from the huge Enndomondo dataset. This is needed if the current hardware can't handle working with such a big file.

## Machine Learning
- After exploring the dataset and creating some sample files we move to the next folder `ML_Models` where the ML model can be trained;
- Inside this folder there are several attepts at making the ML better performing, with the last attempt and best being `ML_Models/ml_gru_model_4.py`;
- All models are saved in the `ML_Models_pth` folder, as both full models and just state dictionaries, for later use in checking their performance or converting them to a `.pte` file using executorch.

## Visualize Results
- After training the model, some results of its performance like MSE and R^2 score are printed to the terminal but we need more information, so we move to the `Testing_ML` folder;
- In here there are several files again showing past approaches at seeing the performance of the ML;
- The most up to date are `Testing_ML/create_random_test_sample_improved.py` and `Testing_ML/testing_ml_models_4.py`;
- First `Testing_ML/create_random_test_sample_improved.py` should be run as this creates a random sample of records from the Endomondo dataset to test the ML on. The data will be put into the `Data_Samples` folder;
- Afterwards the `Testing_ML/testing_ml_models_4.py` can be run with the newly created sample and the chosen ML. 2 files will be created one `...threshold.csv` showing the accuracy, precision, recall and f1 score of the model at different thresholds and one `...predicted_vs_actual_heart_rate_.csv` that shows what the predicted values of the model were and what the actual values should havev been.

## Extra
- There is a `Plots_code` folder containing 3 python codes for doing graphs of different metrics from the dataset like correlation graph or a dual axis line chart of speed and heart rate over cummulative distance. All this graphs should go in the `Plots` folder
- `Testing_ML` folder has some extra scripts for things like checking the input size that the ML model takes (in case the sports category was exploded and you might have forgot how many inputs actually went in the model) or a file that prints all the sports inside a specific data file as to test the ML with the same sports categories that it was trained on.
- There is a `repair_json_file_to_actual_json.py` that converts 'improper' json files to a correct json format, most of the times this is not needed.
- The `ML_Models_pte` folder is there to store resulting `.pte` files from ExecuTorch.
