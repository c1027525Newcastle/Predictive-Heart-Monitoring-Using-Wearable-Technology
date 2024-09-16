import torch

# Load the model's state dictionary
state_dict = torch.load('ML_Models_pth/gru_model_3_data5000_epoch250_sequence5_NomalisationTRUE.pth')

# Inspect the state dictionary
for key, value in state_dict.items():
    print(f'{key}: {value.shape}')
