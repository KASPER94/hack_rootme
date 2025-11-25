import torch
import os

path = 'ch6/model2.pth'

try:
    # Try loading as a full model
    model = torch.load(path, map_location=torch.device('cpu'))
    print("Loaded as full model.")
    print(model)
except Exception as e:
    print(f"Not a full model: {e}")
    try:
        # Try loading as state_dict
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        print("Loaded as state_dict. Keys:")
        for key in state_dict.keys():
            print(key, state_dict[key].shape)
    except Exception as e2:
        print(f"Failed to load: {e2}")

