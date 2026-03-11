import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
split_path = os.path.join(PROJECT_ROOT, "test", "dataset", "crossdocked2020", "split_by_name.pt")
print(f"Checking split file: {split_path}")

try:
    data = torch.load(split_path)
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        for key in data:
            print(f"Key '{key}' length: {len(data[key])}")
            if len(data[key]) > 0:
                print(f"Sample item in '{key}': {data[key][0]}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"Sample: {data[0]}")
        
except Exception as e:
    print(f"Error loading split file: {e}")
