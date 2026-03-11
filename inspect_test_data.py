import torch
import os

data_path = os.path.join(os.path.dirname(__file__), "test", "dataset", "test_data.pt")
print(f"Checking file: {data_path}")

if not os.path.exists(data_path):
    print("File does not exist.")
else:
    try:
        data = torch.load(data_path)
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print("First item sample:")
                print(data[0])
                if isinstance(data[0], tuple):
                     print(f"Tuple structure: {len(data[0])} elements")
                     print(f"Element 0: {data[0][0]}")
                     print(f"Element 1: {data[0][1]}")
        elif isinstance(data, dict):
            print(f"Keys count: {len(data)}")
            first_key = next(iter(data))
            print(f"First key: {first_key}")
            print(f"First value: {data[first_key]}")
    except Exception as e:
        print(f"Error loading data: {e}")
