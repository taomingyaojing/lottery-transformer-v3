
import sys, os, json
sys.path.insert(0, '/home/ubuntu/workspace')
os.chdir('/home/ubuntu/workspace')

import numpy as np
import torch
import torch.nn.functional as F

from ensemble.predict import (
    load_data, predict_v3, predict_xgb, 
    NeighborhoodEnhancedTransformer, encode_draw,
    encode_neighborhood_mask, encode_neighborhood_distance
)

rows = load_data()
print(f"Data: {len(rows)} periods")

result = predict_v3(rows, 'models/v3/best_model.pth')
print(f"V3 result: {result}")

with open('predictions/prediction_v3_neighbor.json', 'w') as f:
    json.dump(result, f)
print("Saved!")
