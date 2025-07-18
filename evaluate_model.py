# Evaluate a face recognition model on twin-face dataset
import json
from inference import 

test_dataset_infor_path = 'test_dataset_infor.json' # Dictionary of keys are twin_id and values are a list of image paths
test_twin_pairs_path = 'test_twin_pairs.json'

with open(test_dataset_infor_path, 'r') as f:
    test_dataset_infor = json.load(f)

with open(test_twin_pairs_path, 'r') as f:
    test_twin_pairs = json.load(f)

test_images_paths = []
for twin_id, image_paths in test_dataset_infor.items():
    test_images_paths.extend(image_paths)

# Generate image pairs from twin_pairs
for twin_pair in test_twin_pairs:
    twin_id_1 = twin_pair[0]
    twin_id_2 = twin_pair[1]

