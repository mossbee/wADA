import json
import random
from itertools import combinations
import torch
from torch.utils.data import Dataset
import numpy as np
from face_alignment import align

class TripletDataset(Dataset):
    def __init__(self, dataset_info_path, twin_pairs_path, mining_level='average', device=None):
        """
        Args:
            dataset_info_path: Path to train_dataset_infor.json
            twin_pairs_path: Path to train_twin_pairs.json
            mining_level: 'minimal' (25%), 'average' (50%), 'maximum' (100%)
            device: Device for face alignment
        """
        self.device = device or 'cpu'
        self.mining_level = mining_level
        
        # Load data
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            twin_pairs = json.load(f)
        
        # Build twin mapping
        self.twin_map = {}
        for id1, id2 in twin_pairs:
            if id1 in self.dataset_info:
                self.twin_map[id1] = id2
            if id2 in self.dataset_info:
                self.twin_map[id2] = id1
        
        # Generate triplets
        self.triplets = self._generate_triplets()
        print(f"Generated {len(self.triplets)} triplets with {mining_level} mining level")
    
    def _generate_triplets(self):
        """Generate triplets based on mining level"""
        triplets = []
        
        for person_id, image_paths in self.dataset_info.items():
            if person_id not in self.twin_map:
                continue
                
            twin_id = self.twin_map[person_id]
            if twin_id not in self.dataset_info:
                continue
            
            twin_images = self.dataset_info[twin_id]
            
            # Skip if person has only one image (can't form anchor-positive pairs)
            if len(image_paths) < 2:
                continue
            
            # Generate all possible anchor-positive pairs for this person
            anchor_positive_pairs = list(combinations(image_paths, 2))
            
            # Apply mining level sampling
            if self.mining_level == 'minimal':
                num_pairs = max(1, len(anchor_positive_pairs) // 4)
            elif self.mining_level == 'average':
                num_pairs = max(1, len(anchor_positive_pairs) // 2)
            else:  # maximum
                num_pairs = len(anchor_positive_pairs)
            
            sampled_pairs = random.sample(anchor_positive_pairs, num_pairs)
            
            # For each anchor-positive pair, create triplets with twin negatives
            for anchor_img, positive_img in sampled_pairs:
                for negative_img in twin_images:
                    triplets.append((anchor_img, positive_img, negative_img))
        
        # Shuffle triplets
        random.shuffle(triplets)
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        try:
            # Load and align faces
            anchor_face = align.get_aligned_face(anchor_path, device=self.device)
            positive_face = align.get_aligned_face(positive_path, device=self.device)
            negative_face = align.get_aligned_face(negative_path, device=self.device)
            
            # Convert to tensors
            anchor_tensor = self._to_tensor(anchor_face)
            positive_tensor = self._to_tensor(positive_face)
            negative_tensor = self._to_tensor(negative_face)
            
            return anchor_tensor, positive_tensor, negative_tensor
            
        except Exception as e:
            # Return dummy tensors if alignment fails
            print(f"Failed to load triplet {idx}: {e}")
            dummy = torch.zeros(3, 112, 112)
            return dummy, dummy, dummy
    
    def _to_tensor(self, pil_image):
        """Convert PIL image to tensor (same preprocessing as inference)"""
        if pil_image is None:
            return torch.zeros(3, 112, 112)
            
        np_img = np.array(pil_image)
        bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        return torch.tensor(bgr_img, dtype=torch.float32).permute(2, 0, 1)

def collate_triplets(batch):
    """Custom collate function for triplet batches"""
    anchors, positives, negatives = zip(*batch)
    
    anchor_batch = torch.stack(anchors)
    positive_batch = torch.stack(positives)
    negative_batch = torch.stack(negatives)
    
    return anchor_batch, positive_batch, negative_batch
