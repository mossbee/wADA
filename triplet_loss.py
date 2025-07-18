import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, distance_metric='euclidean'):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: 'euclidean' or 'cosine'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: Feature embeddings [batch_size, feature_dim]
        """
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            # Convert cosine similarity to distance (1 - similarity)
            pos_sim = F.cosine_similarity(anchor, positive)
            neg_sim = F.cosine_similarity(anchor, negative)
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean(), pos_dist.mean(), neg_dist.mean()

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.5, distance_metric='euclidean'):
        """
        Batch hard triplet loss - mines hardest positive and negative within batch
        """
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, feature_dim]
            labels: [batch_size] - person IDs
        """
        if self.distance_metric == 'euclidean':
            # Compute pairwise distances
            dist_matrix = self._euclidean_distance_matrix(embeddings)
        elif self.distance_metric == 'cosine':
            # Compute pairwise cosine distances (1 - similarity)
            dist_matrix = self._cosine_distance_matrix(embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Mine hardest positive and negative for each anchor
        loss = 0
        valid_triplets = 0
        
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # Find hardest positive (same label, maximum distance)
            pos_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
            if not pos_mask.any():
                continue
                
            hardest_pos_dist = dist_matrix[i][pos_mask].max()
            
            # Find hardest negative (different label, minimum distance)
            neg_mask = (labels != anchor_label)
            if not neg_mask.any():
                continue
                
            hardest_neg_dist = dist_matrix[i][neg_mask].min()
            
            # Compute triplet loss
            triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            loss += triplet_loss
            valid_triplets += 1
        
        if valid_triplets > 0:
            loss = loss / valid_triplets
        
        return loss
    
    def _euclidean_distance_matrix(self, embeddings):
        """Compute pairwise euclidean distances"""
        n = embeddings.size(0)
        dist_matrix = torch.zeros(n, n, device=embeddings.device)
        
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = F.pairwise_distance(
                    embeddings[i:i+1], embeddings[j:j+1], p=2
                )
        
        return dist_matrix
    
    def _cosine_distance_matrix(self, embeddings):
        """Compute pairwise cosine distances"""
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        return distance_matrix
