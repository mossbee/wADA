import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import os
from inference_batch import load_pretrained_model, get_features_from_image_paths
import argparse

class TwinVerificationEvaluator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else next(model.parameters()).device
        
    def cosine_similarity(self, feat1, feat2):
        """Calculate cosine similarity between two feature vectors"""
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        return torch.sum(feat1_norm * feat2_norm, dim=1)
    
    def euclidean_distance(self, feat1, feat2):
        """Calculate euclidean distance between two feature vectors"""
        return torch.norm(feat1 - feat2, p=2, dim=1)
    
    def generate_test_pairs(self, dataset_info, twin_pairs, max_same_person_pairs=50):
        """
        Generate test pairs for verification
        
        Args:
            dataset_info: Dictionary with person_id -> list of image paths
            twin_pairs: List of twin pairs [[id1, twin_id1], [id2, twin_id2], ...]
            max_same_person_pairs: Maximum number of same-person pairs per identity
            
        Returns:
            positive_pairs: List of (img1_path, img2_path) for same person
            negative_pairs: List of (img1_path, img2_path) for different persons (twins)
        """
        positive_pairs = []  # Same person pairs
        negative_pairs = []  # Twin pairs (different persons)
        
        # Generate same person pairs (positive pairs)
        print("Generating same person pairs...")
        for person_id, image_paths in dataset_info.items():
            if len(image_paths) >= 2:
                # Generate all possible pairs for this person
                pairs = list(combinations(image_paths, 2))
                # Limit to max_same_person_pairs to balance dataset
                if len(pairs) > max_same_person_pairs:
                    pairs = pairs[:max_same_person_pairs]
                positive_pairs.extend(pairs)
        
        # Generate twin pairs (negative pairs - different persons)
        print("Generating twin pairs...")
        for twin_pair in twin_pairs:
            id1, id2 = twin_pair
            if id1 in dataset_info and id2 in dataset_info:
                images1 = dataset_info[id1]
                images2 = dataset_info[id2]
                
                # Generate all combinations between twins
                for img1 in images1:
                    for img2 in images2:
                        negative_pairs.append((img1, img2))
        
        print(f"Generated {len(positive_pairs)} positive pairs (same person)")
        print(f"Generated {len(negative_pairs)} negative pairs (twins)")
        
        return positive_pairs, negative_pairs
    
    def extract_features_for_pairs(self, pairs, batch_size=8, max_workers=4):
        """Extract features for all images in pairs"""
        # Collect all unique image paths
        all_image_paths = set()
        for img1, img2 in pairs:
            all_image_paths.add(img1)
            all_image_paths.add(img2)
        
        all_image_paths = list(all_image_paths)
        print(f"Extracting features for {len(all_image_paths)} unique images...")
        
        # Extract features
        features, valid_indices = get_features_from_image_paths(
            all_image_paths, self.model, batch_size=batch_size, 
            max_workers=max_workers, device=self.device
        )
        
        # Create mapping from image path to feature
        feature_dict = {}
        for i, valid_idx in enumerate(valid_indices):
            image_path = all_image_paths[valid_idx]
            feature_dict[image_path] = features[i]
        
        return feature_dict
    
    def compute_similarities(self, pairs, feature_dict, similarity_metric='cosine'):
        """Compute similarities for pairs"""
        similarities = []
        valid_pairs = []
        
        for img1, img2 in pairs:
            if img1 in feature_dict and img2 in feature_dict:
                feat1 = feature_dict[img1]
                feat2 = feature_dict[img2]
                
                if similarity_metric == 'cosine':
                    sim = self.cosine_similarity(feat1, feat2).item()
                elif similarity_metric == 'euclidean':
                    # Convert distance to similarity (higher = more similar)
                    dist = self.euclidean_distance(feat1, feat2).item()
                    sim = 1.0 / (1.0 + dist)  # Convert to similarity
                else:
                    raise ValueError(f"Unknown similarity metric: {similarity_metric}")
                
                similarities.append(sim)
                valid_pairs.append((img1, img2))
        
        return np.array(similarities), valid_pairs
    
    def find_optimal_threshold(self, positive_similarities, negative_similarities):
        """Find optimal threshold using multiple criteria"""
        all_similarities = np.concatenate([positive_similarities, negative_similarities])
        all_labels = np.concatenate([
            np.ones(len(positive_similarities)),
            np.zeros(len(negative_similarities))
        ])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
        
        # Find EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        # Find threshold that maximizes accuracy
        accuracies = []
        for threshold in thresholds:
            predictions = all_similarities >= threshold
            acc = accuracy_score(all_labels, predictions)
            accuracies.append(acc)
        
        best_acc_idx = np.argmax(accuracies)
        best_acc_threshold = thresholds[best_acc_idx]
        best_accuracy = accuracies[best_acc_idx]
        
        # Find threshold that maximizes F1 score
        f1_scores = []
        for threshold in thresholds:
            predictions = all_similarities >= threshold
            _, _, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')
            f1_scores.append(f1)
        
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx]
        best_f1_score = f1_scores[best_f1_idx]
        
        return {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'best_accuracy': best_accuracy,
            'best_acc_threshold': best_acc_threshold,
            'best_f1_score': best_f1_score,
            'best_f1_threshold': best_f1_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc(fpr, tpr)
        }
    
    def evaluate_at_threshold(self, positive_similarities, negative_similarities, threshold):
        """Evaluate performance at a specific threshold"""
        all_similarities = np.concatenate([positive_similarities, negative_similarities])
        all_labels = np.concatenate([
            np.ones(len(positive_similarities)),
            np.zeros(len(negative_similarities))
        ])
        
        predictions = all_similarities >= threshold
        
        # Basic metrics
        accuracy = accuracy_score(all_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')
        
        # Verification specific metrics
        true_positives = np.sum((predictions == 1) & (all_labels == 1))
        false_positives = np.sum((predictions == 1) & (all_labels == 0))
        true_negatives = np.sum((predictions == 0) & (all_labels == 0))
        false_negatives = np.sum((predictions == 0) & (all_labels == 1))
        
        # FAR: False Accept Rate (accepting different persons as same)
        far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # FRR: False Reject Rate (rejecting same persons as different)
        frr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        # TAR: True Accept Rate (correctly accepting same persons)
        tar = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'far': far,
            'frr': frr,
            'tar': tar,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    def plot_results(self, optimization_results, positive_similarities, negative_similarities, save_path=None):
        """Plot ROC curve and other visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        ax1.plot(optimization_results['fpr'], optimization_results['tpr'], 'b-', lw=2, 
                label=f'ROC curve (AUC = {optimization_results["auc"]:.4f})')
        ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Random classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        # EER point
        eer_idx = np.argmin(np.abs(optimization_results['fpr'] - (1 - optimization_results['tpr'])))
        ax1.plot(optimization_results['fpr'][eer_idx], optimization_results['tpr'][eer_idx], 
                'ro', markersize=8, label=f'EER = {optimization_results["eer"]:.4f}')
        
        # Threshold vs Accuracy
        accuracies = []
        for threshold in optimization_results['thresholds']:
            all_similarities = np.concatenate([positive_similarities, negative_similarities])
            all_labels = np.concatenate([
                np.ones(len(positive_similarities)),
                np.zeros(len(negative_similarities))
            ])
            predictions = all_similarities >= threshold
            acc = accuracy_score(all_labels, predictions)
            accuracies.append(acc)
        
        ax2.plot(optimization_results['thresholds'], accuracies, 'g-', lw=2)
        ax2.axvline(optimization_results['best_acc_threshold'], color='r', linestyle='--', 
                   label=f'Best threshold = {optimization_results["best_acc_threshold"]:.4f}')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Threshold vs Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Distribution of similarities
        ax3.hist(positive_similarities, bins=50, alpha=0.7, label='Same person pairs', color='green')
        ax3.hist(negative_similarities, bins=50, alpha=0.7, label='Twin pairs', color='red')
        ax3.axvline(optimization_results['eer_threshold'], color='blue', linestyle='--', 
                   label=f'EER threshold = {optimization_results["eer_threshold"]:.4f}')
        ax3.set_xlabel('Similarity Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Similarity Scores')
        ax3.legend()
        ax3.grid(True)
        
        # FAR vs FRR
        fars = []
        frrs = []
        for threshold in optimization_results['thresholds']:
            eval_result = self.evaluate_at_threshold(positive_similarities, negative_similarities, threshold)
            fars.append(eval_result['far'])
            frrs.append(eval_result['frr'])
        
        ax4.plot(optimization_results['thresholds'], fars, 'r-', lw=2, label='FAR')
        ax4.plot(optimization_results['thresholds'], frrs, 'b-', lw=2, label='FRR')
        ax4.axvline(optimization_results['eer_threshold'], color='green', linestyle='--', 
                   label=f'EER threshold = {optimization_results["eer_threshold"]:.4f}')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('FAR vs FRR')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def run_evaluation(self, dataset_info_path, twin_pairs_path, similarity_metric='cosine', 
                      batch_size=8, max_workers=4, max_same_person_pairs=50, save_results=True):
        """Run complete evaluation pipeline"""
        
        # Load data
        print("Loading dataset information...")
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            twin_pairs = json.load(f)
        
        # Generate test pairs
        positive_pairs, negative_pairs = self.generate_test_pairs(
            dataset_info, twin_pairs, max_same_person_pairs
        )
        
        # Extract features
        all_pairs = positive_pairs + negative_pairs
        feature_dict = self.extract_features_for_pairs(all_pairs, batch_size, max_workers)
        
        # Compute similarities
        print("Computing similarities for positive pairs...")
        positive_similarities, valid_positive_pairs = self.compute_similarities(
            positive_pairs, feature_dict, similarity_metric
        )
        
        print("Computing similarities for negative pairs...")
        negative_similarities, valid_negative_pairs = self.compute_similarities(
            negative_pairs, feature_dict, similarity_metric
        )
        
        print(f"Valid positive pairs: {len(positive_similarities)}")
        print(f"Valid negative pairs: {len(negative_similarities)}")
        
        # Find optimal thresholds
        print("Finding optimal thresholds...")
        optimization_results = self.find_optimal_threshold(positive_similarities, negative_similarities)
        
        # Evaluate at different thresholds
        thresholds_to_test = [
            optimization_results['eer_threshold'],
            optimization_results['best_acc_threshold'],
            optimization_results['best_f1_threshold']
        ]
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nDataset Statistics:")
        print(f"  Total identities: {len(dataset_info)}")
        print(f"  Twin pairs: {len(twin_pairs)}")
        print(f"  Valid positive pairs: {len(positive_similarities)}")
        print(f"  Valid negative pairs: {len(negative_similarities)}")
        print(f"  Similarity metric: {similarity_metric}")
        
        print(f"\nOverall Performance:")
        print(f"  AUC: {optimization_results['auc']:.4f}")
        print(f"  EER: {optimization_results['eer']:.4f}")
        
        print(f"\nOptimal Thresholds:")
        print(f"  EER threshold: {optimization_results['eer_threshold']:.4f}")
        print(f"  Best accuracy threshold: {optimization_results['best_acc_threshold']:.4f} (acc: {optimization_results['best_accuracy']:.4f})")
        print(f"  Best F1 threshold: {optimization_results['best_f1_threshold']:.4f} (f1: {optimization_results['best_f1_score']:.4f})")
        
        # Detailed evaluation at each threshold
        for i, threshold in enumerate(thresholds_to_test):
            threshold_names = ['EER', 'Best Accuracy', 'Best F1']
            eval_result = self.evaluate_at_threshold(positive_similarities, negative_similarities, threshold)
            
            print(f"\n{threshold_names[i]} Threshold ({threshold:.4f}) Performance:")
            print(f"  Accuracy: {eval_result['accuracy']:.4f}")
            print(f"  Precision: {eval_result['precision']:.4f}")
            print(f"  Recall: {eval_result['recall']:.4f}")
            print(f"  F1-Score: {eval_result['f1_score']:.4f}")
            print(f"  FAR (False Accept Rate): {eval_result['far']:.4f}")
            print(f"  FRR (False Reject Rate): {eval_result['frr']:.4f}")
            print(f"  TAR (True Accept Rate): {eval_result['tar']:.4f}")
        
        # Save results
        if save_results:
            results = {
                'optimization_results': optimization_results,
                'positive_similarities': positive_similarities.tolist(),
                'negative_similarities': negative_similarities.tolist(),
                'evaluation_at_thresholds': {}
            }
            
            for i, threshold in enumerate(thresholds_to_test):
                threshold_names = ['eer', 'best_accuracy', 'best_f1']
                eval_result = self.evaluate_at_threshold(positive_similarities, negative_similarities, threshold)
                results['evaluation_at_thresholds'][threshold_names[i]] = eval_result
            
            with open('twin_verification_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to twin_verification_results.json")
        
        # Plot results
        self.plot_results(optimization_results, positive_similarities, negative_similarities, 'twin_verification_plots.png')
        
        return optimization_results, positive_similarities, negative_similarities

def main():
    parser = argparse.ArgumentParser(description='Twin Face Verification Evaluation')
    parser.add_argument('--dataset-info', default='test_dataset_infor.json',
                       help='Path to dataset information JSON file')
    parser.add_argument('--twin-pairs', default='test_twin_pairs.json',
                       help='Path to twin pairs JSON file')
    parser.add_argument('--similarity', choices=['cosine', 'euclidean'], default='cosine',
                       help='Similarity metric to use')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for feature extraction')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of workers for parallel processing')
    parser.add_argument('--max-same-pairs', type=int, default=50,
                       help='Maximum same-person pairs per identity')
    parser.add_argument('--architecture', default='ir_50',
                       help='Model architecture')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print("🔄 Loading AdaFace model...")
    model = load_pretrained_model(args.architecture, device=device)
    
    # Create evaluator
    evaluator = TwinVerificationEvaluator(model, device=device)
    
    # Run evaluation
    evaluator.run_evaluation(
        args.dataset_info,
        args.twin_pairs,
        similarity_metric=args.similarity,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_same_person_pairs=args.max_same_pairs
    )

if __name__ == '__main__':
    main()
