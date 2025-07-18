import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime

# Import our modules
from triplet_dataset import TripletDataset, collate_triplets
from triplet_loss import TripletLoss
from inference_batch import load_pretrained_model
from evaluate_twin_verification import TwinVerificationEvaluator

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_pos_dist = 0
    total_neg_dist = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (anchors, positives, negatives) in enumerate(pbar):
        # Move to device
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Extract features
        anchor_features, _ = model(anchors)
        positive_features, _ = model(positives)
        negative_features, _ = model(negatives)
        
        # Compute loss
        loss, pos_dist, neg_dist = criterion(anchor_features, positive_features, negative_features)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_pos_dist += pos_dist.item()
        total_neg_dist += neg_dist.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Pos': f'{pos_dist.item():.4f}',
            'Neg': f'{neg_dist.item():.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'pos_dist': total_pos_dist / num_batches,
        'neg_dist': total_neg_dist / num_batches
    }

def evaluate_model(model, test_dataset_info, test_twin_pairs, device):
    """Evaluate model on twin verification task"""
    evaluator = TwinVerificationEvaluator(model, device=device)
    
    # Run evaluation (this will print results)
    optimization_results, pos_similarities, neg_similarities = evaluator.run_evaluation(
        test_dataset_info, test_twin_pairs, 
        similarity_metric='cosine', 
        batch_size=8, 
        max_workers=2,
        save_results=False
    )
    
    return optimization_results['auc'], optimization_results['eer']

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune AdaFace with Triplet Loss')
    parser.add_argument('--train-dataset', default='train_dataset_infor.json',
                       help='Training dataset info JSON')
    parser.add_argument('--train-twins', default='train_twin_pairs.json',
                       help='Training twin pairs JSON')
    parser.add_argument('--test-dataset', default='test_dataset_infor.json',
                       help='Test dataset info JSON')
    parser.add_argument('--test-twins', default='test_twin_pairs.json',
                       help='Test twin pairs JSON')
    parser.add_argument('--mining-level', choices=['minimal', 'average', 'maximum'], 
                       default='average', help='Data mining level')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='Triplet loss margin')
    parser.add_argument('--distance-metric', choices=['euclidean', 'cosine'], 
                       default='euclidean', help='Distance metric for triplet loss')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--save-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--eval-every', type=int, default=2,
                       help='Evaluate every N epochs')
    parser.add_argument('--architecture', default='ir_50',
                       help='Model architecture')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load pretrained model
    print("🔄 Loading pretrained AdaFace model...")
    model = load_pretrained_model(args.architecture, device=device)
    print("✅ Model loaded successfully")
    
    # Create dataset and dataloader
    print("📊 Creating triplet dataset...")
    train_dataset = TripletDataset(
        args.train_dataset, 
        args.train_twins,
        mining_level=args.mining_level,
        device=device
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with face alignment
        collate_fn=collate_triplets
    )
    
    # Create loss function and optimizer
    criterion = TripletLoss(margin=args.margin, distance_metric=args.distance_metric)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"🚀 Starting training...")
    print(f"  Dataset: {len(train_dataset)} triplets")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Mining level: {args.mining_level}")
    print(f"  Margin: {args.margin}")
    print(f"  Distance metric: {args.distance_metric}")
    
    # Training loop
    best_auc = 0
    training_log = []
    
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Avg Positive Distance: {train_metrics['pos_dist']:.4f}")
        print(f"Avg Negative Distance: {train_metrics['neg_dist']:.4f}")
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_every == 0:
            print("🔍 Evaluating on test set...")
            model.eval()
            with torch.no_grad():
                auc, eer = evaluate_model(model, args.test_dataset, args.test_twins, device)
            
            print(f"Test AUC: {auc:.4f}, EER: {eer:.4f}")
            
            # Save best model
            if auc > best_auc:
                best_auc = auc
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, train_metrics['loss'], save_path)
                print(f"🏆 New best AUC: {best_auc:.4f}")
        
        # Save training log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'pos_dist': train_metrics['pos_dist'],
            'neg_dist': train_metrics['neg_dist']
        }
        if (epoch + 1) % args.eval_every == 0:
            log_entry.update({'test_auc': auc, 'test_eer': eer})
        
        training_log.append(log_entry)
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], save_path)
    
    # Save final model and training log
    final_save_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs-1, train_metrics['loss'], final_save_path)
    
    log_save_path = os.path.join(args.save_dir, 'training_log.json')
    with open(log_save_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Models saved in: {args.save_dir}")

if __name__ == '__main__':
    main()
