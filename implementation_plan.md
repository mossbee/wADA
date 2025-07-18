# Implementation Plan for AdaFace Fine-tuning with Triplet Loss

## Overview
Fine-tune the entire AdaFace model using triplet loss with hard negatives from twin pairs, replacing the original loss function completely.

## Key Components Needed

### 1. **Triplet Dataset Class** (`triplet_dataset.py`)
- Load `train_dataset_infor.json` and `train_twin_pairs.json`
- Generate triplets: (anchor, positive, hard_negative)
- Data mining levels: minimal (25%), average (50%), maximum (100%)
- Handle single-image persons (only use as negatives)

### 2. **Triplet Loss Implementation** (`triplet_loss.py`)
- Standard triplet loss with margin
- Hard negative mining within batch (optional)
- Distance metrics: euclidean/cosine

### 3. **Training Script** (`train_triplet.py`)
- Load pretrained AdaFace model
- Fine-tune entire model with triplet loss
- Training loop with evaluation
- Save checkpoints

### 4. **Evaluation Integration**
- Use existing `evaluate_twin_verification.py` for testing
- Compare before/after fine-tuning performance

## Files to Keep/Modify/Remove

### **Keep & Modify:**
- `net.py` - Model architecture (keep as-is)
- `inference.py` - Basic inference (keep as-is) 
- `inference_batch.py` - Batch processing (keep as-is)
- `evaluate_twin_verification.py` - Evaluation (keep as-is)
- `face_alignment/` - Face alignment (keep as-is)

### **New Files:**
- `triplet_dataset.py` - Triplet data generation
- `triplet_loss.py` - Loss function
- `train_triplet.py` - Training script

### **Remove/Not Needed:**
- `main.py` - Original training script
- `train_val.py` - Original training/validation
- `head.py` - Classification head (not needed for triplet loss)
- `config.py` - Original config (create new simple config)
- `data.py` - Original data loading
- `utils.py` - Original utilities (if not used elsewhere)
- `dataset/` - Original dataset classes
- `scripts/` - Original training scripts

## Implementation Structure

```
AdaFace/
├── net.py                          # Keep - model architecture
├── inference.py                    # Keep - single inference
├── inference_batch.py              # Keep - batch inference  
├── evaluate_twin_verification.py   # Keep - evaluation
├── face_alignment/                 # Keep - face alignment
├── pretrained/                     # Keep - pretrained weights
├── triplet_dataset.py              # NEW - triplet data loader
├── triplet_loss.py                 # NEW - triplet loss
├── train_triplet.py                # NEW - training script
├── train_dataset_infor.json        # Input - training data
├── train_twin_pairs.json           # Input - twin pairs
├── test_dataset_infor.json         # Input - test data
└── test_twin_pairs.json            # Input - test twin pairs
```

## Key Design Decisions

1. **Triplet Generation Strategy:**
   - Anchor: Random image from person
   - Positive: Different image from same person
   - Hard Negative: Random image from twin of anchor person
   - Skip persons with only 1 image for anchor/positive roles

2. **Data Mining Levels:**
   - Minimal (25%): Sample 25% of possible triplets
   - Average (50%): Sample 50% of possible triplets  
   - Maximum (100%): Use all possible triplets

3. **Training Approach:**
   - Load pretrained AdaFace weights
   - Fine-tune entire model (no frozen layers)
   - Use triplet loss exclusively
   - Batch hard mining within each batch

4. **Evaluation:**
   - Use existing evaluation pipeline
   - Compare original vs fine-tuned model performance
   - Test on `test_dataset_infor.json` + `test_twin_pairs.json`

## Expected Workflow

1. **Prepare Data:** Generate triplets from training data
2. **Train:** Fine-tune model with triplet loss
3. **Evaluate:** Test on twin verification task
4. **Compare:** Original vs fine-tuned performance

This plan keeps the codebase minimal and focused, reusing existing evaluation infrastructure while adding only the necessary components for triplet-based fine-tuning.
