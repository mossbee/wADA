# Twin Face Verification Evaluation

This repository contains tools for evaluating face verification performance on twin pairs using the AdaFace model. It focuses on hard cases: distinguishing between twins (negative pairs) and recognizing the same person across different images (positive pairs).

## Files Overview

- `inference_batch.py` - Batch face feature extraction utilities
- `evaluate_twin_verification.py` - Main evaluation script with comprehensive metrics
- `prepare_twin_data.py` - Data preparation utilities
- `test_dataset_infor.json` - Sample dataset information file
- `test_twin_pairs.json` - Sample twin pairs file

## Quick Start

### 1. Prepare Your Data

You need two JSON files:

**Dataset Info (`test_dataset_infor.json`)**:
```json
{
    "person1": [
        "/path/to/person1/image1.jpg",
        "/path/to/person1/image2.jpg"
    ],
    "person2": [
        "/path/to/person2/image1.jpg",
        "/path/to/person2/image2.jpg"
    ]
}
```

**Twin Pairs (`test_twin_pairs.json`)**:
```json
[
    ["person1", "person3"],  // person1 and person3 are twins
    ["person2", "person4"]   // person2 and person4 are twins
]
```

### 2. Auto-generate Data Files (Optional)

If you have an organized image folder, use the preparation script:

```bash
# For nested folder structure (person_folders/images)
python prepare_twin_data.py --image-folder /path/to/images --structure nested

# For flat structure (person_id_image.jpg)
python prepare_twin_data.py --image-folder /path/to/images --structure flat
```

**Note**: You'll need to manually edit the twin pairs file to define actual twin relationships.

### 3. Run Evaluation

```bash
python evaluate_twin_verification.py \
    --dataset-info test_dataset_infor.json \
    --twin-pairs test_twin_pairs.json \
    --similarity cosine \
    --batch-size 8
```

## Evaluation Metrics

The evaluation provides comprehensive verification metrics:

### Core Metrics
- **Accuracy**: Overall correctness
- **Precision**: When predicting "same person", how often correct
- **Recall**: How many actual "same person" pairs were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### Verification-Specific Metrics
- **FAR (False Accept Rate)**: Rate of incorrectly accepting different persons as same
- **FRR (False Reject Rate)**: Rate of incorrectly rejecting same persons as different
- **TAR (True Accept Rate)**: Rate of correctly accepting same persons
- **EER (Equal Error Rate)**: Point where FAR = FRR
- **AUC**: Area Under ROC Curve

### Threshold Optimization

The script finds optimal thresholds using three criteria:
1. **EER Threshold**: Where FAR equals FRR
2. **Best Accuracy Threshold**: Maximizes overall accuracy
3. **Best F1 Threshold**: Maximizes F1-score

## Test Pairs Generation

The evaluation creates two types of test pairs:

### Positive Pairs (Same Person)
- Generated from multiple images of the same person
- Should result in high similarity scores
- Target: Accept as "same person"

### Negative Pairs (Different Persons - Twins)
- Generated between twin individuals
- Most challenging cases for face recognition
- Should result in low similarity scores
- Target: Reject as "different persons"

## Output

### Console Output
```
Dataset Statistics:
  Total identities: 50
  Twin pairs: 10
  Valid positive pairs: 245
  Valid negative pairs: 180

Overall Performance:
  AUC: 0.8743
  EER: 0.1234

Optimal Thresholds:
  EER threshold: 0.4567 
  Best accuracy threshold: 0.4890 (acc: 0.8456)
  Best F1 threshold: 0.4234 (f1: 0.8123)

EER Threshold (0.4567) Performance:
  Accuracy: 0.8456
  Precision: 0.8234
  Recall: 0.8678
  F1-Score: 0.8452
  FAR: 0.1234
  FRR: 0.1234
  TAR: 0.8766
```

### Generated Files
- `twin_verification_results.json`: Detailed numerical results
- `twin_verification_plots.png`: ROC curves and performance visualizations

### Plots Generated
1. **ROC Curve**: True Positive Rate vs False Positive Rate
2. **Threshold vs Accuracy**: Find optimal accuracy threshold
3. **Similarity Distribution**: Histogram of positive vs negative pair similarities
4. **FAR vs FRR**: Error rates across different thresholds

## Arguments

### evaluate_twin_verification.py
```bash
--dataset-info      Path to dataset info JSON (default: test_dataset_infor.json)
--twin-pairs        Path to twin pairs JSON (default: test_twin_pairs.json)
--similarity        Similarity metric: cosine|euclidean (default: cosine)
--batch-size        Batch size for feature extraction (default: 8)
--max-workers       Parallel workers for alignment (default: 4)
--max-same-pairs    Max same-person pairs per identity (default: 50)
--architecture      Model architecture (default: ir_50)
```

### prepare_twin_data.py
```bash
--image-folder      Path to images folder (required)
--dataset-output    Output dataset info JSON (default: dataset_info.json)
--twins-output      Output twin pairs JSON (default: twin_pairs.json)
--structure         Folder structure: nested|flat (default: nested)
--min-images        Minimum images per person (default: 2)
--validate-only     Only validate existing files
```

## Performance Interpretation

### Good Performance Indicators
- **High AUC** (>0.9): Model can distinguish twins well
- **Low EER** (<0.1): Balanced error rates
- **High TAR at low FAR**: Good genuine acceptance with low false acceptance

### What Makes This Challenging
- **Twins share genetic similarity**: Highest difficulty in face recognition
- **Intra-class variation**: Same person across different conditions
- **Real-world conditions**: Lighting, pose, expression variations

## Tips for Best Results

1. **Data Quality**: Use high-quality, aligned face images
2. **Balanced Dataset**: Similar number of images per person
3. **Diverse Conditions**: Include various poses, lighting, expressions
4. **Twin Definition**: Clearly define twin relationships
5. **Threshold Selection**: Choose based on your application needs:
   - **Security applications**: Lower FAR (higher threshold)
   - **User convenience**: Lower FRR (lower threshold)
   - **Balanced**: Use EER threshold

## Similarity Metrics

### Cosine Similarity (Recommended)
- Range: [-1, 1], higher = more similar
- Focuses on feature direction, invariant to magnitude
- Better for face verification tasks

### Euclidean Distance
- Converted to similarity: 1/(1+distance)
- Range: [0, 1], higher = more similar
- Sensitive to feature magnitude

## Example Use Cases

1. **Security Systems**: Optimize for low FAR to prevent unauthorized access
2. **Photo Organization**: Optimize for low FRR to group same person photos
3. **Research**: Use EER for balanced evaluation across different models
4. **Forensics**: Detailed analysis of challenging twin cases

## Troubleshooting

### Common Issues
- **No faces detected**: Check image quality and face alignment
- **Low performance**: Verify twin pair definitions
- **Memory errors**: Reduce batch size or max workers
- **File not found**: Check image paths in dataset info

### Performance Debugging
- Check similarity score distributions
- Analyze failed cases manually
- Verify data quality and labeling
- Compare different similarity metrics
