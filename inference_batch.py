import net
import torch
import os
from face_alignment import align
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50', device=None):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    statedict = torch.load(adaface_models[architecture], weights_only=False, map_location=device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model = model.to(device)
    return model

def to_input(pil_rgb_image, device=None):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(bgr_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def to_input_batch(pil_rgb_images, device=None):
    """Convert a list of PIL images to a batched tensor"""
    tensors = []
    for pil_img in pil_rgb_images:
        if pil_img is not None:  # Handle alignment failures
            np_img = np.array(pil_img)
            bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
            tensor = torch.tensor(bgr_img, dtype=torch.float32).permute(2, 0, 1)
            tensors.append(tensor)
    
    if tensors:
        batch_tensor = torch.stack(tensors)
        if device is not None:
            batch_tensor = batch_tensor.to(device)
        return batch_tensor
    else:
        empty_tensor = torch.empty(0, 3, 112, 112)
        if device is not None:
            empty_tensor = empty_tensor.to(device)
        return empty_tensor

def get_feature_from_image(image_path, model, device=None):
    """
    Extract face feature from a single image
    
    Args:
        image_path (str): Path to the image file
        model: Loaded AdaFace model
        device: Device to run inference on
        
    Returns:
        torch.Tensor or None: Feature tensor of shape [1, 512] if successful, None if failed
    """
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    try:
        # Align face
        aligned_rgb_img = align.get_aligned_face(image_path, device=device)
        if aligned_rgb_img is None:
            print(f"Face alignment failed for {image_path}")
            return None
        
        # Convert to tensor
        bgr_tensor_input = to_input(aligned_rgb_img, device=device)
        
        # Extract feature
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        
        return feature
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_features_from_folder(folder_path, model, batch_size=8, max_workers=4, device=None):
    """
    Extract face features from all images in a folder using batch processing
    
    Args:
        folder_path (str): Path to folder containing images
        model: Loaded AdaFace model
        batch_size (int): Batch size for processing
        max_workers (int): Number of workers for parallel face alignment
        device: Device to run inference on
        
    Returns:
        tuple: (features, image_paths, valid_indices)
            - features: List of feature tensors [1, 512]
            - image_paths: List of all image paths processed
            - valid_indices: Indices of successfully processed images
    """
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for fname in sorted(os.listdir(folder_path)):
        if any(fname.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, fname))
    
    if not image_paths:
        print(f"No images found in {folder_path}")
        return [], [], []
    
    print(f"Found {len(image_paths)} images in {folder_path}")
    
    # Step 1: Align faces in parallel
    print("Aligning faces...")
    aligned_faces = [None] * len(image_paths)
    
    def align_single(idx_path):
        idx, path = idx_path
        try:
            return idx, align.get_aligned_face(path, device=device)
        except Exception as e:
            print(f"Alignment failed for {path}: {e}")
            return idx, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(align_single, (i, path)): i 
                  for i, path in enumerate(image_paths)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning faces"):
            idx, aligned_face = future.result()
            aligned_faces[idx] = aligned_face
    
    # Step 2: Filter out failed alignments
    valid_faces = []
    valid_indices = []
    for i, face in enumerate(aligned_faces):
        if face is not None:
            valid_faces.append(face)
            valid_indices.append(i)
    
    if not valid_faces:
        print("No faces could be aligned")
        return [], image_paths, []
    
    print(f"Successfully aligned {len(valid_faces)}/{len(image_paths)} faces")
    
    # Step 3: Process in batches for feature extraction
    print("Extracting features...")
    features = []
    
    for i in tqdm(range(0, len(valid_faces), batch_size), desc="Feature extraction"):
        batch_faces = valid_faces[i:i+batch_size]
        batch_tensor = to_input_batch(batch_faces, device=device)
        
        if batch_tensor.size(0) > 0:
            with torch.no_grad():
                batch_features, _ = model(batch_tensor)
                
                # Split batch features back to individual features
                for j in range(batch_features.size(0)):
                    features.append(batch_features[j:j+1])  # Keep [1, 512] shape
    
    return features, image_paths, valid_indices

def get_features_from_image_paths(image_paths, model, batch_size=8, max_workers=4, device=None):
    """
    Extract face features from a list of image paths using batch processing
    
    Args:
        image_paths (list): List of image file paths
        model: Loaded AdaFace model
        batch_size (int): Batch size for processing
        max_workers (int): Number of workers for parallel face alignment
        device: Device to run inference on
        
    Returns:
        tuple: (features, valid_indices)
            - features: List of feature tensors [1, 512]
            - valid_indices: Indices of successfully processed images
    """
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Step 1: Align faces in parallel
    print("Aligning faces...")
    aligned_faces = [None] * len(image_paths)
    
    def align_single(idx_path):
        idx, path = idx_path
        try:
            return idx, align.get_aligned_face(path, device=device)
        except Exception as e:
            print(f"Alignment failed for {path}: {e}")
            return idx, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(align_single, (i, path)): i 
                  for i, path in enumerate(image_paths)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning faces"):
            idx, aligned_face = future.result()
            aligned_faces[idx] = aligned_face
    
    # Step 2: Filter out failed alignments
    valid_faces = []
    valid_indices = []
    for i, face in enumerate(aligned_faces):
        if face is not None:
            valid_faces.append(face)
            valid_indices.append(i)
    
    if not valid_faces:
        print("No faces could be aligned")
        return [], []
    
    print(f"Successfully aligned {len(valid_faces)}/{len(image_paths)} faces")
    
    # Step 3: Process in batches for feature extraction
    print("Extracting features...")
    features = []
    
    for i in tqdm(range(0, len(valid_faces), batch_size), desc="Feature extraction"):
        batch_faces = valid_faces[i:i+batch_size]
        batch_tensor = to_input_batch(batch_faces, device=device)
        
        if batch_tensor.size(0) > 0:
            with torch.no_grad():
                batch_features, _ = model(batch_tensor)
                
                # Split batch features back to individual features
                for j in range(batch_features.size(0)):
                    features.append(batch_features[j:j+1])
    return features, valid_indices

if __name__ == '__main__':
    # Choose device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print("🔄 Loading AdaFace model...")
    model = load_pretrained_model('ir_50', device=device)
    
    # Example 1: Get feature from a single image
    print("\n📸 Example 1: Single Image Feature Extraction")
    single_image_path = 'test_images/90003d16.jpg'
    if os.path.exists(single_image_path):
        feature = get_feature_from_image(single_image_path, model, device=device)
        if feature is not None:
            print(f"✅ Feature extracted successfully! Shape: {feature.shape}")
            print(f"Feature: {feature.squeeze().cpu().numpy()[:10]}...")
        else:
            print("❌ Failed to extract feature")
    else:
        print(f"❌ Image not found: {single_image_path}")
    
    # Example 2: Batch processing from folder
    print("\n📁 Example 2: Batch Feature Extraction from Folder")
    test_folder = 'test_images'
    if os.path.exists(test_folder):
        features, image_paths, valid_indices = get_features_from_folder(
            test_folder, model, batch_size=4, max_workers=2, device=device
        )
        print(f"✅ Processed {len(features)} images successfully")
        print(f"Valid images: {len(valid_indices)}/{len(image_paths)}")
        if features:
            print(f"Feature shape: {features[0].shape}")
    else:
        print(f"❌ Folder not found: {test_folder}")
    
    # Example 3: Twin verification evaluation (if data exists)
    print("\n👥 Example 3: Twin Verification Evaluation")
    if os.path.exists('test_dataset_infor.json') and os.path.exists('test_twin_pairs.json'):
        print("📋 Dataset files found. You can run twin verification evaluation with:")
        print("python evaluate_twin_verification.py")
        print("  --dataset-info test_dataset_infor.json")
        print("  --twin-pairs test_twin_pairs.json")
        print("  --similarity cosine")
        print("  --batch-size 8")
    else:
        print("❌ Twin verification data not found. Please ensure you have:")
        print("  - test_dataset_infor.json (person_id -> image_paths mapping)")
        print("  - test_twin_pairs.json (twin pairs)")