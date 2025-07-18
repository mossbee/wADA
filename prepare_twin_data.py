#!/usr/bin/env python3
"""
Sample script to prepare data for twin verification evaluation.
This script shows how to create the required JSON files from your dataset.
"""

import json
import os
import argparse
from pathlib import Path

def create_sample_dataset_info(image_folder, output_file):
    """
    Create dataset info JSON from a folder structure like:
    image_folder/
        person1/
            img1.jpg
            img2.jpg
        person2/
            img1.jpg
            img2.jpg
        ...
    """
    dataset_info = {}
    
    if not os.path.exists(image_folder):
        print(f"❌ Image folder not found: {image_folder}")
        return
    
    for person_folder in sorted(os.listdir(image_folder)):
        person_path = os.path.join(image_folder, person_folder)
        
        if os.path.isdir(person_path):
            image_files = []
            for img_file in sorted(os.listdir(person_path)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                    image_files.append(os.path.join(person_path, img_file))
            
            if image_files:
                dataset_info[person_folder] = image_files
    
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Created dataset info with {len(dataset_info)} identities: {output_file}")
    return dataset_info

def create_sample_twin_pairs(dataset_info, output_file, manual_pairs=None):
    """
    Create twin pairs JSON. This is a manual process since twin relationships
    need to be defined based on your knowledge of the data.
    
    Args:
        dataset_info: Dictionary from create_sample_dataset_info
        output_file: Output JSON file path
        manual_pairs: List of [id1, id2] pairs where id1 and id2 are twins
    """
    
    if manual_pairs is None:
        # Example: If you have identities with naming pattern like "person1_twin1", "person1_twin2"
        # This is just an example - you'll need to define this based on your data
        
        twin_pairs = []
        
        # Example logic: Look for patterns in identity names
        all_ids = list(dataset_info.keys())
        
        # This is just a sample - you need to customize this logic
        for i, id1 in enumerate(all_ids):
            for j, id2 in enumerate(all_ids[i+1:], i+1):
                # Example: If IDs follow pattern like "family1_person1" and "family1_person2"
                if "_" in id1 and "_" in id2:
                    family1 = id1.split("_")[0]
                    family2 = id2.split("_")[0]
                    if family1 == family2:
                        twin_pairs.append([id1, id2])
        
        if not twin_pairs:
            # Create some example pairs for demonstration
            all_ids = list(dataset_info.keys())
            if len(all_ids) >= 4:
                twin_pairs = [
                    [all_ids[0], all_ids[1]],  # Assume first two are twins
                    [all_ids[2], all_ids[3]]   # Assume next two are twins
                ]
                print("⚠️  Created example twin pairs. Please manually edit the twin pairs file.")
    else:
        twin_pairs = manual_pairs
    
    with open(output_file, 'w') as f:
        json.dump(twin_pairs, f, indent=2)
    
    print(f"✅ Created twin pairs with {len(twin_pairs)} pairs: {output_file}")
    return twin_pairs

def create_flat_dataset_info(image_folder, output_file, images_per_person=None):
    """
    Create dataset info from a flat folder with naming convention like:
    person1_img1.jpg, person1_img2.jpg, person2_img1.jpg, etc.
    """
    dataset_info = {}
    
    if not os.path.exists(image_folder):
        print(f"❌ Image folder not found: {image_folder}")
        return
    
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            # Extract person ID from filename
            # Assuming format: personID_imageID.ext
            base_name = os.path.splitext(img_file)[0]
            
            if "_" in base_name:
                person_id = "_".join(base_name.split("_")[:-1])  # Everything except last part
            else:
                # If no underscore, use the base name without extension
                person_id = base_name
            
            if person_id not in dataset_info:
                dataset_info[person_id] = []
            
            dataset_info[person_id].append(os.path.join(image_folder, img_file))
    
    # Filter out persons with too few images
    if images_per_person:
        filtered_info = {k: v for k, v in dataset_info.items() if len(v) >= images_per_person}
        print(f"Filtered from {len(dataset_info)} to {len(filtered_info)} identities (min {images_per_person} images)")
        dataset_info = filtered_info
    
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Created dataset info with {len(dataset_info)} identities: {output_file}")
    return dataset_info

def validate_data_files(dataset_info_file, twin_pairs_file):
    """Validate the created data files"""
    
    # Load and validate dataset info
    with open(dataset_info_file, 'r') as f:
        dataset_info = json.load(f)
    
    print(f"📊 Dataset Statistics:")
    print(f"  Total identities: {len(dataset_info)}")
    
    total_images = sum(len(images) for images in dataset_info.values())
    avg_images = total_images / len(dataset_info) if dataset_info else 0
    print(f"  Total images: {total_images}")
    print(f"  Average images per identity: {avg_images:.1f}")
    
    # Check if image files exist
    missing_images = 0
    for person_id, image_paths in dataset_info.items():
        for img_path in image_paths:
            if not os.path.exists(img_path):
                missing_images += 1
    
    if missing_images > 0:
        print(f"⚠️  Warning: {missing_images} image files not found")
    else:
        print("✅ All image files exist")
    
    # Load and validate twin pairs
    with open(twin_pairs_file, 'r') as f:
        twin_pairs = json.load(f)
    
    print(f"👥 Twin Pairs Statistics:")
    print(f"  Total twin pairs: {len(twin_pairs)}")
    
    # Check if all twin IDs exist in dataset
    missing_ids = set()
    for pair in twin_pairs:
        for person_id in pair:
            if person_id not in dataset_info:
                missing_ids.add(person_id)
    
    if missing_ids:
        print(f"⚠️  Warning: Twin IDs not found in dataset: {missing_ids}")
    else:
        print("✅ All twin IDs found in dataset")
    
    return dataset_info, twin_pairs

def main():
    parser = argparse.ArgumentParser(description='Prepare data for twin verification evaluation')
    parser.add_argument('--image-folder', required=True,
                       help='Path to folder containing images')
    parser.add_argument('--dataset-output', default='dataset_info.json',
                       help='Output path for dataset info JSON')
    parser.add_argument('--twins-output', default='twin_pairs.json',
                       help='Output path for twin pairs JSON')
    parser.add_argument('--structure', choices=['nested', 'flat'], default='nested',
                       help='Image folder structure: nested (person_folders) or flat (person_id_in_filename)')
    parser.add_argument('--min-images', type=int, default=2,
                       help='Minimum images per person')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing files, do not create new ones')
    
    args = parser.parse_args()
    
    if args.validate_only:
        if os.path.exists(args.dataset_output) and os.path.exists(args.twins_output):
            validate_data_files(args.dataset_output, args.twins_output)
        else:
            print("❌ Cannot validate: One or both files don't exist")
        return
    
    print("🔄 Creating dataset files...")
    
    # Create dataset info
    if args.structure == 'nested':
        dataset_info = create_sample_dataset_info(args.image_folder, args.dataset_output)
    else:
        dataset_info = create_flat_dataset_info(args.image_folder, args.dataset_output, args.min_images)
    
    if not dataset_info:
        print("❌ Failed to create dataset info")
        return
    
    # Create twin pairs (you need to manually define these)
    print("\n⚠️  IMPORTANT: You need to manually define twin pairs!")
    print("The script will create a template, but you must edit it to define actual twin relationships.")
    
    twin_pairs = create_sample_twin_pairs(dataset_info, args.twins_output)
    
    # Validate the created files
    print("\n🔍 Validating created files...")
    validate_data_files(args.dataset_output, args.twins_output)
    
    print(f"\n✅ Data preparation complete!")
    print(f"📁 Files created:")
    print(f"  Dataset info: {args.dataset_output}")
    print(f"  Twin pairs: {args.twins_output}")
    print(f"\n🚀 You can now run the evaluation:")
    print(f"python evaluate_twin_verification.py \\")
    print(f"  --dataset-info {args.dataset_output} \\")
    print(f"  --twin-pairs {args.twins_output}")

if __name__ == '__main__':
    main()
