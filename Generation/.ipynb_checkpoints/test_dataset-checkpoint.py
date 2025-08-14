# test_dataset_with_images.py

import os
import random
import shutil  # Import the shutil library for robust file copying
import torch
from tqdm import tqdm

# Import your EEGDataset class from your main script.
from eegdatasets_leaveone import EEGDataset 

def run_dataset_sanity_check():
    """
    Initializes a multi-subject dataset, samples random indices, verifies the
    data alignment, and saves a copy of the retrieved image for visual inspection.
    """
    print("--- Initializing Multi-Subject Dataset for Testing ---")
    
    # --- 1. SETUP: Create a multi-subject dataset instance ---
    test_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
    try:
        dataset = EEGDataset(data_path="/ibex/user/thagafhh/data/Preprocessed_data_250Hz",
                             subjects=test_subjects, 
                             train=False)
    except Exception as e:
        print(f"\nERROR: Failed to initialize EEGDataset. Please check paths and __init__ method.")
        print(f"Error details: {e}")
        return

    # --- NEW: Create a directory to save the verification images ---
    output_image_dir = "dataset_verification_images"
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Verification images will be saved in the '{output_image_dir}/' directory.")
    
    print(f"\nDataset initialized successfully with {len(dataset)} total samples.")
    print("-" * 50)

    # --- 2. DEFINE KEY PARAMETERS ---
    n_cls = 1654
    samples_per_class = 10
    chunks_per_sample = 4
    samples_per_subject = n_cls * samples_per_class * chunks_per_sample

    # --- 3. THE TEST LOOP ---
    num_samples_to_test = 5
    print(f"Picking {num_samples_to_test} random indices to verify...\n")

    for i in range(num_samples_to_test):
        random_index = random.randint(0, len(dataset) - 1)
        
        print(f"--- Test Sample #{i+1}: Verifying index {random_index} ---")

        # --- 4. MANUALLY CALCULATE EXPECTED VALUES ---
        expected_subject_index = random_index // samples_per_subject
        expected_subject_id = test_subjects[expected_subject_index]
        expected_label = dataset.labels[random_index].item()
        
        expected_text_index = expected_label
        expected_text = dataset.text[expected_text_index]

        global_image_trial_index = random_index // chunks_per_sample
        image_offset_in_class = global_image_trial_index % samples_per_class
        expected_img_index = expected_label * samples_per_class + image_offset_in_class
        expected_img_path = dataset.img[expected_img_index]
        
        # Get the folder name for better context
        expected_folder_name = os.path.basename(os.path.dirname(expected_img_path))
        
        print(f"MANUAL CALCULATION:")
        print(f"  - Belongs to Subject: '{expected_subject_id}' (index {expected_subject_index})")
        print(f"  - Expected Label: {expected_label} (corresponds to folder '{expected_folder_name}')")
        print(f"  - Expected Text: '{expected_text}'")
        print(f"  - Expected Image Path: '.../{'/'.join(expected_img_path.split('/')[-2:])}'")

        # --- 5. GET ACTUAL VALUES FROM THE DATASET ---
        try:
            x, actual_label, actual_text, text_features, actual_img_path, img_features = dataset[random_index]
            actual_label = actual_label.item()
            
            print(f"\nDATASET OUTPUT:")
            print(f"  - Actual Label: {actual_label}")
            print(f"  - Actual Text: '{actual_text}'")
            print(f"  - Actual Image Path: '.../{'/'.join(actual_img_path.split('/')[-2:])}'")

            # --- NEW: Save a copy of the retrieved image ---
            try:
                # Create a descriptive filename for the saved image
                actual_folder_name = os.path.basename(os.path.dirname(actual_img_path))
                file_extension = os.path.splitext(actual_img_path)[1]
                
                new_filename = f"sample_{i+1}_idx_{random_index}_lbl_{actual_label}_{actual_folder_name}{file_extension}"
                destination_path = os.path.join(output_image_dir, new_filename)
                
                # Use shutil.copy for a robust file copy
                shutil.copy(actual_img_path, destination_path)
                
                print(f"  - ✅ Saved verification image to: '{destination_path}'")
            except Exception as e:
                print(f"  - ⚠️ WARNING: Could not save verification image. Error: {e}")

            # --- 6. VERIFY WITH ASSERTIONS ---
            assert actual_label == expected_label, "Label mismatch!"
            assert actual_text == expected_text, "Text mismatch!"
            assert actual_img_path == expected_img_path, "Image Path mismatch!"
            
            print("\n✅ PASS: All values match expectations.")

        except AssertionError as e:
            print(f"\n❌ FAIL: {e}")
        except Exception as e:
            print(f"\n❌ ERROR: An exception occurred while getting item {random_index}: {e}")
        
        print("-" * 50)

if __name__ == '__main__':
    run_dataset_sanity_check()