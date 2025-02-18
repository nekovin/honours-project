import torch
import re
import os

import re
import os
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil

def norm_to_target(image, target):
    image_std = torch.std(image)
    image_mean = torch.mean(image)

    target_mean = torch.mean(target)
    target_std = torch.std(target)

    standardised = (image - image_mean) / (image_std + 1e-6)

    normalized = standardised * target_std + target_mean
    
    return normalized

def normalize_data(data, target_min=0, target_max=1):
    """Normalize data to target range"""
    current_min = data.min()
    current_max = data.max()
    
    if current_min == current_max:
        return torch.ones_like(data) * target_min
    
    normalized = (data - current_min) / (current_max - current_min)
    normalized = normalized * (target_max - target_min) + target_min
    return normalized

def extract_number(filename):
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else -1

def get_images_from_datapath(dataset_path):
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'{dataset_path} not found')
    else:
        return sorted(
            [os.path.join(dataset_path, image) for image in os.listdir(dataset_path)],
            key=lambda x: extract_number(os.path.basename(x))
        )
    
def fuse_images(images, n_neighbours):
    fused_images = []
    pairs = []
    
    for i in range(n_neighbours, len(images) - n_neighbours):
        fused_images = mean_fusion(images, i, n_neighbours)
        pairs.append((images[i], fused_images))

    return pairs

def get_informative_noise(paired_images, dynamic, background_threshold):
    informative_noise = []
    originals = []
    for pair in paired_images:
        raw = pair[0]
        fused_samples = [img[1] for img in pair[1]]
        informative_noise.append(speckle_split_and_threshold(raw, fused_samples, dynamic, background_threshold))
        originals.append(raw)
    return originals, informative_noise

def mean_fusion(images, img_index, n):
    fused_images = []

    for j in range(1, n):

        left = max(0, img_index - j)
        right = min(len(images), img_index + j + 1)

        stacked_images = np.stack(images[left:right], axis=0)
        fused_image_array = np.mean(stacked_images, axis=0).astype(np.uint8)

        fused_images.append((j, fused_image_array))

    return fused_images

def speckle_split(oct1, oct2):
    return ((oct1 - oct2) ** 2) / (oct1 ** 2 + oct2 ** 2 + 1e-8)  # Adding epsilon to avoid division by zero

def dynamic_background_selection(oct_avg, percentile=50):
    
    row_intensity = np.mean(oct_avg, axis=1)  # Mean intensity per row
    col_intensity = np.mean(oct_avg, axis=0)  # Mean intensity per column

    row_threshold = np.percentile(row_intensity, percentile)
    col_threshold = np.percentile(col_intensity, percentile)

    row_indices = np.where(row_intensity <= row_threshold)[0]
    col_indices = np.where(col_intensity <= col_threshold)[0]

    min_rows = max(len(row_indices), int(0.05 * oct_avg.shape[0]))
    min_cols = max(len(col_indices), int(0.05 * oct_avg.shape[1]))

    background_region = oct_avg[row_indices[:min_rows], :][:, col_indices[:min_cols]]
    return background_region

def apply_threshold(octa, oct_avg, dynamic = True, background_threshold=50):
    if dynamic:
        background_region = dynamic_background_selection(oct_avg, background_threshold)
    else:
        background_region = oct_avg[:background_threshold, :]  # Assume background is in top 50 rows
    
    mean_bg = np.mean(background_region)
    std_bg = np.std(background_region)
    threshold = mean_bg + 2 * std_bg  # Threshold set at 2 std above mean background

    thresholded = np.where(oct_avg > threshold, octa, 0)  # Apply threshold
    return thresholded

def speckle_split_and_threshold(raw, fused_images, dynamic = True, background_threshold = 50):
    oct1 = raw.astype(np.float32)
    thresholded_results = []
    for fused_img in fused_images:
        
        oct2 = fused_img.astype(np.float32)
        
        octa = speckle_split(oct1, oct2)

        oct_avg = (oct1 + oct2) / 2
        thresholded = apply_threshold(octa, oct_avg, dynamic, background_threshold)
        thresholded_results.append(thresholded)
        

    return np.mean(thresholded_results, axis=0)

def group_pairs(noisy_sets):
    noisy_pairs = []
    for noisy_group in noisy_sets:
        num_images = len(noisy_group)

        if num_images < 2:
            continue 

        j = 0
        while j < num_images - 1: 
            noisy_pairs.append((noisy_group[j], noisy_group[j + 1]))
            j += 2 

    print(f"Pairs: {len(noisy_pairs)}")

    return noisy_pairs

def save_stage1_data(originals_dataset, constraint_dataset, model, device):
    assert len(originals_dataset[0]) == len(originals_dataset[0])

    group = 0

    for raw, noisy_pairs in zip(originals_dataset, constraint_dataset):

        if os.path.exists(f"C:\\temp\\stage1\\{group}"):
            shutil.rmtree(f"C:\\temp\\stage1\\{group}")

        if not os.path.exists(f"C:\\temp\\stage1\\{group}"):
            os.makedirs(f"C:\\temp\\stage1\\{group}")
            os.makedirs(f"C:\\temp\\stage1\\{group}\\constraint")
            os.makedirs(f"C:\\temp\\stage1\\{group}\\raw")

        final_stage_1_dataset = []

        final_stage_1_dataset.append(raw)
        constraint_imgs = list(map(lambda img: model(torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device))[0]
                            .squeeze().detach().cpu().numpy(), noisy_pairs))
        final_stage_1_dataset.append(constraint_imgs)
        
        raw, constraints = final_stage_1_dataset

        for i, (raw_img, con) in enumerate(zip(raw, constraints)):
            plt.imsave(f"C:\\temp\\stage1\\{group}\\constraint\\{i}.png", con, cmap='gray')
            plt.imsave(f"C:\\temp\\stage1\\{group}\\raw\\{i}.png", raw_img, cmap='gray')
        
        group += 1

def save_fused(fused_dataset):

    for patient in range(len(fused_dataset)):
        base_path = rf'C:\temp\stage1\{patient}\fused'
        os.makedirs(base_path, exist_ok=True) 

        for raw_idx, instance in enumerate(fused_dataset[patient]):  
            original = instance[0]  # The original image (not used in saving)
            fused_instances = instance[1]  # List of fused images at different levels

            # Iterate through each fused level
            for fused_idx, level_fused in enumerate(fused_instances):  
                level = level_fused[0]  # Level index (not needed for saving)
                fused_img = level_fused[1]  # The actual fused image

                # Define save path (e.g., "C:\temp\stage1\0\fused\0_1.png")
                save_path = os.path.join(base_path, f"{raw_idx}_{fused_idx + 1}.png")

                # Convert to PIL image and save
                fused_img_pil = Image.fromarray(np.uint8(fused_img))
                fused_img_pil.save(save_path)

                print(f"Saved: {save_path}")

        print(f"All fused images saved under: {base_path}")


def compute_constraint(original, fused):
    """
    Compute the constraint image between original and fused images
    Args:
        original: Original OCT image (torch.Tensor or numpy.ndarray)
        fused: Fused/denoised OCT image (torch.Tensor or numpy.ndarray)
    Returns:
        Constraint image (same type as input)
    """
    # Convert to numpy if tensors
    return_tensor = torch.is_tensor(original)
    if return_tensor:
        original = original.cpu().numpy()
        fused = fused.cpu().numpy()
    
    # Ensure proper shape
    if len(original.shape) == 4:  # [B, C, H, W]
        original = original.squeeze()
    if len(fused.shape) == 4:
        fused = fused.squeeze()
    
    # Convert to float32 for computation
    original = original.astype(np.float32)
    fused = fused.astype(np.float32)
    
    # Compute difference squared
    diff_squared = np.square(original - fused)
    
    # Compute sum squared
    sum_squared = np.square(original) + np.square(fused)
    
    # Compute constraint (similar to speckle split)
    constraint = diff_squared / (sum_squared + 1e-8)
    
    # Apply thresholding based on original image statistics
    background_threshold = 50
    background_region = original[:background_threshold, :]
    
    mean_bg = np.mean(background_region)
    std_bg = np.std(background_region)
    threshold = mean_bg + 2 * std_bg
    
    # Apply threshold
    thresholded_constraint = np.where(original > threshold, constraint, 0)
    
    # Convert back to tensor if input was tensor
    if return_tensor:
        thresholded_constraint = torch.from_numpy(thresholded_constraint)
        
    return thresholded_constraint
