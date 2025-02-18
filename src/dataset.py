

from src.utils import normalize_data
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms
import os
import re
from PIL import Image
from src.utils import compute_constraint
from tqdm import tqdm


debug = False

class Stage1(Dataset):
    def __init__(self, data, transform=None):
        self.data = data 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_group = self.data[idx]

        img1, img2 = random.sample(noisy_group, 2)

        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=0)  # Add channel dim if grayscale
        if len(img2.shape) == 2:
            img2 = np.expand_dims(img2, axis=0)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img1 = normalize_data(img1)
        img2 = normalize_data(img2)

        return img1, img2  # Return a tuple of two noisy images
    
class Stage2(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        raw_data = self.data[0]
        constraint_data = self.data[1]

        noisy_img = raw_data[idx]
        constraint_img = constraint_data[idx]

        if len(noisy_img.shape) == 2:
            noisy_img = np.expand_dims(noisy_img, axis=0)
        if len(constraint_img.shape) == 2:
            constraint_img = np.expand_dims(constraint_img, axis=0)

        if self.transform:
            noisy_img = self.transform(noisy_img)
            constraint_img = self.transform(constraint_img)

        noisy_img = normalize_data(noisy_img)
        constraint_img = normalize_data(constraint_img)

        return noisy_img, constraint_img  # Return a tuple of noisy and constraint images
    

##########

import torch
from torch.utils.data import Dataset
from src.utils import normalize_data


class PairedStage2Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.pairs = []
        self.transform = transform
        
        # Process each sequence
        for key in data.keys():
            raw_data = data[key]['raw']
            constraint_data = data[key]['constraint']
            
            # Create pairs within this sequence
            for i in range(len(raw_data) - 1):
                self.pairs.append((
                    (raw_data[i], constraint_data[i]),         # First pair
                    (raw_data[i + 1], constraint_data[i + 1])  # Second pair (next consecutive image)
                ))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        (raw1, constraint1), (raw2, constraint2) = self.pairs[idx]
        
        # Process first image pair
        if len(raw1.shape) == 2:
            raw1 = np.expand_dims(raw1, axis=0)
        if len(constraint1.shape) == 2:
            constraint1 = np.expand_dims(constraint1, axis=0)
        
        # Process second image pair
        if len(raw2.shape) == 2:
            raw2 = np.expand_dims(raw2, axis=0)
        if len(constraint2.shape) == 2:
            constraint2 = np.expand_dims(constraint2, axis=0)
        
        # Apply transforms if needed
        if self.transform:
            raw1 = self.transform(raw1)
            constraint1 = self.transform(constraint1)
            raw2 = self.transform(raw2)
            constraint2 = self.transform(constraint2)
        
        # Normalize
        raw1 = normalize_data(raw1)
        constraint1 = normalize_data(constraint1)
        raw2 = normalize_data(raw2)
        constraint2 = normalize_data(constraint2)
        
        # Return both pairs
        return {
            'first_pair': (raw1, constraint1),
            'second_pair': (raw2, constraint2)
        }


def get_stage_2_loaders(img_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[:3]),  # Convert to CHW and keep RGB
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data = get_stage2_data()
    
    # Create our paired dataset
    paired_dataset = PairedStage2Dataset(data, transform=transform)
    
    # Split into train and validation sets
    train_size = int(0.7 * len(paired_dataset))
    val_size = len(paired_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(paired_dataset, [train_size, val_size])
    
    # Define a custom collate function
    def custom_collate_fn(batch):
        first_raw_imgs = []
        first_constraint_imgs = []
        second_raw_imgs = []
        second_constraint_imgs = []
        
        for item in batch:
            first_raw, first_constraint = item['first_pair']
            second_raw, second_constraint = item['second_pair']
            
            first_raw_imgs.append(first_raw)
            first_constraint_imgs.append(first_constraint)
            second_raw_imgs.append(second_raw)
            second_constraint_imgs.append(second_constraint)
        
        return {
            'first_pair': (
                torch.stack(first_raw_imgs),
                torch.stack(first_constraint_imgs)
            ),
            'second_pair': (
                torch.stack(second_raw_imgs),
                torch.stack(second_constraint_imgs)
            )
        }
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

class PreprocessedFusedDataset(Dataset):
    def __init__(self, base_path, patient_id, transform=None, compute_new_constraint=True):
        self.transform = transform
        self.pairs = []
        self.compute_new_constraint = compute_new_constraint

        # Paths
        fused_path = os.path.join(base_path, str(patient_id), 'fused')
        raw_path = os.path.join(base_path, str(patient_id), 'raw')
        constraint_path = os.path.join(base_path, str(patient_id), 'constraint')

        if debug: print(f"Processing Patient: {patient_id}")
        if debug: print(f"Paths: Raw={raw_path}, Fused={fused_path}, Constraint={constraint_path}")

        # Get sorted filenames
        def extract_numbers(filename):
            return tuple(map(int, re.findall(r'\d+', filename)))

        fused_files = sorted([f for f in os.listdir(fused_path) if f.endswith('.png') and f != 'patient_level.png'], key=extract_numbers)
        raw_files = sorted([f for f in os.listdir(raw_path) if f.endswith('.png')], key=extract_numbers)
        constraint_files = sorted([f for f in os.listdir(constraint_path) if f.endswith('.png')], key=extract_numbers)

        if debug: 
            print(f"Found {len(fused_files)} fused images, {len(raw_files)} raw images, {len(constraint_files)} constraint images.")

        # Store file paths instead of loading images immediately
        for fused_file in tqdm(fused_files, desc="Indexing Fused Images"):
            base_idx = int(fused_file.split('_')[0])  # Extract base image index
            
            if base_idx < len(raw_files):  # Ensure index is within bounds
                self.pairs.append({
                    'raw_path': os.path.join(raw_path, raw_files[base_idx]),
                    'constraint_path': os.path.join(constraint_path, constraint_files[base_idx]),
                    'fused_path': os.path.join(fused_path, fused_file)
                })

        if debug: print(f"Total {len(self.pairs)} fused image pairs indexed.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            item = self.pairs[idx]
            
            # Load images directly as PIL images
            raw = Image.open(item['raw_path']).convert('L')
            original_constraint = Image.open(item['constraint_path']).convert('L')
            fused = Image.open(item['fused_path']).convert('L')

            # Apply transformations if specified
            if self.transform:
                raw = self.transform(raw)
                original_constraint = self.transform(original_constraint)
                fused = self.transform(fused)

            # Compute new constraint between raw and fused if requested
            if self.compute_new_constraint:
                new_constraint = compute_constraint(raw, fused)
                # Ensure new_constraint has the same shape and type as other tensors
                if torch.is_tensor(new_constraint):
                    new_constraint = new_constraint.float()
                    if new_constraint.dim() == 2:
                        new_constraint = new_constraint.unsqueeze(0)  # Add channel dimension
            else:
                new_constraint = original_constraint

            return {
                'raw': raw,
                'original_constraint': original_constraint,
                'new_constraint': new_constraint,
                'fused': fused,
                'metadata': {
                    'raw_path': item['raw_path'],
                    'fused_path': item['fused_path']
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {str(e)}")