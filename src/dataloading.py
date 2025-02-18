
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
import numpy as np
from PIL import Image
import os

from src.dataset import Stage1, Stage2
from src.models.n2n_unet import get_n2n_unet_model
from src.utils import get_images_from_datapath, fuse_images, get_informative_noise, group_pairs
from src.loss import Noise2VoidLoss

from src.dataset import PreprocessedFusedDataset

debug = False

def get_full_dataset(a=0, b=0, dynamic_background=True, background_threshold=90):
    
    full_paired_dataset = []
    originals_dataset = []
    fused_dataset = []

    for i in range(a, b): # for patients a to b

        p, diabetes = i, 2
        patient = f'RawDataQA ({p})'
        if diabetes != 0:
            patient = patient.replace(' ', f'-{diabetes} ')

        raw_images = get_patient_images(diabetes, patient)

        paired_images = fuse_images(raw_images, 10) # n neighbours

        full_paired_dataset.append(paired_images)

        fused_dataset.append(paired_images)

    constraint = []
    for paired_dataset in full_paired_dataset:
        originals, informative_noise = get_informative_noise(paired_dataset, dynamic_background, background_threshold)
        constraint.append(informative_noise)
        originals_dataset.append(originals)

    return constraint, originals_dataset, fused_dataset

def get_patient_images(diabetes, patient):

    datasetpath = rf'C:\Datasets\OCTData\ICIP training data\\{diabetes}\{patient}'

    images = get_images_from_datapath(datasetpath)

    loaded_images = [Image.open(image) for image in images]

    raw_images = [np.array(image) for image in loaded_images]

    return raw_images

def get_stage1_loaders(full_noisy_pairs, img_size):

    noisy_pairs = group_pairs(full_noisy_pairs)
        
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.tensor(img, dtype=torch.float32)), 
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    dataset_size = len(noisy_pairs)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    random.shuffle(noisy_pairs)

    train_data = noisy_pairs[:train_size]
    val_data = noisy_pairs[train_size:train_size + val_size]
    test_data = noisy_pairs[train_size + val_size:]

    train_dataset = Stage1(data=train_data, transform=transform)
    val_dataset = Stage1(data=val_data, transform=transform)
    test_dataset = Stage1(data=test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train size: {len(train_loader)}, Validation size: {len(val_loader)}, Test size: {len(test_loader)}")

    assert next(iter(train_loader))[0].shape == (1, 1, img_size, img_size), "First noisy image shape mismatch!"
    assert next(iter(train_loader))[1].shape == (1, 1, img_size, img_size), "Second noisy image shape mismatch!"

    return train_loader, val_loader, test_loader

def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_n2n_unet_model()
    model = model.to(device)

    criterion = Noise2VoidLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return device, model, criterion, optimizer
    

#########

def get_stage2_data(datapath = r'C:\temp\stage1'):
    groups = os.listdir(datapath)

    data = {}
    for group in groups:
        raw, denoised = [], []
        group_path = os.path.join(datapath, group)
        files = os.listdir(group_path)
        if debug: print(files)
        for file in files:
            file_path = os.path.join(group_path, file)
            for img in os.listdir(file_path):

                if debug: print(img)
                if debug: print(file_path)

                path = os.path.join(file_path, img)

                if 'raw' in file_path:
                    raw_img = Image.open(path)
                    raw.append(raw_img)
                elif 'constraint' in file_path:
                    denoised_img = Image.open(path) 
                    denoised.append(denoised_img)

        raw = [np.array(image) for image in raw]
        denoised = [np.array(image) for image in denoised]
        data[group] = {'raw': raw, 'constraint': denoised}
        if debug: print('---')

    return data

def get_stage_2_loaders_(img_size):

    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[:3]),  # Convert to CHW and keep RGB
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data = get_stage2_data()

    stage2_dataset = []
    for d in data.keys():
        raw_data = data[d]['raw']
        constraint_data = data[d]['constraint']
        stage2_dataset.append(Stage2((raw_data, constraint_data), transform=transform))

    stage2_dataset = torch.utils.data.ConcatDataset(stage2_dataset)

    train_size = int(0.7 * len(stage2_dataset))
    val_size = len(stage2_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(stage2_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader

def get_trained_unet1(img_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_n2n_unet_model()

    model = model.to(device)

    criterion = Noise2VoidLoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5)

    

    try:
        model.load_state_dict(torch.load(f'C:\\temp\\checkpoints\\stage1_{img_size}_last_{str(model)}_model.pth')['model_state_dict'])
        print('Model loaded')

    except:
        print('No model found')

    model.eval()

    if debug:
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return device, model, criterion, optimizer




#######

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


def get_preprocessed_data_loaders(base_path, patient_ids, img_size, batch_size=1):
    """Create DataLoaders for multiple patients (IDs 0-2)"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])

    all_datasets = []
    
    for patient_id in patient_ids:
        try:
            dataset = PreprocessedFusedDataset(
                base_path=base_path,
                patient_id=patient_id,
                transform=transform,
                compute_new_constraint=True
            )

            if len(dataset) == 0:
                raise ValueError(f"Dataset for Patient {patient_id} is empty")

            all_datasets.append(dataset)

        except Exception as e:
            print(f"Skipping Patient {patient_id} due to error: {str(e)}")

    if not all_datasets:
        raise RuntimeError("No valid patient datasets found!")

    # Merge datasets from all patients
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)

    # Split dataset
    train_size = int(0.7 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

    if debug:
        print(f"Train Samples: {train_size}, Validation Samples: {val_size}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader