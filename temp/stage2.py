from PIL import Image
import numpy as np
import os

debug = False


import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm


    

def vis(data):
    print(len(data['0']['raw']))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(data['0']['raw'][5])
    ax[1].imshow(data['0']['denoised'][5])

def ex():
    
    sample = next(iter(train_loader))
    raw_sample, constraint_sample = sample[0], sample[1]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(raw_sample.squeeze().numpy(), cmap='gray')
    ax[1].imshow(constraint_sample.squeeze().numpy(), cmap='gray')

def usage():
    img1, img2, constraint1, constraint2, informative_noise1, informative_noise2 = oct_pairs[0][0], oct_pairs[0][1], constraint_pairs[0][0], constraint_pairs[0][1], informative_noise[0], informative_noise[1]

    informativeimg1 = torch.tensor(informative_noise1).float().unsqueeze(0).unsqueeze(0).to(device)
    informativeimg2 = torch.tensor(informative_noise2).float().unsqueeze(0).unsqueeze(0).to(device)

    #img = img.to(device)
    with torch.no_grad():
        #img = img.unsqueeze(0).unsqueeze(0)
        #informativeimg = informativeimg1.float()
        pred1, _ = model(informativeimg1)
        pred2, _ = model(informativeimg2)

    denoised_octa = torch.abs(pred1 - pred2)

    fig, ax = plt.subplots(1,5, figsize=(10,5))
    ax[0].imshow(img1.cpu().numpy(), cmap='gray')
    ax[1].imshow(informativeimg1[0][0].cpu().numpy(), cmap='gray')
    ax[2].imshow(pred1[0][0].cpu().numpy(), cmap='gray')
    ax[3].imshow(constraint1, cmap='gray')
    ax[4].imshow(denoised_octa[0][0].cpu().numpy(), cmap='gray')
    plt.show()

    alpha = 0.1  # Weight for the OCTA constraint term
    loss_octa = alpha * torch.norm(denoised_octa.to(device) - constraint1.to(device), p=2) ** 2
    loss_octa

    #loss_total = loss_pred1 + loss_pred2 + loss_octa

import numpy as np
import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from IPython.display import display, clear_output  # Enables dynamic updates


def speckle_split(oct1, oct2):
    """Calculate speckle split between two images"""
    if torch.is_tensor(oct1):
        oct1 = oct1.cpu().numpy()
    if torch.is_tensor(oct2):
        oct2 = oct2.cpu().numpy()
    return ((oct1 - oct2) ** 2) / (oct1 ** 2 + oct2 ** 2 + 1e-8)

def apply_threshold(octa, oct_avg, dynamic=False, background_threshold=50):
    """Apply thresholding based on background statistics"""
    if torch.is_tensor(octa):
        octa = octa.cpu().numpy()
    if torch.is_tensor(oct_avg):
        oct_avg = oct_avg.cpu().numpy()
    

    background_region = oct_avg[:background_threshold, :]
    
    mean_bg = np.mean(background_region)
    std_bg = np.std(background_region)
    threshold = mean_bg + 2 * std_bg
    
    thresholded = np.where(oct_avg > threshold, octa, 0)
    return thresholded

def get_informative_noise_from_two_images(img1, img2):
    """Get informative noise between two consecutive images"""
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Reshape if needed
    if len(img1.shape) == 1:
        h = int(np.sqrt(len(img1)))
        img1 = img1.reshape(h, h)
    if len(img2.shape) == 1:
        h = int(np.sqrt(len(img2)))
        img2 = img2.reshape(h, h)
    
    oct1 = img1.astype(np.float32)
    oct2 = img2.astype(np.float32)
    
    octa = speckle_split(oct1, oct2)
    
    oct_avg = (oct1 + oct2) / 2
    
    thresholded = apply_threshold(octa, oct_avg, dynamic=False, background_threshold=50)
    
    return thresholded

def get_stage_2_data():
    oct_pair = []
    constraint_pairs = []
    informative_noise_list = []
    
    for idx, sample in enumerate(train_loader):
        # Get the two consecutive images
        img1 = sample['first_pair'][0][0][0]
        img2 = sample['second_pair'][0][0][0]

        constraint1 = sample['first_pair'][1][0][0]
        constraint2 = sample['second_pair'][1][0][0]
        
        informative_noise = get_informative_noise_from_two_images(img1, img2)
        
        oct_pair.append((img1, img2))
        constraint_pairs.append((constraint1, constraint2))
        informative_noise_list.append(informative_noise)

    return oct_pair, constraint_pairs, informative_noise_list



def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_ssn2v_(ssn2v, train_loader, optimizer, num_epochs, visualize_every=50):
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    train_loss = []
    val_loss = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device once
            oct1 = batch['first_pair'][0].to(ssn2v.device)
            oct2 = batch['second_pair'][0].to(ssn2v.device)
            octa_constraint = batch['first_pair'][1].to(ssn2v.device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            with torch.cuda.amp.autocast():  # Mixed precision training
                loss, denoised_oct1, denoised_oct2 = ssn2v.train_step(oct1, oct2, octa_constraint[0])
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

            plot_loss(train_loss, val_loss)
            
            # Visualize periodically
            if batch_idx % visualize_every == 0:
                visualize_stage2(
                    oct1[0][0].cpu(),
                    oct2[0][0].cpu(),
                    octa_constraint[0][0].cpu(),
                    denoised_oct1.detach().cpu()[0][0],
                    denoised_oct2.detach().cpu()[0][0]
                )
                
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_epoch_loss)
        val_loss.append(0)
        print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}')

def visualise_n2v(img, img2, output):
    clear_output(wait=True)  # Ensures only the latest visualization is displayed
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))  # Fix incorrect unpacking
    ax[0].imshow(img.squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Input Noisy Image')
    ax[1].imshow(output.squeeze(), cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Output Image')
    ax[2].imshow(img2.squeeze(), cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Target Noisy Image')
    
    plt.show()

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

###########


import torch
import torch.nn.functional as F
from src.dataloading import get_trained_unet1
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import torch
from src.visualise import plot_loss
    

        



def check_tensor_stats(tensor, name=""):
    print(f"{name} - Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}")



    


@torch.no_grad()  # Disables gradient computation to save memory
def validate_ssn2v(ssn2v, val_loader):
    ssn2v.model.eval()  # Set model to evaluation mode
    val_loss = 0
    val_batches = 0
    
    for batch in val_loader:
        oct1 = batch['raw'].to(ssn2v.device)
        oct2 = batch['fused'].to(ssn2v.device)
        octa_constraint = batch['original_constraint'].to(ssn2v.device)
        octb_constraint = batch['new_constraint'].to(ssn2v.device)

        loss, _, _ = ssn2v.train_step(oct1, oct2, octa_constraint, octb_constraint)  # Get loss
        val_loss += loss.item()
        val_batches += 1

    ssn2v.model.train()  # Set back to training mode after validation
    return val_loss / val_batches if val_batches > 0 else 0