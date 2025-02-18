import torch
import torch.nn as nn
from src.visualise import visualise_n2v, plot_loss
import os
import numpy as np
import matplotlib.pyplot as plt
from src.visualise import visualise_n2v, plot_loss, visualize_stage2
import torch.nn.functional as F
from tqdm import tqdm

def train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda', scratch=False):

    if not os.path.exists('C:\\temp\\checkpoints'):
        os.makedirs('C:\\temp\\checkpoints')

    if scratch == True:
        model = model
        history = {'train_loss': [], 'val_loss': []}
        old_epoch = 0
        print("Training from scratch")
    else:
        try:
            checkpoint = torch.load(f'C:\\temp\\checkpoints\\stage1_{img_size}_best_{str(model)}_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            old_epoch = epoch
            avg_train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {val_loss:.6f} from epoch {epoch+1}")
        except:
            print("No model found, training from scratch")
            model = model
            history = {'train_loss': [], 'val_loss': []}
            old_epoch = 0

    model = model.to(device)
    best_val_loss = float('inf')
    vis = False
    
    
    for epoch in range(epochs):
        # In train_stage1 function, add after the epoch loop starts:
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (noisy_img1, noisy_img2) in enumerate(train_loader):
            noisy_img1 = noisy_img1.to(device)
            noisy_img2 = noisy_img2.to(device)

            #print(f"Input range: {noisy_img1.min().item():.4f} to {noisy_img1.max().item():.4f}")
            #print(f"Target range: {noisy_img2.min().item():.4f} to {noisy_img2.max().item():.4f}")
            
            optimizer.zero_grad()
            
            outputs, features = model(noisy_img1) # res unet
            #outputs = norm_to_target(outputs, noisy_img2)
            
            # Calculate loss
            loss = criterion(outputs, noisy_img2)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if vis == True:
                if (batch_idx + 1) % 50 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                        f'Loss: {loss.item():.6f}')
        
        avg_train_loss = running_loss / len(train_loader)
        
        val_loss = validate_n2v(model, val_loader, criterion, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        plot_loss(history['train_loss'], history['val_loss'])
        
        # Save best model
        save = True
        if val_loss < best_val_loss and save == True:
            print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch+1}")
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + old_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, f'C:\\temp\\checkpoints\\stage1_{img_size}_best_{str(model)}_model.pth')
            
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print("-" * 50)

        if save:
            torch.save({
                'epoch': epoch + old_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, f'C:\\temp\\checkpoints\\stage1_{img_size}_last_{str(model)}_model.pth')

    return model, history

def validate_n2v(model, val_loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for noisy_img1, noisy_img2 in val_loader:
            noisy_img1 = noisy_img1.to(device)
            noisy_img2 = noisy_img2.to(device)
            
            outputs, features = model(noisy_img1)
            #outputs = norm_to_target(outputs, noisy_img2)
            loss = criterion(outputs, noisy_img2)
            
            total_loss += loss.item()
            
            if total_loss == loss.item():  # First iteration
                visualise_n2v(
                    noisy_img1.cpu().detach().numpy(),
                    noisy_img2.cpu().detach().numpy(),
                    outputs.cpu().detach().numpy()
                )
    
    return total_loss / len(val_loader)


def train_stage1_patched(img_size, model, train_loader, val_loader, criterion, optimizer, 
                     patch_size=128, stride=96, epochs=10, device='cuda', scratch=False):
    """
    Patch-based training for Stage 1
    Args:
        patch_size: Size of patches to extract (default 128)
        stride: Stride between patches (default 96 for 32px overlap)
    """
    if not os.path.exists('C:\\temp\\checkpoints'):
        os.makedirs('C:\\temp\\checkpoints')

    # Load checkpoint or start from scratch
    if scratch:
        model = model
        history = {'train_loss': [], 'val_loss': []}
        old_epoch = 0
        print("Training from scratch")
    else:
        try:
            checkpoint = torch.load(f'C:\\temp\\checkpoints\\stage1_{img_size}_best_{str(model)}_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            old_epoch = epoch
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {epoch+1}")
        except:
            print("No model found, training from scratch")
            model = model
            history = {'train_loss': [], 'val_loss': []}
            old_epoch = 0

    model = model.to(device)
    best_val_loss = float('inf')
    vis = False
    
    # Create blending weights once
    blend_weights = create_blending_weights(patch_size).to(device)
    
    for epoch in range(epochs):
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (noisy_img1, noisy_img2) in enumerate(train_loader):
            noisy_img1 = noisy_img1.to(device)
            noisy_img2 = noisy_img2.to(device)
            
            # Extract patches
            patches1, positions, _ = patch_processor(
                noisy_img1, 
                patch_size=patch_size, 
                stride=stride,
                device=device
            )
            patches2 = patch_processor(noisy_img2, patch_size, stride, device)[0]
            
            # Process patches in batch
            optimizer.zero_grad()
            
            batch_loss = 0
            processed_patches = []
            
            # Process patches in smaller sub-batches to save memory
            sub_batch_size = 4
            for i in range(0, len(patches1), sub_batch_size):
                sub_patches1 = patches1[i:i + sub_batch_size]
                sub_patches2 = patches2[i:i + sub_batch_size]
                
                outputs, _ = model(sub_patches1)
                loss = criterion(outputs, sub_patches2)
                
                # Accumulate loss and gradients
                loss = loss / ((len(patches1) + sub_batch_size - 1) // sub_batch_size)
                loss.backward()
                
                batch_loss += loss.item() * sub_batch_size
                processed_patches.append(outputs.detach())
            
            # Optimizer step after processing all patches
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += batch_loss
            
            # Visualize if needed
            if vis and (batch_idx + 1) % 50 == 0:
                # Reconstruct full image from patches for visualization
                processed_patches = torch.cat(processed_patches, dim=0)
                reconstructed = reconstruct_from_patches(
                    processed_patches,
                    positions,
                    noisy_img1.shape,
                    blend_weights
                )
                
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {batch_loss:.6f}')
                
                # Add visualization code here if needed
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validate with patches
        val_loss = validate_n2v_patched(model, val_loader, criterion, patch_size, stride, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        plot_loss(history['train_loss'], history['val_loss'])
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch+1}")
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + old_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, f'C:\\temp\\checkpoints\\stage1_{img_size}_best_{str(model)}_model.pth')
            
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print("-" * 50)

        # Save last model
        torch.save({
            'epoch': epoch + old_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'history': history
        }, f'C:\\temp\\checkpoints\\stage1_{img_size}_last_{str(model)}_model.pth')

    return model, history

@torch.no_grad()
def validate_n2v_patched(model, val_loader, criterion, patch_size=128, stride=96, device='cuda'):
    """Patched validation function"""
    model.eval()
    total_loss = 0.0
    blend_weights = create_blending_weights(patch_size).to(device)
    
    for noisy_img1, noisy_img2 in val_loader:
        noisy_img1 = noisy_img1.to(device)
        noisy_img2 = noisy_img2.to(device)
        
        # Extract patches
        patches1, positions, _ = patch_processor(
            noisy_img1, 
            patch_size=patch_size, 
            stride=stride,
            device=device
        )
        patches2 = patch_processor(noisy_img2, patch_size, stride, device)[0]
        
        batch_loss = 0
        processed_patches = []
        
        # Process patches
        for p1, p2 in zip(patches1, patches2):
            output, _ = model(p1.unsqueeze(0))
            loss = criterion(output, p2.unsqueeze(0))
            batch_loss += loss.item()
            processed_patches.append(output)
            
        total_loss += batch_loss / len(patches1)
        
        # Visualize first batch
        if total_loss == batch_loss / len(patches1):
            processed_patches = torch.cat(processed_patches, dim=0)
            reconstructed = reconstruct_from_patches(
                processed_patches,
                positions,
                noisy_img1.shape,
                blend_weights
            )
            
            visualise_n2v(
                noisy_img1.cpu().numpy(),
                noisy_img2.cpu().numpy(),
                reconstructed.cpu().numpy()
            )
    
    return total_loss / len(val_loader)


#####################

def train_ssn2v_improved(img_size, ssn2v, train_loader, val_loader, optimizer, num_epochs, device, visualize_every=50, percentile=50):
    """
    Improved training loop for SSN2V with better loss tracking and visualization
    """
    scaler = torch.cuda.amp.GradScaler()
    train_losses = {'total': [], 'n2v': [], 'octa': [], 'background': []}
    val_losses = {'total': [], 'n2v': [], 'octa': [], 'background': []}
    
    # Learning rate scheduler with reduced patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,  # Reduced patience for faster adaptation
        verbose=True
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 10  # Reduced patience for early stopping
    
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        ssn2v.model.train()
        epoch_losses = {'total': 0, 'n2v': 0, 'octa': 0, 'background': 0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Load and normalize data
            oct1 = batch['raw'].to(device)
            oct2 = batch['fused'].to(device)
            octa_constraint = batch['original_constraint'].to(device)
            octb_constraint = batch['new_constraint'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with automatic mixed precision
            with torch.cuda.amp.autocast():
                loss_dict = ssn2v.train_step(oct1, oct2, octa_constraint, octb_constraint, percentile)
                
            # Gradient scaling and clipping
            scaler.scale(loss_dict['total_loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ssn2v.model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update loss tracking
            for key in epoch_losses.keys():
                if key == 'total':
                    epoch_losses[key] += loss_dict['total_loss'].item()
                else:
                    epoch_losses[key] += loss_dict[f'{key}_loss'].item()
            num_batches += 1
            
            # Visualization and logging
            if batch_idx % visualize_every == 0:
                with torch.no_grad():
                    visualize_stage2(
                        oct1[0][0].cpu(),
                        oct2[0][0].cpu(),
                        octa_constraint[0][0].cpu(),
                        octb_constraint[0][0].cpu(),
                        loss_dict['denoised_oct1'].detach().cpu()[0][0],
                        loss_dict['denoised_oct2'].detach().cpu()[0][0]
                    )
                    print(f"Batch {batch_idx} Losses:")
                    print(f"Total: {loss_dict['total_loss'].item():.4f}")
                    print(f"N2V: {loss_dict['n2v_loss'].item():.4f}")
                    print(f"OCTA: {loss_dict['octa_loss'].item():.4f}")
                    print(f"Background: {loss_dict['background_loss'].item():.4f}")
        
        # Compute average losses for epoch
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            train_losses[k].append(v)
        
        # Validation phase
        val_avg_losses = validate_ssn2v_improved(ssn2v, val_loader, device, percentile)
        for k, v in val_avg_losses.items():
            val_losses[k].append(v)
        
        # Learning rate scheduling based on total validation loss
        scheduler.step(val_avg_losses['total'])
        
        # Early stopping check
        if val_avg_losses['total'] < best_loss:
            best_loss = val_avg_losses['total']
            patience_counter = 0
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': ssn2v.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, f'C:\\temp\\checkpoints\\stage2_{img_size}_last_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print("Training Losses:")
        for k, v in avg_losses.items():
            print(f"{k.capitalize()}: {v:.4f}")
        print("\nValidation Losses:")
        for k, v in val_avg_losses.items():
            print(f"{k.capitalize()}: {v:.4f}")
        print("-" * 50)
    
    return train_losses, val_losses

@torch.no_grad()
def validate_ssn2v_improved(ssn2v, val_loader, device, percentile):
    """
    Improved validation function with detailed loss tracking
    """
    ssn2v.model.eval()
    val_losses = {'total': 0, 'n2v': 0, 'octa': 0, 'background': 0}
    num_batches = 0
    
    for batch in val_loader:
        oct1 = batch['raw'].to(device)
        oct2 = batch['fused'].to(device)
        octa_constraint = batch['original_constraint'].to(device)
        octb_constraint = batch['new_constraint'].to(device)
        
        loss_dict = ssn2v.train_step(oct1, oct2, octa_constraint, octb_constraint, percentile)
        
        for key in val_losses.keys():
            if key == 'total':
                val_losses[key] += loss_dict['total_loss'].item()
            else:
                val_losses[key] += loss_dict[f'{key}_loss'].item()
        num_batches += 1
    
    # Compute averages
    avg_losses = {k: v / num_batches for k, v in val_losses.items()}
    
    ssn2v.model.train()
    return avg_losses

class SSN2VStage2Tuned:
    def __init__(self, unet_model, device, alpha=2.0, beta=1.0, gamma=0.5, mask_percentage=0.15):
        self.model = unet_model
        self.device = device
        self.alpha = alpha  # OCTA constraint weight
        self.beta = beta    # Background constraint weight
        self.gamma = gamma  # Smoothness constraint weight
        self.mask_percentage = mask_percentage
        
    def parameters(self):
        """Return model parameters for optimizer"""
        return self.model.parameters()
        
    def normalize_input(self, x):
        """Normalize input to [0,1] range"""
        if torch.is_tensor(x):
            x_min = x.min()
            x_max = x.max()
            return (x - x_min) / (x_max - x_min + 1e-8)
        else:
            x_min = np.min(x)
            x_max = np.max(x)
            return (x - x_min) / (x_max - x_min + 1e-8)
    
    def create_n2v_mask(self, x):
        """Create Noise2Void blind-spot mask"""
        B, C, H, W = x.shape
        mask = torch.ones((B, C, H, W), device=self.device)
        n_pixels = int(self.mask_percentage * H * W)
        
        for b in range(B):
            flat_indices = torch.randperm(H * W, device=self.device)[:n_pixels]
            rows = flat_indices // W
            cols = flat_indices % W
            mask[b, :, rows, cols] = 0
            
        return mask
    
    def compute_background_mask(self, x, percentile=50):
        """
        Compute dynamic background mask using multiple regions
        Args:
            x: Input tensor [B, C, H, W]
            percentile: Percentile for background threshold
        """
        if torch.is_tensor(x):
            # Move to CPU for numpy operations
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        B, C, H, W = x_np.shape
        background_masks = []
        
        for b in range(B):
            # Get single image
            oct_img = x_np[b, 0]  # Assume single channel
            
            # Compute row and column intensities
            row_intensity = np.mean(oct_img, axis=1)
            col_intensity = np.mean(oct_img, axis=0)
            
            # Compute thresholds
            row_threshold = np.percentile(row_intensity, percentile)
            col_threshold = np.percentile(col_intensity, percentile)
            
            # Get background indices
            row_indices = np.where(row_intensity <= row_threshold)[0]
            col_indices = np.where(col_intensity <= col_threshold)[0]
            
            # Ensure minimum background region
            min_rows = max(len(row_indices), int(0.05 * H))
            min_cols = max(len(col_indices), int(0.05 * W))
            
            # Get background statistics
            background_region = oct_img[row_indices[:min_rows], :][:, col_indices[:min_cols]]
            mean_bg = np.mean(background_region)
            std_bg = np.std(background_region)
            threshold = mean_bg + 2 * std_bg
            
            # Create mask
            mask = oct_img < threshold
            background_masks.append(mask)
        
        # Stack masks and convert back to tensor
        background_masks = np.stack(background_masks)
        background_masks = np.expand_dims(background_masks, axis=1)  # Add channel dim
        
        if torch.is_tensor(x):
            return torch.from_numpy(background_masks).float().to(x.device)
        return background_masks.astype(np.float32)
        
    
    def compute_octa(self, oct1, oct2):
        """Compute OCTA signal with proper normalization"""
        diff_squared = torch.pow(oct1 - oct2, 2)
        sum_squared = torch.pow(oct1, 2) + torch.pow(oct2, 2)
        eps = 1e-6
        return diff_squared / (sum_squared + eps)
    
    def compute_weighted_n2v_loss(self, pred, target, mask):
        """N2V loss with gradient-based weighting"""
        # Compute gradients
        grad_x = torch.abs(F.conv2d(target, self.sobel_x(), padding=1))
        grad_y = torch.abs(F.conv2d(target, self.sobel_y(), padding=1))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Weight loss higher in high-gradient regions
        weights = 1.0 + 2.0 * (grad_mag / (grad_mag.max() + 1e-8))
        
        # Compute masked difference
        masked_diff = (pred - target) * (1 - mask)
        
        return torch.mean(weights * masked_diff ** 2)
    
    def compute_smoothness_loss(self, x):
        """Total variation smoothness loss"""
        diff_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        diff_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return torch.mean(diff_x) + torch.mean(diff_y)
    
    def compute_feature_consistency_loss(self, features1, features2):
        """Compute feature consistency between two sets of features"""
        loss = 0
        for f1, f2 in zip(features1, features2):
            loss += torch.mean((f1 - f2) ** 2)
        return loss / len(features1)
    
    def sobel_x(self):
        """Sobel X gradient kernel"""
        kernel = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).to(self.device)
        return kernel
    
    def sobel_y(self):
        """Sobel Y gradient kernel"""
        kernel = torch.tensor([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).to(self.device)
        return kernel
    
    def _process_patch(self, oct1_patch, oct2_patch, octa_patch, octb_patch, percentile):
        """
        Process a single patch with all SSN2V components
        Args:
            oct1_patch: First OCT patch [1, C, H, W]
            oct2_patch: Second OCT patch [1, C, H, W]
            octa_patch: OCTA constraint patch [1, C, H, W]
            octb_patch: OCTB constraint patch [1, C, H, W]
            percentile: Percentile for background threshold
        """
        # Normalize patch inputs
        oct1_patch = self.normalize_input(oct1_patch)
        oct2_patch = self.normalize_input(oct2_patch)
        octb_patch = self.normalize_input(octb_patch)
        
        # Create masks for this patch
        mask1 = self.create_n2v_mask(oct1_patch)
        mask2 = self.create_n2v_mask(oct2_patch)
        
        # Forward pass
        denoised_oct1, features1 = self.model(oct1_patch)
        denoised_oct2, features2 = self.model(oct2_patch)
        
        # Get background mask for this patch
        background_mask = self.compute_background_mask(oct1_patch, percentile)
        foreground_mask = 1 - background_mask
        
        # 1. N2V Loss with gradient weighting (only in foreground)
        loss_n2v1 = self.compute_weighted_n2v_loss(denoised_oct1, oct1_patch, mask1 * foreground_mask)
        loss_n2v2 = self.compute_weighted_n2v_loss(denoised_oct2, oct2_patch, mask2 * foreground_mask)
        
        # 2. OCTA constraint
        computed_octa = self.compute_octa(denoised_oct1, denoised_oct2)
        flow_mask = (octa_patch > 0).float() * foreground_mask
        loss_octa = self.alpha * torch.mean(flow_mask * (computed_octa - octa_patch) ** 2)
        
        # 3. Background constraint with binary cross-entropy
        loss_background = self.beta * (
            torch.mean(-background_mask * torch.log(torch.clamp(1 - denoised_oct1, min=1e-6))) +
            torch.mean(-background_mask * torch.log(torch.clamp(1 - denoised_oct2, min=1e-6)))
        )
        
        # 4. Edge-aware smoothness
        edge_mask = self.compute_edge_regions(background_mask)
        loss_smooth = self.gamma * (
            torch.mean(edge_mask * self.compute_smoothness_loss(denoised_oct1)) +
            torch.mean(edge_mask * self.compute_smoothness_loss(denoised_oct2))
        )
        
        # 5. Feature consistency within patch
        loss_feature = 0.1 * self.compute_feature_consistency_loss(features1, features2)
        
        # 6. OCTB structural constraint
        loss_octb = self.alpha * (
            torch.mean(foreground_mask * (denoised_oct1 - octb_patch) ** 2) +
            torch.mean(foreground_mask * (denoised_oct2 - octb_patch) ** 2)
        )
        
        # Total loss
        total_loss = (loss_n2v1 + loss_n2v2 + 
                    loss_octa + 
                    loss_background * 2.0 +
                    loss_smooth +
                    loss_feature +
                    loss_octb)
        
        return {
            'total_loss': total_loss,
            'n2v_loss': loss_n2v1 + loss_n2v2,
            'octa_loss': loss_octa,
            'background_loss': loss_background,
            'smooth_loss': loss_smooth,
            'feature_loss': loss_feature,
            'octb_loss': loss_octb,
            'denoised_oct1': denoised_oct1,
            'denoised_oct2': denoised_oct2
        }
        
    def train_step(self, oct1, oct2, octa_constraint, octb_constraint, percentile):
        """Execute training step with patch-based processing"""
        # Extract patches
        patches_oct1, positions, blend_weights = patch_processor(oct1, patch_size=128, stride=64)
        patches_oct2 = patch_processor(oct2, patch_size=128, stride=64)[0]
        patches_octa = patch_processor(octa_constraint, patch_size=128, stride=64)[0]
        patches_octb = patch_processor(octb_constraint, patch_size=128, stride=64)[0]
        
        # Initialize loss accumulators
        accumulated_losses = {
            'total_loss': 0,
            'n2v_loss': 0,
            'octa_loss': 0,
            'background_loss': 0,
            'smooth_loss': 0,
            'feature_loss': 0,
            'octb_loss': 0
        }
        
        denoised_patches1 = []
        denoised_patches2 = []
        
        # Process patches
        num_patches = len(patches_oct1)
        for i in range(num_patches):
            patch_dict = self._process_patch(
                patches_oct1[i:i+1],
                patches_oct2[i:i+1],
                patches_octa[i:i+1],
                patches_octb[i:i+1],
                percentile
            )
            
            # Accumulate losses
            for key in accumulated_losses.keys():
                accumulated_losses[key] += patch_dict[key]
            
            # Store denoised patches
            denoised_patches1.append(patch_dict['denoised_oct1'])
            denoised_patches2.append(patch_dict['denoised_oct2'])
        
        # Average losses
        for key in accumulated_losses.keys():
            accumulated_losses[key] /= num_patches
        
        # Reconstruct full images
        denoised_oct1 = reconstruct_from_patches(
            denoised_patches1, 
            positions, 
            oct1.shape,
            blend_weights
        )
        denoised_oct2 = reconstruct_from_patches(
            denoised_patches2, 
            positions, 
            oct2.shape,
            blend_weights
        )
        
        # Return all components
        return {
            'total_loss': accumulated_losses['total_loss'],
            'n2v_loss': accumulated_losses['n2v_loss'],
            'octa_loss': accumulated_losses['octa_loss'],
            'background_loss': accumulated_losses['background_loss'],
            'smooth_loss': accumulated_losses['smooth_loss'],
            'feature_loss': accumulated_losses['feature_loss'],
            'octb_loss': accumulated_losses['octb_loss'],
            'denoised_oct1': denoised_oct1,
            'denoised_oct2': denoised_oct2
        }

    def compute_edge_regions(self, background_mask, kernel_size=3):
        """Compute regions near background-foreground transitions"""
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(background_mask.device)
        dilated = F.conv2d(background_mask, kernel, padding=kernel_size//2)
        eroded = F.conv2d(background_mask, kernel, padding=kernel_size//2)
        return torch.abs(dilated - eroded)
            
    @torch.no_grad()
    def inference(self, oct_image):
        """Run inference on a single image"""
        self.model.eval()
        
        # Handle input format
        if not torch.is_tensor(oct_image):
            oct_image = torch.from_numpy(oct_image).to(self.device)
        if len(oct_image.shape) == 2:
            oct_image = oct_image.unsqueeze(0).unsqueeze(0)
        elif len(oct_image.shape) == 3:
            oct_image = oct_image.unsqueeze(0)
            
        # Normalize input
        oct_image = self.normalize_input(oct_image)
        
        # Forward pass
        denoised, _ = self.model(oct_image)
        
        return denoised
    
    def load_checkpoint(self, path, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.alpha = checkpoint.get('alpha', self.alpha)
        self.beta = checkpoint.get('beta', self.beta)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.mask_percentage = checkpoint.get('mask_percentage', self.mask_percentage)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint.get('epoch', None), checkpoint.get('loss', None)
    
def extract_patches(image, patch_size=128, stride=64):
    """Extract overlapping patches from image tensor"""
    B, C, H, W = image.shape
    patches = []
    positions = []
    
    for h in range(0, H-patch_size+1, stride):
        for w in range(0, W-patch_size+1, stride):
            patch = image[:, :, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
            positions.append((h, w))
            
    return torch.stack(patches, dim=0).squeeze(1), positions

def create_blending_weights(patch_size):
    """
    Create Gaussian-like blending weights for patch overlap
    Args:
        patch_size: Size of the square patch
    Returns:
        Tensor of shape [1, 1, patch_size, patch_size] with blending weights
    """
    import torch
    
    # Create coordinate grids
    x = torch.linspace(-1, 1, patch_size)
    y = torch.linspace(-1, 1, patch_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Compute radial distance from center
    radius = torch.sqrt(xx**2 + yy**2)
    
    # Create Gaussian-like weights
    sigma = 0.5
    weights = torch.exp(-(radius**2) / (2 * sigma**2))
    
    # Normalize weights
    weights = weights / weights.max()
    
    # Add batch and channel dimensions
    weights = weights.unsqueeze(0).unsqueeze(0)
    
    return weights

def patch_processor(image, patch_size=128, stride=64, device='cuda'):
    """
    Extract overlapping patches from image with position tracking
    Args:
        image: Input tensor [B, C, H, W]
        patch_size: Size of patches to extract
        stride: Stride between patches
    Returns:
        patches: List of patches
        positions: List of (h, w) positions
    """
    B, C, H, W = image.shape
    patches = []
    positions = []
    
    # Get blending weights once
    blend_weights = create_blending_weights(patch_size).to(device)
    
    for h in range(0, H-patch_size+1, stride):
        for w in range(0, W-patch_size+1, stride):
            patch = image[:, :, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
            positions.append((h, w))
    
    return torch.cat(patches, dim=0), positions, blend_weights

def reconstruct_from_patches(patches, positions, original_shape, blend_weights=None):
    """
    Reconstruct image from patches with Gaussian blending
    Args:
        patches: List of patches
        positions: List of (h, w) positions
        original_shape: Target shape [B, C, H, W]
        blend_weights: Pre-computed blending weights
    Returns:
        Reconstructed image
    """
    B, C, H, W = original_shape
    recon = torch.zeros(original_shape, device=patches[0].device)
    count = torch.zeros(original_shape, device=patches[0].device)
    
    patch_size = patches[0].shape[-2:]  # Handle non-square patches
    
    # Create blending weights if not provided
    if blend_weights is None:
        blend_weights = create_blending_weights(patch_size[0]).to(patches[0].device)
    
    for patch, (h, w) in zip(patches, positions):
        # Apply blending weights
        weighted_patch = patch * blend_weights
        recon[:, :, h:h+patch_size[0], w:w+patch_size[1]] += weighted_patch
        count[:, :, h:h+patch_size[0], w:w+patch_size[1]] += blend_weights
    
    # Average overlapping regions
    mask = (count > 0).float()
    recon = recon / (count + 1e-8) * mask
    
    return recon