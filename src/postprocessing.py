import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

class SSN2VPostProcessor:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_background_threshold(self, oct_image):
        """
        Compute adaptive background threshold using top portion of image
        """
        if torch.is_tensor(oct_image):
            oct_image = oct_image.cpu().numpy()
        
        if len(oct_image.shape) == 3:  # Handle batch dimension
            oct_image = oct_image[0]
            
        # Use top 50 rows as background reference
        background_region = oct_image[:50, :]
        mean_bg = np.mean(background_region)
        std_bg = np.std(background_region)
        return mean_bg + 2 * std_bg

    def apply_retina_thresholding(self, oct_image, margin=10):
        """
        Apply retina-based thresholding to remove background noise
        """
        if torch.is_tensor(oct_image):
            oct_image = oct_image.cpu().numpy()
            
        if len(oct_image.shape) == 3:  # Handle batch dimension
            oct_image = oct_image[0]
            
        threshold = self.compute_background_threshold(oct_image)
        
        # Create retina mask
        retina_mask = oct_image > threshold
        
        # Dilate mask slightly to include margins
        from scipy.ndimage import binary_dilation
        retina_mask = binary_dilation(retina_mask, iterations=margin)
        
        # Apply mask to image
        thresholded_image = oct_image * retina_mask
        
        return thresholded_image

    def denoise_median_filter(self, oct_image, kernel_size=3):
        """
        Apply median filtering using scipy instead of PyTorch
        """
        if torch.is_tensor(oct_image):
            oct_image = oct_image.cpu().numpy()
            
        if len(oct_image.shape) == 3:  # Handle batch dimension
            oct_image = oct_image[0]
            
        # Apply median filter
        denoised = median_filter(oct_image, size=kernel_size)
        
        return denoised

    def process_image(self, denoised_image):
        """
        Apply full post-processing pipeline
        """
        # Ensure correct dimensionality
        if torch.is_tensor(denoised_image):
            denoised_image = denoised_image.cpu().numpy()
            
        if len(denoised_image.shape) == 3:  # Handle batch dimension
            denoised_image = denoised_image[0]
            
        # 1. Apply retina thresholding
        thresholded = self.apply_retina_thresholding(denoised_image)
        
        # 2. Apply light median filtering
        filtered = self.denoise_median_filter(thresholded)
        
        # 3. Normalize output to [0,1]
        processed = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8)
        
        return processed