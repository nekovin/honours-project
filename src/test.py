import torch
from src.visualise import visualise_n2v
import matplotlib.pyplot as plt

def test_stage1(img_size, model, test_loader, criterion, device, path):

    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(e)
    
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            noisy, target = data
            noisy, target = noisy.to(device), target.to(device)
            output, features = model(noisy)
            loss = criterion(output, target)
            test_loss += loss.item()
    
    visualise_n2v(
                    noisy.cpu().detach().numpy(),
                    target.cpu().detach().numpy(),
                    output.cpu().detach().numpy()
                )
    return test_loss / len(test_loader)

### 

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

class SSN2VEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def compute_psnr(self, denoised, reference):
        """
        Compute Peak Signal-to-Noise Ratio
        """
        return psnr(reference, denoised, data_range=reference.max() - reference.min())
    
    def compute_ssim(self, denoised, reference):
        """
        Compute Structural Similarity Index
        """
        return ssim(reference, denoised, data_range=reference.max() - reference.min())
    
    def compute_contrast(self, image):
        """
        Compute local contrast measure
        """
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        
        from scipy.ndimage import gaussian_filter
        
        # Compute local standard deviation
        local_std = gaussian_filter(image, sigma=3, order=0)
        local_mean = gaussian_filter(image**2, sigma=3, order=0)
        local_contrast = np.sqrt(local_mean - local_std**2)
        
        return np.mean(local_contrast)
    
    def compute_speckle_index(self, image):
        """
        Compute speckle index (ratio of std dev to mean in local patches)
        """
        if torch.is_tensor(image):
            image = image.cpu().numpy()
            
        patch_size = 7
        patches = np.lib.stride_tricks.sliding_window_view(
            image, (patch_size, patch_size)
        )
        
        means = np.mean(patches, axis=(2,3))
        stds = np.std(patches, axis=(2,3))
        speckle_index = np.mean(stds / (means + 1e-8))
        
        return speckle_index
    
    def evaluate_image(self, denoised, reference, original=None):
        """
        Compute comprehensive evaluation metrics
        """
        self.metrics = {
            'psnr': self.compute_psnr(denoised, reference),
            'ssim': self.compute_ssim(denoised, reference),
            'contrast': self.compute_contrast(denoised),
            'speckle_index': self.compute_speckle_index(denoised)
        }
        
        if original is not None:
            self.metrics.update({
                'improvement_psnr': self.compute_psnr(denoised, reference) - 
                                  self.compute_psnr(original, reference),
                'improvement_ssim': self.compute_ssim(denoised, reference) - 
                                  self.compute_ssim(original, reference)
            })
        
        return self.metrics
    
    def visualize_results(self, original, denoised, reference, processed_denoised):
        """
        Visualize evaluation results with metrics
        """
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        
        axes[1].imshow(denoised, cmap='gray')
        axes[1].set_title('Denoised')
        
        axes[2].imshow(reference, cmap='gray')
        axes[2].set_title('Reference')

        axes[3].imshow(processed_denoised, cmap='gray')
        axes[3].set_title('Processed Denoised')
        
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()])
        fig.text(4, 1, metrics_text, transform=axes[2].transAxes)
        
        plt.tight_layout()
        return fig
    
def process_stage2_example(val_loader, device, ssn2v_model):
    from src.postprocessing import SSN2VPostProcessor
    from src.evaluate import SSN2VEvaluator

    sample = next(iter(val_loader))

    # Predict the output
    sample.keys()

    oct1 = sample['raw'].to(device)
    oct2 = sample['fused'].to(device)
    octa_constraint = sample['original_constraint'].to(device)
    octb_constraint = sample['fused'].to(device)

    # Get model output
    output = ssn2v_model.train_step(oct1, oct2, octa_constraint, octb_constraint)

    # Initialize processors
    processor = SSN2VPostProcessor(device='cuda')
    evaluator = SSN2VEvaluator()

    # Get denoised image and ensure correct shape
    denoised_image = output['denoised_oct1'].detach().cpu().numpy()
    if len(denoised_image.shape) == 4:  # [B, C, H, W]
        denoised_image = denoised_image[0, 0]  # Take first image from batch and channel
    elif len(denoised_image.shape) == 3:  # [C, H, W]
        denoised_image = denoised_image[0]  # Take first channel

    # Process the denoised image
    processed_image = processor.process_image(denoised_image)

    # Prepare reference and original images with correct shapes
    oct1_np = oct1.detach().cpu().numpy()
    oct2_np = oct2.detach().cpu().numpy()

    if len(oct1_np.shape) == 4:  # [B, C, H, W]
        oct1_np = oct1_np[0, 0]  # Take first image from batch and channel
        oct2_np = oct2_np[0, 0]
    elif len(oct1_np.shape) == 3:  # [C, H, W]
        oct1_np = oct1_np[0]  # Take first channel
        oct2_np = oct2_np[0]

    # Verify shapes
    print(f"Processed image shape: {processed_image.shape}")
    print(f"OCT1 shape: {oct1_np.shape}")
    print(f"OCT2 shape: {oct2_np.shape}")

    # Ensure all images have the same shape
    assert oct1_np.shape == oct2_np.shape == processed_image.shape, \
        f"Shape mismatch: oct1={oct1_np.shape}, oct2={oct2_np.shape}, processed={processed_image.shape}"

    # Compute evaluation metrics
    metrics = evaluator.evaluate_image(
        processed_image,
        oct2_np,  # reference
        oct1_np   # original
    )

    # Visualize results
    evaluator.visualize_results(oct1_np, processed_image, oct2_np, denoised_image)

    print(f"Metrics: {metrics}")