import matplotlib.pyplot as plt
from IPython.display import display, clear_output  # Enables dynamic updates


def visualise_n2v(img, img2, output):

    clear_output(wait=True) 
    ax, fig = plt.subplots(1, 3, figsize=(20, 50))
    fig[0].imshow(img.squeeze(), cmap='gray')
    fig[0].axis('off')
    fig[0].set_title('Input Noisy Image')
    fig[1].imshow(output.squeeze(), cmap='gray')
    fig[1].axis('off')
    fig[1].set_title('Output Image')
    fig[2].imshow(img2.squeeze(), cmap='gray')
    fig[2].axis('off')
    fig[2].set_title('Target Noisy Image')
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

def plot_training_progress(train_losses, val_losses):
    """
    Plot training and validation losses over time
    """
    plt.figure(figsize=(15, 5))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses['total'], label='Train Total')
    plt.plot(val_losses['total'], label='Val Total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot component losses
    plt.subplot(1, 2, 2)
    for key in ['n2v', 'octa', 'background']:
        plt.plot(train_losses[key], label=f'Train {key}')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def visualise_batch(sample): # pass in batch
    raw, constraint, fused, new_constraint = sample['raw'][0][0], sample['original_constraint'][0][0], sample['fused'][0][0], sample['new_constraint'][0][0]
    metadata = sample['metadata']

    print(f"Raw: {raw.shape}, Constraint: {constraint.shape}, Fused: {fused.shape}")

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(raw, cmap='gray')
    ax[1].imshow(constraint, cmap='gray')
    ax[2].imshow(fused, cmap='gray')
    ax[3].imshow(new_constraint, cmap='gray')

def visualize_processing_steps(processor, original_image):
    """
    Visualize each step of the post-processing pipeline
    """
    # Convert to numpy if needed
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    if len(original_image.shape) == 3:
        original_image = original_image[0]
        
    # Get intermediate results
    thresholded = processor.apply_retina_thresholding(original_image)
    filtered = processor.denoise_median_filter(thresholded)
    processed = processor.process_image(original_image)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original')
    
    axes[1].imshow(thresholded, cmap='gray')
    axes[1].set_title('After Thresholding')
    
    axes[2].imshow(filtered, cmap='gray')
    axes[2].set_title('After Median Filter')
    
    axes[3].imshow(processed, cmap='gray')
    axes[3].set_title('Final Processed')
    
    plt.tight_layout()
    return fig

def visualize_stage2(oct1, oct2, octa_constraint, octb_constraint, denoised_oct1, denoised_oct2):
    clear_output(wait=True)  # Clears previous outputs, ensuring only the latest visualizations appear
    
    fig, ax = plt.subplots(1, 6, figsize=(15, 5))
    ax[0].imshow(oct1, cmap='gray')
    ax[0].set_title('OCT 1')
    ax[1].imshow(oct2, cmap='gray')
    ax[1].set_title('OCT 2')
    ax[2].imshow(octa_constraint, cmap='gray')
    ax[2].set_title('OCTA Constraint')
    ax[3].imshow(octb_constraint, cmap='gray')
    ax[3].set_title('OCTB Constraint')
    ax[4].imshow(denoised_oct1, cmap='gray')
    ax[4].set_title('Denoised OCT 1')
    ax[5].imshow(denoised_oct2, cmap='gray')
    ax[5].set_title('Denoised OCT 2')
    
    plt.show()