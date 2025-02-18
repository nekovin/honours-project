
# Honours Project
## Preprocessing and Data Loading
- 300x300 grayscale
- normalised between 0 and 1
- scaled to 256x256

## Stage 0: Fusion 
- Fuse the images by n neighbours on both sides

## Stage 1: Constraint Learning
- A guided process in which to extract the meaningful structures from the raw and fused images with N2V
- MSE and L1

## Stage 2: Feature Learning and Intermediate Denoising
- Uses the weights and loss from the first to learn an approximate output
- MSE and L1

## Stage 3: Contrast Learning and Final Denoising
- Contrasts the Intermediate Denoising to this Denoising with N2N or N2V
- CNR, PSNR, SSIM, SNR

## Debugging
- Problem was caused by normalisation issues. Data is between -1 and 0 when it was trying to norm between 0 and 1