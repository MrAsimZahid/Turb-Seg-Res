import os
import warnings
import logging
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    seamLessClone, load_images, gaussian_mean, getFlowMaskGlobal,
    stabilize_GPU_optimized, computePseudoCn2V2, load_and_predict
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

@dataclass
class Config:
    """Default configuration settings"""
    do_stabilize: bool = True
    process_frames: int = 100
    resize_factor: int = 1
    max_stb: int = 50
    input_path: str = 'Input/Single_Car/*.png'
    save_path: str = 'Output/Single_Car/'
    model_path: str = 'PretrainedModel/restormer_ASUSim_trained.pth'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    kernel_spatial_size: int = 20

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Turbulence Segmentation and Restoration')
    parser.add_argument('--input', type=str, help='Input path (e.g., Input/sniper/*.png)')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--frames', type=int, help='Number of frames to process')
    parser.add_argument('--resize', type=float, help='Resize factor')
    parser.add_argument('--no-stabilize', action='store_true', help='Disable stabilization')
    parser.add_argument('--model', type=str, help='Path to pretrained model')
    parser.add_argument('--max-stb', type=int, help='Maximum stabilization pixels')
    return parser.parse_args()

@torch.no_grad()
def process_batch(img_tensor: torch.Tensor, config: Config) -> torch.Tensor:
    """Process a batch of images"""
    # Compute pseudo Cn2 and stabilize if enabled
    pseudo_cn2 = computePseudoCn2V2(img_tensor)
    if config.do_stabilize:
        img_tensor, _, _ = stabilize_GPU_optimized(img_tensor, MaxStb=config.max_stb)
    pseudo_cn2 = (computePseudoCn2V2(img_tensor)**0.8) * 1.3

    # Generate masks
    masks = getFlowMaskGlobal(img_tensor, n=5, ThMagnify=1.5)
    mask_tensor = torch.tensor(np.array(masks), dtype=torch.float, device=config.device)
    mask_tensor = rearrange(mask_tensor, 'n h w -> 1 1 n h w')

    # Process masks
    kernel_size = [int(pseudo_cn2*2+1), config.kernel_spatial_size, config.kernel_spatial_size]
    kernel_size = [k + 1 if k % 2 == 0 else k for k in kernel_size]
    kernel = torch.ones(kernel_size, device=config.device)
    kernel = rearrange(kernel, 'd h w -> 1 1 d h w')

    padding = [k // 2 for k in kernel_size]
    dilated_mask = F.conv3d(mask_tensor, kernel, padding=padding)
    masks = rearrange((dilated_mask > 0).float(), '1 1 n h w -> n h w')
    masks = repeat(masks, 'n h w -> n c h w', c=3)

    # Process images
    gaussian_img = gaussian_mean(img_tensor, masks, sigma=pseudo_cn2)
    return gaussian_img, masks


def save_results(processed: torch.Tensor, config: Config) -> None:
    """Save processed images"""
    os.makedirs(config.save_path, exist_ok=True)
    logger.info(f'Saving results to {config.save_path}')

    for i in tqdm(range(len(processed)), desc="Saving images"):
        # Get image and ensure it's on CPU
        image = processed[i].cpu()
        
        # Convert to numpy and ensure correct dimensions (H,W,C)
        image_np = image.numpy()
        
        # Rearrange dimensions if needed (from C,H,W to H,W,C)
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
            
        # Ensure values are in correct range
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        try:
            # Try to enhance the image
            output = load_and_predict(image_np, config.model_path)
            plt.imsave(f'{config.save_path}/{str(i).zfill(4)}_Enhanced.jpg', 
                      output, 
                      format='jpg')
        except Exception as e:
            logger.warning(f"Failed to enhance image {i}: {str(e)}")
            # Save unenhanced image
            try:
                plt.imsave(f'{config.save_path}/{str(i).zfill(4)}_NotEnhanced.jpg', 
                          image_np,
                          format='jpg')
            except Exception as save_error:
                logger.error(f"Failed to save unenhanced image {i}: {str(save_error)}")
                # Try one last time with dimension check
                if len(image_np.shape) == 3 and image_np.shape[2] != 3:
                    image_np = np.transpose(image_np, (1, 2, 0))
                plt.imsave(f'{config.save_path}/{str(i).zfill(4)}_NotEnhanced.jpg',
                          image_np,
                          format='jpg')

def main():
    """Main execution function"""
    # Load configuration
    args = parse_args()
    config = Config()
    
    # Update config with CLI arguments
    if args.input:
        config.input_path = args.input
    if args.output:
        config.save_path = args.output
    if args.frames:
        config.process_frames = args.frames
    if args.resize:
        config.resize_factor = args.resize
    if args.no_stabilize:
        config.do_stabilize = False
    if args.model:
        config.model_path = args.model
    if args.max_stb:
        config.max_stb = args.max_stb

    # Check GPU requirements
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 24.0:
            logger.warning(f'Available GPU VRAM: {gpu_memory:.2f} GB might not be sufficient')

    try:
        # Load and process images
        logger.info(f'Loading images from {config.input_path}')
        img_tensor = load_images(config.input_path, ResizeFactor=config.resize_factor)
        img_tensor = img_tensor[:config.process_frames]
        logger.info(f'Processing {len(img_tensor)} frames')

        # Process images in batches
        processed_imgs, masks = process_batch(img_tensor, config)
        
        # Save results
        save_results(processed_imgs, config)
        logger.info('Processing completed successfully')

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()