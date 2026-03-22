import os
import numpy as np
from PIL import Image
import tqdm
import torch
import lpips
import sys

# 1. Import FID calculation API
from pytorch_fid import fid_score

# 2. Import skimage metrics (Compatible with version 0.17.2)
from skimage.measure import compare_psnr, compare_ssim

def psnr_ssim_cal(cloudfree, predict):
    """
    Calculates PSNR and SSIM using skimage parameters.
    """
    psnr = compare_psnr(cloudfree, predict, data_range=255)
    ssim = compare_ssim(cloudfree, predict, multichannel=True, 
                        gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
    return psnr, ssim

def get_core_name(filename):
    """
    Extracts the base filename to match pairs.
    """
    name = os.path.splitext(filename)[0]
    return name.replace('_fake_B', '').replace('_real_B', '')

def calculate_all_metrics():
    # --- 1. Directory Path Configuration ---
    # Absolute or relative paths to your image directories
    fake_dir = r'CTGAN-main/CTGAN-main/images/ourtiffs'
    real_dir = r'CTGAN-main/CTGAN-main/images/real'

    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.png')])
    real_index = {get_core_name(f): f for f in real_files}
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.endswith('.png')])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. Initialize LPIPS Model ---
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    match_count = 0
    skip_count = 0

    print(f"Starting per-image metric calculation (PSNR/SSIM/LPIPS)... Total pairs: {len(fake_files)}")

    # --- 3. Main Calculation Loop ---
    for f_name in tqdm.tqdm(fake_files):
        core = get_core_name(f_name)
        
        if core in real_index:
            fake_path = os.path.join(fake_dir, f_name)
            real_path = os.path.join(real_dir, real_index[core])

            img_fake_pil = Image.open(fake_path).convert('RGB')
            img_real_pil = Image.open(real_path).convert('RGB')
            
            img_fake_np = np.array(img_fake_pil)
            img_real_np = np.array(img_real_pil)

            # Skip images that are too small for SSIM calculation
            if min(img_fake_np.shape[0], img_fake_np.shape[1]) < 7:
                skip_count += 1
                continue
            
            try:
                # Calculate PSNR/SSIM
                psnr, ssim = psnr_ssim_cal(img_real_np, img_fake_np)
                
                # Calculate LPIPS
                t_fake = lpips.im2tensor(img_fake_np).to(device)
                t_real = lpips.im2tensor(img_real_np).to(device)
                
                with torch.no_grad():
                    lpips_val = loss_fn_alex(t_real, t_fake).item()

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips_val)
                match_count += 1
                
            except Exception as e:
                print(f"\nCalculation error for {f_name}: {e}")

    # --- 4. FID Score Calculation via API ---
    print("\n--- Initializing FID Calculation (All Images) ---")
    fid_value = None
    num_workers = 0 if sys.platform == "win32" else 8
    
    try:
        # Directly calling the pytorch-fid core function
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=8,
            device=device,
            dims=2048, # Inception feature dimension, default is 2048
            num_workers=num_workers
        )
    except Exception as e:
        print(f"❌ FID calculation failed: {e}")

    # --- 5. Final Summary Output ---
    if match_count > 0:
        print(f"\n" + "="*45)
        print(f"📊 FINAL EXPERIMENTAL RESULTS SUMMARY")
        print(f"="*45)
        print(f"Successfully matched pairs: {match_count}")
        print(f"Skipped (dimensions too small): {skip_count}")
        print(f"-"*45)
        print(f"Mean PSNR:  {np.mean(psnr_list):.4f} dB (↑)")
        print(f"Mean SSIM:  {np.mean(ssim_list):.4f} (↑)")
        print(f"Mean LPIPS: {np.mean(lpips_list):.4f} (↓)")
        if fid_value is not None:
            print(f"FID Score:  {fid_value:.4f} (↓)")
        else:
            print(f"FID Score:  Calculation Failed")
        print(f"="*45)
    else:
        print("\n❌ Error: No matching pairs found. Please verify directory paths or filename formats.")

if __name__ == "__main__":
    calculate_all_metrics()