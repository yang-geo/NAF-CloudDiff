"""
Authors: Ruijie He, Botong Cai, Ziqi Yang
"""

import torch
import os
import numpy as np
import tifffile as tiff
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import time

# Import custom components
from dataset import Sen2_MTC_New_Multi_Simple
from ours_unet import UNet
from network import Network


def run_inference():
    # --- 1. Environment and Path Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_root = r'CTGAN/CTGAN/CTGAN/Sen2_MTC/dataset'
    model_path = 'checkpoints/model_epoch_150.pth'

    # Setup output directories
    base_output = 'inference_results_150'
    tiff_dir = os.path.join(base_output, 'tiffs')  # For 16-bit TIFF data
    img_dir = os.path.join(base_output, 'images')  # For 8-bit PNG preview

    os.makedirs(tiff_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # --- 2. Model Initialization ---
    unet_params = {"width": 64, "enc_blk_nums": [1, 1, 1, 1], "dec_blk_nums": [1, 1, 1, 1]}
    beta_params = {"train": {"schedule": "sigmoid", "n_timestep": 2000}}

    unet = UNet(**unet_params)
    model = Network(unet, beta_params).to(device)
    model.set_new_noise_schedule(device=device, phase='train')

    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Weights loaded: {model_path}")
    else:
        print(f"❌ Weight file not found: {model_path}");
        return

    # --- 3. Data Loading ---
    test_dataset = Sen2_MTC_New_Multi_Simple(data_root=my_root, mode='train')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"📦 Total samples: {len(test_dataset)}")

    # --- 4. Inference Loop ---
    print(f"🚀 Starting inference...")
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            cond = batch['cond_image'].to(device)
            file_name = batch['path'][0]

            # Execute DPM-Solver inference
            restored = model.restoration(cond)

            # --- 5. Post-processing ---
            # Denormalize from [-1, 1] to [0, 1]
            restored_normalized = (restored.clamp(-1, 1) + 1) / 2

            # --- 6. Save Results ---

            # (A) Save Preview Image (PNG)
            # Apply brightness boost (x3.5) for visualization
            # vis_boost = torch.clamp(restored_normalized * 3.5, 0, 1)
            # save_png_path = os.path.join(img_dir, file_name)
            # vutils.save_image(vis_boost, save_png_path)

            # (B) Save Research Data (16-bit TIFF)
            # Use original normalized data (no boost) for scientific integrity
            res_np = restored_normalized.squeeze().cpu().numpy().transpose(1, 2, 0)
            res_tiff = (res_np * 10000).astype(np.uint16)

            tiff_name = file_name.replace('.png', '.tif')
            save_tiff_path = os.path.join(tiff_dir, tiff_name)
            tiff.imwrite(save_tiff_path, res_tiff)

            if (i + 1) % 10 == 0:
                print(f" ┖ Processed: [{i + 1}/{len(test_dataset)}]")

    print(f"🏁 Task complete!")
    print(f"📂 TIFF path: {tiff_dir}")
    print(f"🖼️ Image path: {img_dir}")


if __name__ == "__main__":
    run_inference()