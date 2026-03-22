"""
Authors: Ruijie He, Botong Cai, Ziqi Yang
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# --- Import Core Components ---
from dataset import Sen2_MTC_New_Multi_Simple
from ours_unet import UNet
from network import Network

def worker_init_fn(worker_id):
    """ Prevent duplicate seeds in multi-process data augmentation """
    np.random.seed(torch.initial_seed() % 2 ** 32 + worker_id)

def train():
    # --- 1. Device Detection ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f"🔥 Starting Cloud Removal Model Engine...")
    print(f"📍 Current Device: {device}")
    if torch.cuda.is_available():
        print(f"🎸 GPU Model: {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    # --- 2. Path and Directory Preparation ---
    my_root = r'CTGAN/CTGAN/CTGAN/Sen2_MTC/dataset'
    os.makedirs('checkpoints', exist_ok=True)

    # --- 3. Dataset and Loader Setup ---
    train_dataset = Sen2_MTC_New_Multi_Simple(data_root=my_root, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    val_dataset = Sen2_MTC_New_Multi_Simple(data_root=my_root, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"✅ Data loaded: {len(train_dataset)} samples in the training set")

    # --- 4. Model Initialization ---
    unet_params = {
        "width": 64,
        "enc_blk_nums": [1, 1, 1, 1],
        "dec_blk_nums": [1, 1, 1, 1]
    }
    beta_params = {
        "train": {
            "schedule": "sigmoid",
            "n_timestep": 2000
        }
    }

    # Assemble network components
    unet = UNet(**unet_params)
    model = Network(unet, beta_params).to(device)

    # Crucial step: Initialize noise schedule constants
    model.set_new_noise_schedule(device=device, phase='train')

    # --- 5. [IMPORTANT] Specify Loss Type as MSE ---
    # Explicitly use Mean Squared Error and move to GPU
    model.loss_fn = nn.MSELoss().to(device)
    print(f"📏 Training Loss Type specified: MSE")

    # --- 6. Optimizer Configuration ---
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- 7. Main Training Loop ---
    print(f"🚀 Starting 2000 training epochs...")
    start_time = time.time()

    for epoch in range(1, 2001):
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            # Move data to device: gt [B, 3, 256, 256] and cond [B, 9, 256, 256]
            gt = batch['gt_image'].to(device)
            cond = batch['cond_image'].to(device)

            optimizer.zero_grad()

            # Trigger Forward logic: Continuous noise addition + NAF-UNet prediction of x0
            loss = model(gt, cond)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print real-time Loss every 10 iterations
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"   [Epoch {epoch}] Iter [{i + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")

        # Calculate and output average Loss for the current Epoch
        avg_loss = epoch_loss / len(train_loader)
        elapsed = (time.time() - start_time) / 60
        print(f"⭐ End of Epoch [{epoch}] | Average Loss: {avg_loss:.6f} | Elapsed Time: {elapsed:.2f} min")

        # --- 8. Periodic Saving (Every 10 Epochs) ---
        if epoch % 10 == 0:
            save_path = f"checkpoints/model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 [SAVE] Weights saved to: {save_path}")

    print(f"🏁 Task completed! Total time: {(time.time() - start_time) / 3600:.2f} hours")

# --- 9. Program Entry Point ---
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted manually. Exiting.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")