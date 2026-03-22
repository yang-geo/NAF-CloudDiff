"""
Authors: Ruijie He, Botong Cai, Ziqi Yang
"""

import torch
import torch.utils.data as data
import numpy as np
import tifffile as tiff
import os

class Sen2_MTC_New_Multi_Simple(data.Dataset):
    """
    Dataset class for Multi-Temporal Cloud Removal (Sen2-MTC).

    This class loads 3 multi-temporal cloudy images as input and 1 cloud-free image
    as the target (Ground Truth).
    """

    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []  # Stores list of [cloud_0, cloud_1, cloud_2, ground_truth] paths
        self.image_name = []

        # Load tile identifiers from a text file (train.txt or val.txt)
        txt_path = os.path.join(data_root, f'{mode}.txt')
        self.tile_list = np.loadtxt(txt_path, dtype=str)

        # Indexing logic: traverse tiles and pair cloudy images with cloudless counterparts
        for tile in self.tile_list:
            cloudless_path = os.path.join(self.data_root, 'Sen2_MTC/Sen2_MTC', tile, 'cloudless')
            image_name_list = [name.split('.')[0] for name in os.listdir(cloudless_path)]

            for name in image_name_list:
                # Define paths for 3 temporal cloudy snapshots and the 1 target ground truth
                c0 = os.path.join(self.data_root, 'Sen2_MTC/Sen2_MTC', tile, 'cloud', name + '_0.tif')
                c1 = os.path.join(self.data_root, 'Sen2_MTC/Sen2_MTC', tile, 'cloud', name + '_1.tif')
                c2 = os.path.join(self.data_root, 'Sen2_MTC/Sen2_MTC', tile, 'cloud', name + '_2.tif')
                gt = os.path.join(self.data_root, 'Sen2_MTC/Sen2_MTC', tile, 'cloudless', name + '.tif')

                self.filepair.append([c0, c1, c2, gt])
                self.image_name.append(name)

    def image_read(self, path, rot_k, flip_type):
        """
        Helper to read, augment, and normalize Sentinel-2 .tif files.
        """
        # Load TIF file and transpose from (H, W, C) to PyTorch's (C, H, W)
        img = tiff.imread(path).astype(np.float32)
        img = img.transpose((2, 0, 1))

        # Apply geometric data augmentation only during training
        if self.mode == 'train':
            if flip_type != 0:
                img = np.flip(img, flip_type)
            if rot_k != 0:
                img = np.rot90(img, rot_k, (1, 2))

        # Convert to Tensor and Normalize
        image = torch.from_numpy(img.copy()).float()
        image = image / 10000.0  # Scale typical S2 reflectance (0-10000) to 0-1
        image = (image - 0.5) / 0.5  # Map range to [-1, 1] for GAN/Diffusion models
        return image

    def __getitem__(self, index):
        paths = self.filepair[index]

        # Generate random augmentation parameters (consistent across all 4 images in the set)
        rot_k = np.random.randint(0, 4) if self.mode == 'train' else 0
        flip_type = np.random.randint(0, 3) if self.mode == 'train' else 0

        # Read images (c0, c1, c2 are cloudy inputs; gt is cloud-free target)
        img_c0 = self.image_read(paths[0], rot_k, flip_type)
        img_c1 = self.image_read(paths[1], rot_k, flip_type)
        img_c2 = self.image_read(paths[2], rot_k, flip_type)
        img_gt = self.image_read(paths[3], rot_k, flip_type)

        return {
            # Target image (usually first 3 channels: RGB)
            'gt_image': img_gt[:3, :, :],
            # Conditional input: 3 RGB images stacked along channel dimension (9 channels total)
            'cond_image': torch.cat([img_c0[:3, :, :], img_c1[:3, :, :], img_c2[:3, :, :]], dim=0),
            'path': self.image_name[index] + ".png"
        }

    def __len__(self):
        return len(self.filepair)
