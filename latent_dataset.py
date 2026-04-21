"""
Dataset that loads pre-computed VAE latents from shard files.
All latents are loaded into RAM for maximum training speed.
"""
import torch
import os
import glob
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    Loads pre-computed VAE latents from shard .pt files into RAM.
    ~10GB for full ImageNet (1.28M × 4×32×32 × fp16).
    
    Args:
        latent_dir: Directory containing shard_XXXXXXX.pt files
    """
    def __init__(self, latent_dir):
        print(f"Loading latents from {latent_dir}...")
        shard_files = sorted(glob.glob(os.path.join(latent_dir, "shard_*.pt")))
        assert len(shard_files) > 0, f"No shard files found in {latent_dir}"
        
        all_latents = []
        all_labels = []
        
        for sf in shard_files:
            data = torch.load(sf, weights_only=True)
            all_latents.append(data["latents"])
            all_labels.append(data["labels"])
        
        self.latents = torch.cat(all_latents, dim=0)  # (N, 4, 32, 32) fp16
        self.labels = torch.cat(all_labels, dim=0)     # (N,) int64
        
        print(f"Loaded {len(self.latents)} latents, shape={self.latents.shape}, "
              f"dtype={self.latents.dtype}, "
              f"RAM usage: {self.latents.nbytes / 1e9:.1f} GB")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, index):
        return self.latents[index].float(), self.labels[index]
