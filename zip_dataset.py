"""
ImageFolder-compatible dataset that reads images directly from a zip file.
Avoids extracting the full ImageNet dataset to disk.
"""
import zipfile
import io
import os
import random
from PIL import Image
from torch.utils.data import Dataset


class ZipImageFolder(Dataset):
    """
    A dataset that reads images from a zip archive organized in ImageFolder structure.
    Expected zip structure: <prefix>/class_dir/image.JPEG
    
    Args:
        zip_path: Path to the zip file
        prefix: Path prefix inside the zip to the train folder 
                (e.g., 'ILSVRC/Data/CLS-LOC/train')
        transform: Optional transform to apply to images
    """
    def __init__(self, zip_path, prefix='ILSVRC/Data/CLS-LOC/train', transform=None):
        self.zip_path = zip_path
        self.prefix = prefix.rstrip('/')
        self.transform = transform
        
        # We'll open a zip handle per-worker later; for indexing use one handle
        print(f"Indexing zip file: {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = zf.namelist()
        
        # Filter to only JPEG files under the prefix
        image_files = [
            n for n in all_names
            if n.startswith(self.prefix + '/') and n.lower().endswith('.jpeg')
        ]
        
        # Extract class directories and sort them for consistent class indices
        class_set = set()
        for path in image_files:
            # path like: ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_10026.JPEG
            rel = path[len(self.prefix) + 1:]  # n01440764/n01440764_10026.JPEG
            class_dir = rel.split('/')[0]
            class_set.add(class_dir)
        
        self.classes = sorted(list(class_set))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build samples list: (zip_path_in_archive, class_index)
        self.samples = []
        for path in image_files:
            rel = path[len(self.prefix) + 1:]
            class_dir = rel.split('/')[0]
            class_idx = self.class_to_idx[class_dir]
            self.samples.append((path, class_idx))
        
        # Sort for reproducibility
        self.samples.sort(key=lambda x: x[0])
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes.")
        
        # Per-worker zip file handle (will be initialized in worker_init or on first access)
        self._zip_handle = None
    
    def _get_zip_handle(self):
        """Lazily open zip handle (one per DataLoader worker process)."""
        if self._zip_handle is None:
            self._zip_handle = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip_handle
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        zf = self._get_zip_handle()
        
        with zf.open(path) as f:
            img_bytes = f.read()
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def get_random_images(self, n, image_size=256):
        """
        Extract n random images from the zip, resize to image_size x image_size.
        Returns a list of PIL Images.
        Used for FID reference computation.
        """
        indices = random.sample(range(len(self.samples)), min(n, len(self.samples)))
        images = []
        zf = self._get_zip_handle()
        
        for idx in indices:
            path, _ = self.samples[idx]
            with zf.open(path) as f:
                img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            # Center crop and resize
            w, h = img.size
            short = min(w, h)
            left = (w - short) // 2
            top = (h - short) // 2
            img = img.crop((left, top, left + short, top + short))
            img = img.resize((image_size, image_size), Image.BICUBIC)
            images.append(img)
        
        return images


def zip_worker_init_fn(worker_id):
    """
    Worker init function for DataLoader.
    Ensures each worker gets its own zip file handle.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Reset the zip handle so each worker opens its own
        dataset._zip_handle = None


# Need torch for worker_init_fn
import torch
