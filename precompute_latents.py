"""
Pre-compute VAE latents for all ImageNet images.
Can run alongside training using minimal GPU memory (batch_size=4).
Saves latents as shards to /workspace/DispLoss/latents/
"""
import torch
import numpy as np
import os
import sys
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.models import AutoencoderKL
from zip_dataset import ZipImageFolder, zip_worker_init_fn
from PIL import Image

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=str, default="/root/imagenet-object-localization-challenge.zip")
    parser.add_argument("--output-dir", type=str, default="/workspace/DispLoss/latents")
    parser.add_argument("--batch-size", type=int, default=8, help="Small batch to avoid OOM when running alongside training")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=10000, help="Number of samples per shard file")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if resuming
    existing_shards = sorted([f for f in os.listdir(args.output_dir) if f.startswith("shard_") and f.endswith(".pt")])
    start_idx = 0
    if existing_shards:
        last_shard = existing_shards[-1]
        # shard_XXXXXX.pt -> XXXXXX is the start index
        last_start = int(last_shard.split("_")[1].split(".")[0])
        last_data = torch.load(os.path.join(args.output_dir, last_shard), weights_only=True)
        start_idx = last_start + len(last_data["latents"])
        print(f"Resuming from index {start_idx} (found {len(existing_shards)} existing shards)")

    # Setup transform (same as training)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ZipImageFolder(args.zip_path, prefix='ILSVRC/Data/CLS-LOC/train', transform=transform)
    
    # Use a subset starting from start_idx
    if start_idx > 0:
        from torch.utils.data import Subset
        indices = list(range(start_idx, len(dataset)))
        dataset_subset = Subset(dataset, indices)
    else:
        dataset_subset = dataset

    loader = DataLoader(
        dataset_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=zip_worker_init_fn,
    )

    # Load VAE
    print(f"Loading VAE (sd-vae-ft-{args.vae})...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(args.device)
    vae.eval()
    vae.requires_grad_(False)

    # Process
    all_latents = []
    all_labels = []
    global_idx = start_idx
    shard_count = len(existing_shards)
    total = len(dataset)
    
    t0 = time.time()
    
    print(f"Processing {total - start_idx} images...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            
            # VAE encode
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                latents = vae.encode(images).latent_dist.sample().mul_(0.18215)
            
            all_latents.append(latents.cpu().half())  # save as fp16
            all_labels.append(labels)
            global_idx += len(images)

            # Save shard when accumulated enough
            if len(all_latents) * args.batch_size >= args.shard_size or global_idx >= total:
                latents_cat = torch.cat(all_latents, dim=0)
                labels_cat = torch.cat(all_labels, dim=0)
                
                shard_start = global_idx - len(latents_cat)
                shard_path = os.path.join(args.output_dir, f"shard_{shard_start:07d}.pt")
                torch.save({"latents": latents_cat, "labels": labels_cat}, shard_path)
                
                elapsed = time.time() - t0
                speed = global_idx - start_idx
                eta = (total - global_idx) / (speed / elapsed) if speed > 0 else 0
                
                shard_count += 1
                print(f"[Shard {shard_count}] Saved {shard_path} | "
                      f"{global_idx}/{total} ({100*global_idx/total:.1f}%) | "
                      f"{speed/elapsed:.1f} img/s | ETA: {eta/60:.0f} min")
                
                all_latents = []
                all_labels = []
                torch.cuda.empty_cache()

    # Save metadata
    meta = {"total": total, "num_shards": shard_count, "image_size": args.image_size, "shard_size": args.shard_size}
    torch.save(meta, os.path.join(args.output_dir, "metadata.pt"))
    print(f"\nDone! Saved {shard_count} shards to {args.output_dir}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
