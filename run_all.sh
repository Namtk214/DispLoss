#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Pre-computing VAE latents..."
echo "=========================================="

cd /workspace/DispLoss

python3 precompute_latents.py \
  --batch-size 128 \
  --num-workers 12 \
  --output-dir /workspace/DispLoss/latents

echo ""
echo "=========================================="
echo "Step 2: Starting training with latents..."
echo "=========================================="

rm -rf /workspace/DispLoss/results/00*-SiT-B-2-* 2>/dev/null

WANDB_KEY=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 \
ENTITY=Fingerprint_Recognition \
PROJECT=SiT-DispLoss \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
  --model SiT-B/2 \
  --latent-dir /workspace/DispLoss/latents \
  --global-batch-size 128 \
  --micro-batch-size 128 \
  --image-size 256 \
  --epochs 400 \
  --num-sampling-steps 50 \
  --eval-every 25000 \
  --num-fid-samples 4096 \
  --ckpt-every 100000 \
  --sample-every 2000 \
  --log-every 100 \
  --num-workers 0 \
  --wandb
