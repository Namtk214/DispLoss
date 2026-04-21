"""
FID evaluation module for SiT training.
Computes FID between generated samples and real ImageNet images.
"""
import torch
import numpy as np
from scipy import linalg
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


def get_inception_model(device):
    """Load InceptionV3 model for FID computation."""
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final FC, get 2048-d features
    model.eval()
    model.to(device)
    return model


_inception_model = None

def _get_cached_inception(device):
    """Cache inception model to avoid reloading every eval."""
    global _inception_model
    if _inception_model is None:
        _inception_model = get_inception_model(device)
    return _inception_model


def compute_inception_features(images_tensor, device, batch_size=64):
    """
    Compute InceptionV3 features for a batch of images.
    
    Args:
        images_tensor: (N, 3, H, W) tensor, values in [-1, 1]
        device: torch device
        batch_size: batch size for inception forward pass
    
    Returns:
        features: (N, 2048) numpy array
    """
    model = _get_cached_inception(device)
    
    # Inception expects 299x299 input, values in [0, 1] approx
    resize = transforms.Resize((299, 299), antialias=True)
    
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i+batch_size].to(device)
            # Convert from [-1, 1] to [0, 1]
            batch = (batch + 1) / 2.0
            batch = resize(batch)
            features = model(batch)
            if isinstance(features, tuple):
                features = features[0]
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)


def compute_fid_from_features(real_features, fake_features):
    """
    Compute FID given two sets of Inception features.
    
    Args:
        real_features: (N, 2048) numpy array
        fake_features: (N, 2048) numpy array
    
    Returns:
        fid_score: float
    """
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    diff = mu_real - mu_fake
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logger.warning(f"FID: Imaginary component {m}")
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


@torch.no_grad()
def generate_fid_samples(
    ema_model, 
    vae, 
    transport_sampler, 
    n_samples, 
    latent_size, 
    device, 
    num_classes=1000,
    num_steps=50, 
    cfg_scale=4.0,
    batch_size=32,
):
    """
    Generate samples for FID evaluation using the EMA model.
    
    Args:
        ema_model: EMA model in eval mode
        vae: VAE decoder
        transport_sampler: Sampler object from transport module
        n_samples: number of images to generate
        latent_size: latent spatial size (e.g., 32 for 256x256 images)
        device: torch device
        num_classes: number of classes
        num_steps: number of ODE sampling steps
        cfg_scale: classifier-free guidance scale
        batch_size: per-batch generation size
    
    Returns:
        all_samples: (N, 3, 256, 256) tensor in [-1, 1]
    """
    sample_fn = transport_sampler.sample_ode(
        sampling_method="dopri5",
        num_steps=num_steps,
        atol=1e-6,
        rtol=1e-3,
    )
    
    use_cfg = cfg_scale > 1.0
    all_samples = []
    generated = 0
    
    while generated < n_samples:
        current_batch = min(batch_size, n_samples - generated)
        
        z = torch.randn(current_batch, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (current_batch,), device=device)
        
        if use_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([num_classes] * current_batch, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)
            model_fn = ema_model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = ema_model.forward
        
        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        
        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)
        
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(samples, -1, 1)
        all_samples.append(samples.cpu())
        generated += current_batch
        logger.info(f"  Generated {generated}/{n_samples} samples for FID")
    
    return torch.cat(all_samples, dim=0)[:n_samples]


def real_images_to_tensor(pil_images):
    """Convert list of PIL images to tensor in [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensors = [transform(img) for img in pil_images]
    return torch.stack(tensors)


def evaluate_fid(
    ema_model,
    vae,
    transport_sampler,
    n_samples,
    latent_size,
    device,
    zip_path="/root/imagenet-object-localization-challenge.zip",
    image_size=256,
    num_steps=50,
    cfg_scale=4.0,
    gen_batch_size=32,
):
    """
    Full FID evaluation pipeline.
    
    Args:
        ema_model: EMA model
        vae: VAE decoder
        transport_sampler: Sampler from transport module
        n_samples: number of samples for FID
        latent_size: latent spatial size
        device: torch device
        zip_path: path to ImageNet zip for real images
        image_size: output image size
        num_steps: ODE sampling steps
        cfg_scale: CFG scale
        gen_batch_size: batch size for generation
    
    Returns:
        fid_score: float
    """
    logger.info(f"Starting FID evaluation with {n_samples} samples...")
    
    # 1. Generate fake images
    logger.info("Generating fake images...")
    fake_images = generate_fid_samples(
        ema_model, vae, transport_sampler,
        n_samples=n_samples,
        latent_size=latent_size,
        device=device,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        batch_size=gen_batch_size,
    )
    
    # 2. Get real images from zip
    logger.info("Extracting real images from zip...")
    from zip_dataset import ZipImageFolder
    real_ds = ZipImageFolder(zip_path, prefix='ILSVRC/Data/CLS-LOC/train', transform=None)
    real_pil_images = real_ds.get_random_images(n_samples, image_size=image_size)
    real_images = real_images_to_tensor(real_pil_images)
    
    # 3. Compute inception features
    logger.info("Computing inception features for fake images...")
    fake_features = compute_inception_features(fake_images, device)
    
    logger.info("Computing inception features for real images...")
    real_features = compute_inception_features(real_images, device)
    
    # 4. Compute FID
    fid_score = compute_fid_from_features(real_features, fake_features)
    logger.info(f"FID score: {fid_score:.4f}")
    
    return fid_score

