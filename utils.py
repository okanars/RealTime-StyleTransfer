import torch
import numpy as np
from PIL import Image

def gram_matrix(y):
    """
    Calculates the Gram Matrix (correlation between feature maps).
    Result is (batch, channel, channel)
    """
    (b, c, h, w) = y.size()
    features = y.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.LANCZOS)
    return img

def normalize_batch(batch):
    # Normalize using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a Numpy array for OpenCV/Display.
    Reverses the ImageNet normalization.
    """
    # Clone to avoid modifying original
    image = tensor.clone().detach().cpu()
    
    # Remove batch dim if present (1, C, H, W) -> (C, H, W)
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # Denormalize (Reverse of (x - mean) / std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    # Clamp values to [0, 1]
    image = image.clamp(0, 1)
    
    # (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).numpy()
    
    # [0, 1] -> [0, 255]
    image = (image * 255).astype(np.uint8)
    
    return image

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")