import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_image_path, image_size=256, augment=True):
        self.content_dir = content_dir
        self.image_files = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Transforms for Content Images
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and Transform Style Image (We do this once)
        self.style_image = self.load_style_image(style_image_path, image_size)

    def load_style_image(self, path, size):
        img = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img) # Returns (C, H, W)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.content_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')
            image = self.transform(image)
            # Return both content image and the fixed style image
            return image, self.style_image
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a dummy tensor or handle gracefully
            return torch.zeros(3, 256, 256), self.style_image