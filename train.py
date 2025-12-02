import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import TransformerNet, Vgg16
from dataset import StyleTransferDataset
from utils import normalize_batch, gram_matrix, get_device

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def train(args):
    device = get_device()
    
    # Dataset & Dataloader
    train_dataset = StyleTransferDataset(args.content_dir, args.style_image, args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Models
    transformer = TransformerNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    
    optimizer = optim.Adam(transformer.parameters(), args.lr)
    mse_loss = nn.MSELoss()

    # Pre-calculate Gram Matrix for the Style Image
    # We take the style image from the first item in dataset (since it's constant)
    style_tensor = train_dataset.style_image.unsqueeze(0).to(device)
    style_features = vgg(style_tensor)
    style_gram = [gram_matrix(y) for y in style_features]

    for epoch in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x) # Generated Image

            # Normalize for VGG (ImageNet stats are expected by Vgg16)
            # Since our input x is already normalized in dataset, we might need to be careful.
            # However, TransformerNet output is raw. Let's assume dataset gives normalized x.
            # We should probably feed normalized x and y to VGG.
            # But Transformer outputs need to be normalized before VGG?
            # Let's simplify: VGG expects normalized input.
            
            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            # Content Loss (relu2_2 is good for content)
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Style Loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, style_gram):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1} [{count}/{len(train_dataset)}] "
                      f"Content: {agg_content_loss / (batch_id + 1):.2f} "
                      f"Style: {agg_style_loss / (batch_id + 1):.2f}")

    # Save Model
    transformer.eval()
    save_path = os.path.join(args.save_model_dir, "model.pth")
    torch.save(transformer.state_dict(), save_path)
    print(f"Model saved at {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for training")
    parser.add_argument("--content-dir", type=str, required=True, help="path to content images")
    parser.add_argument("--style-image", type=str, required=True, help="path to style image")
    parser.add_argument("--save-model-dir", type=str, default="./checkpoints", help="path to save model")
    parser.add_argument("--checkpoint-model-dir", type=str, default="./checkpoints", help="path to save checkpoint")
    parser.add_argument("--image-size", type=int, default=256, help="training image size")
    parser.add_argument("--style-weight", type=float, default=1e5, help="weight for style-loss")
    parser.add_argument("--content-weight", type=float, default=1e0, help="weight for content-loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--log-interval", type=int, default=50, help="number of batches to print")

    args = parser.parse_args()
    check_paths(args)
    train(args)

if __name__ == "__main__":
    main()