import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ----------------------------
# 1. Multi-Task Model
# ----------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, backbone='mobilenet_v3_small', num_classes=21):
        super().__init__()
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = base_model.features

        # Depth decoder
        self.depth_decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        )

        # Segmentation decoder
        self.segmentation_decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        features = self.encoder(x)
        depth = self.depth_decoder(features)
        segmentation = self.segmentation_decoder(features)
        return depth, segmentation


# ----------------------------
# 2. Loss Function
# ----------------------------
def multitask_loss(pred_depth, true_depth, pred_seg, true_seg):
    true_depth_resized = nn.functional.interpolate(true_depth, size=pred_depth.shape[2:], mode='bilinear', align_corners=False)
    true_seg_resized = nn.functional.interpolate(true_seg.unsqueeze(1).float(), size=pred_seg.shape[2:], mode='nearest').squeeze(1).long()

    depth_loss = nn.L1Loss()(pred_depth, true_depth_resized)
    seg_loss = nn.CrossEntropyLoss()(pred_seg, true_seg_resized)
    return depth_loss + seg_loss, depth_loss, seg_loss


# ----------------------------
# 3. Dataset
# ----------------------------
class DepthSegDataset(Dataset):
    def __init__(self, root_dir, transform=None, input_size=(256, 256)):
        self.image_paths = []
        self.depth_paths = []
        self.seg_paths = []
        self.input_size = input_size

        subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        for sub in subfolders:
            photo_files = sorted(glob.glob(os.path.join(sub, "photo", "*.jpg")))
            depth_files = sorted(glob.glob(os.path.join(sub, "depth", "*.png")))
            seg_files = sorted(glob.glob(os.path.join(sub, "instance", "*.png")))

            if not (len(photo_files) == len(depth_files) == len(seg_files)):
                raise ValueError(f"Mismatch in number of files in {sub}")

            self.image_paths.extend(photo_files)
            self.depth_paths.extend(depth_files)
            self.seg_paths.extend(seg_files)

        self.img_transform = transform or transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self.resize = transforms.Resize(self.input_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.img_transform(img)

        depth = Image.open(self.depth_paths[idx])
        depth_np = np.array(depth).astype(np.float32) / 1000.0
        depth_np = np.clip(depth_np, 0, 10)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        seg = Image.open(self.seg_paths[idx])
        seg_resized = self.resize(seg)
        seg_tensor = torch.from_numpy(np.array(seg_resized)).long()

        return img, depth_tensor, seg_tensor


# ----------------------------
# 4. Validation Function
# ----------------------------
@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss, total_depth_loss, total_seg_loss = 0, 0, 0

    for images, depths, segs in tqdm(dataloader, desc="Validating"):
        images, depths, segs = images.to(device), depths.to(device), segs.to(device)

        with torch.amp.autocast(device_type='cuda'):
            pred_depths, pred_segs = model(images)
            loss, d_loss, s_loss = multitask_loss(pred_depths, depths, pred_segs, segs)

        total_loss += loss.item()
        total_depth_loss += d_loss.item()
        total_seg_loss += s_loss.item()

    num_batches = len(dataloader)
    print(f"\nValidation Results:")
    print(f"  Avg Total Loss: {total_loss / num_batches:.4f}")
    print(f"  Avg Depth Loss: {total_depth_loss / num_batches:.4f}")
    print(f"  Avg Seg Loss  : {total_seg_loss / num_batches:.4f}")


# ----------------------------
# 5. Entry Point
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth)') 
    parser.add_argument('--input_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, required=True, help='Number of segmentation classes')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = DepthSegDataset(root_dir=args.val_dir, input_size=tuple(args.input_size))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiTaskModel(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    validate(model, val_loader, device)


if __name__ == '__main__':
    main()
