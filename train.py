import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# ========================
# Dataset Definitions
# ========================
class DepthDataset(Dataset):
    def __init__(self, img_dir, depth_dir, transform=None):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform
        img_files = set(os.listdir(img_dir))
        depth_files = set(os.listdir(depth_dir))
        self.images = sorted(list(img_files & depth_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform:
            img = self.transform(img)
        depth = T.Resize((128, 128))(depth)
        depth = T.ToTensor()(depth)
        return img, depth.squeeze(0)


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_size=(128, 128), num_classes=34):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        self.images = sorted(list(img_files & mask_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            img = self.transform(img)
            
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        
        if mask.max() >= self.num_classes:
            print(f"Warning: Mask contains class {mask.max()} (num_classes={self.num_classes})")
            mask = np.clip(mask, 0, self.num_classes-1)
            
        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


# ========================
# Model Definition
# ========================
class SimpleMultiTaskNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.depth_head = nn.Conv2d(32, 1, 1)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        feat = self.encoder(x)
        return {
            "depth": self.depth_head(feat),
            "seg": self.seg_head(feat)
        }


# ========================
# Metrics
# ========================
def compute_depth_mae(pred, target):
    """Compute Mean Absolute Error for depth prediction"""
    return torch.mean(torch.abs(pred - target)).item()


def compute_seg_accuracy(pred, target):
    """Compute pixel accuracy for segmentation"""
    pred = torch.argmax(pred, dim=1)
    return (pred == target).float().mean().item()


# ========================
# Training Function
# ========================
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 34  # Based on your max class index of 33

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    # Load datasets
    depth_train = DepthDataset(
        "multi_task_dataset/depth_dataset/images/train",
        "multi_task_dataset/depth_dataset/depth/train",
        transform
    )
    depth_val = DepthDataset(
        "multi_task_dataset/depth_dataset/images/val",
        "multi_task_dataset/depth_dataset/depth/val",
        transform
    )
    seg_train = SegmentationDataset(
        "multi_task_dataset/segmentation_dataset/images/train",
        "multi_task_dataset/segmentation_dataset/masks/train",
        transform,
        target_size=(128, 128),
        num_classes=NUM_CLASSES
    )
    seg_val = SegmentationDataset(
        "multi_task_dataset/segmentation_dataset/images/val",
        "multi_task_dataset/segmentation_dataset/masks/val",
        transform,
        target_size=(128, 128),
        num_classes=NUM_CLASSES
    )

    # Verify datasets
    print(f"Depth Train: {len(depth_train)} images")
    print(f"Depth Val: {len(depth_val)} images")
    print(f"Seg Train: {len(seg_train)} images")
    print(f"Seg Val: {len(seg_val)} images")

    # DataLoaders
    depth_loader_train = DataLoader(depth_train, batch_size=4, shuffle=True)
    seg_loader_train = DataLoader(seg_train, batch_size=4, shuffle=True)
    depth_loader_val = DataLoader(depth_val, batch_size=4)
    seg_loader_val = DataLoader(seg_val, batch_size=4)

    # Model setup
    model = SimpleMultiTaskNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    depth_loss_fn = nn.L1Loss()
    seg_loss_fn = nn.CrossEntropyLoss()

    best_depth_mae = float('inf')
    best_seg_acc = 0.0

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        depth_iter = iter(depth_loader_train)
        seg_iter = iter(seg_loader_train)
        steps = min(len(depth_loader_train), len(seg_loader_train))
        total_loss = 0

        try:
            for _ in tqdm(range(steps), desc="Training"):
                optimizer.zero_grad()

                # Depth
                x_d, y_d = next(depth_iter)
                x_d, y_d = x_d.to(device), y_d.to(device)
                out_d = model(x_d)["depth"]
                out_d = F.interpolate(out_d, size=y_d.shape[-2:], mode='bilinear', align_corners=False)
                loss_d = depth_loss_fn(out_d.squeeze(1), y_d)

                # Segmentation
                x_s, y_s = next(seg_iter)
                x_s, y_s = x_s.to(device), y_s.to(device)
                out_s = model(x_s)["seg"]
                out_s_resized = F.interpolate(out_s, size=y_s.shape[-2:], mode='bilinear', align_corners=False)
                loss_s = seg_loss_fn(out_s_resized, y_s)

                loss = loss_d + loss_s
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        except StopIteration:
            print("Warning: Uneven dataset sizes caused early termination")
            pass

        scheduler.step()
        print(f"Avg Train Loss: {total_loss / steps:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            # Depth MAE
            mae_total = 0
            for x, y in depth_loader_val:
                x, y = x.to(device), y.to(device)
                pred = model(x)["depth"]
                pred = F.interpolate(pred, size=y.shape[-2:], mode='bilinear', align_corners=False)
                mae_total += compute_depth_mae(pred.squeeze(1), y)
            depth_mae = mae_total / len(depth_loader_val)
            print(f"Depth MAE: {depth_mae:.4f}")

            # Segmentation Accuracy
            acc_total = 0
            for x, y in seg_loader_val:
                x, y = x.to(device), y.to(device)
                pred = model(x)["seg"]
                pred_resized = F.interpolate(pred, size=y.shape[-2:], mode='bilinear', align_corners=False)
                acc_total += compute_seg_accuracy(pred_resized, y)
            seg_acc = acc_total / len(seg_loader_val)
            print(f"Segmentation Accuracy: {seg_acc:.4f}")

            # Save checkpoints
            if depth_mae < best_depth_mae:
                best_depth_mae = depth_mae
                torch.save(model.state_dict(), "best_model_depth.pth")
                print("✅ Saved new best depth model")

            if seg_acc > best_seg_acc:
                best_seg_acc = seg_acc
                torch.save(model.state_dict(), "best_model_segmentation.pth")
                print("✅ Saved new best segmentation model")


if __name__ == "__main__":
    run()