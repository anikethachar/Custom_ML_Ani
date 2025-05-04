import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Set environment variable to avoid OMP runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
# Inspect Checkpoint
# ========================
def inspect_checkpoint(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # Add weights_only=True
    print(checkpoint.keys())  # Check the keys available in the checkpoint
    return checkpoint

# ========================
# Load Trained Model
# ========================
def load_model(num_classes=34, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = SimpleMultiTaskNet(num_classes=num_classes).to(device)
    
    # Load both checkpoints and combine
    depth_checkpoint = inspect_checkpoint("best_model_depth.pth", device=device)
    seg_checkpoint = inspect_checkpoint("best_model_segmentation.pth", device=device)
    
    # If the checkpoint contains the entire model state_dict, load it directly
    if 'depth_head' in depth_checkpoint:
        model.depth_head.load_state_dict(depth_checkpoint['depth_head'])
    else:
        model.load_state_dict(depth_checkpoint)  # Adjust this based on the structure of your checkpoint
    
    if 'seg_head' in seg_checkpoint:
        model.seg_head.load_state_dict(seg_checkpoint['seg_head'])
    else:
        model.load_state_dict(seg_checkpoint)  # Adjust this based on the structure of your checkpoint
    
    model.eval()
    return model

# ========================
# Preprocessing
# ========================
def preprocess_image(image_path, size=(128, 128)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Add batch dimension

# ========================
# Visualization
# ========================
def visualize_results(image, depth_pred, seg_pred, num_classes=34):
    plt.figure(figsize=(18, 6))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title("Input Image")
    plt.axis('off')
    
    # Depth Prediction
    plt.subplot(1, 3, 2)
    depth_display = depth_pred.squeeze().cpu().numpy()
    plt.imshow(depth_display, cmap='viridis')
    plt.colorbar()
    plt.title("Depth Prediction")
    plt.axis('off')
    
    # Segmentation Prediction
    plt.subplot(1, 3, 3)
    seg_display = seg_pred.argmax(dim=1).squeeze().cpu().numpy()
    plt.imshow(seg_display, cmap='tab20', vmin=0, vmax=num_classes-1)
    plt.colorbar(ticks=range(num_classes))
    plt.title("Segmentation Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ========================
# Main Testing Function
# ========================
def test_on_image(image_path, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Preprocess
    image_tensor = preprocess_image(image_path).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        depth_pred = F.interpolate(outputs['depth'], size=image_tensor.shape[-2:], mode='bilinear')
        seg_pred = F.interpolate(outputs['seg'], size=image_tensor.shape[-2:], mode='bilinear')
    
    # Visualize
    visualize_results(image_tensor, depth_pred, seg_pred)

    return depth_pred, seg_pred

# ========================
# Run the Test
# ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    # 1. Load your trained model
    model = load_model(num_classes=34, device=device)
    
    # 2. Specify path to your test image - use raw string or forward slashes
    test_image_path = r"E:\Project\Depth\Custom\multi_task_dataset\test\0000000103.png"  # Raw string
    
    # 3. Run the test
    depth_pred, seg_pred = test_on_image(test_image_path, model, device)
    
    # 4. Optionally save the results (depth and segmentation predictions)
    torch.save({
        'depth_pred': depth_pred,
        'seg_pred': seg_pred,
    }, "test_results.pth")
