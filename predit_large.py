# predict_large.py
# Predicts lesions on large images (e.g., 2400x4300) using sliding window approach
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend; no known GUI on cluster. 
import matplotlib.pyplot as plt
import numpy as np

# CONFIGS
input_dir = os.path.expanduser("~/kliebengrp/new_images_large")  # folder with large images
output_dir = os.path.expanduser("~/kliebengrp/predicted_masks_large")  # output folder
checkpoint_path = os.path.expanduser("~/kliebengrp/checkpoints/lesion_latest.pth")

# Sliding window parameters
PATCH_SIZE = (277, 394)  # Size the model was trained on
OVERLAP = 0.2  # 20% overlap between patches to avoid edge artifacts
STRIDE = (int(PATCH_SIZE[0] * (1 - OVERLAP)), int(PATCH_SIZE[1] * (1 - OVERLAP)))

os.makedirs(output_dir, exist_ok=True)

# SET UP MODEL ARCHITECTURE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

# Adjust classifier for single output class
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
if model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

model = model.to(device)

# LOAD TRAINED WEIGHTS
print(f"Loading trained model from {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"Model loaded successfully on {device}")

# IMAGE TRANSFORM (same as training)
transform_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_patches(image, patch_size, stride):
    """
    Extract overlapping patches from a large image.
    Returns: list of (patch, top_left_coords)
    """
    patches = []
    h, w = image.size[1], image.size[0]  # PIL uses (width, height)
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    for top in range(0, h - patch_h + 1, stride_h):
        for left in range(0, w - patch_w + 1, stride_w):
            box = (left, top, left + patch_w, top + patch_h)
            patch = image.crop(box)
            patches.append((patch, (top, left)))
    
    # Handle right and bottom edges if image doesn't divide evenly
    # Add patches from the very right edge
    if w % stride_w != 0:
        for top in range(0, h - patch_h + 1, stride_h):
            left = w - patch_w
            box = (left, top, w, top + patch_h)
            patch = image.crop(box)
            patches.append((patch, (top, left)))
    
    # Add patches from the very bottom edge
    if h % stride_h != 0:
        for left in range(0, w - patch_w + 1, stride_w):
            top = h - patch_h
            box = (left, top, left + patch_w, h)
            patch = image.crop(box)
            patches.append((patch, (top, left)))
    
    # Bottom-right corner
    if (h % stride_h != 0) and (w % stride_w != 0):
        box = (w - patch_w, h - patch_h, w, h)
        patch = image.crop(box)
        patches.append((patch, (h - patch_h, w - patch_w)))
    
    return patches

def stitch_predictions(predictions, image_size, patch_size, stride):
    """
    Stitch overlapping patch predictions into a full-size mask.
    Uses averaging for overlapping regions.
    """
    h, w = image_size
    patch_h, patch_w = patch_size
    
    # Create accumulation arrays
    full_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)
    
    for pred_mask, (top, left) in predictions:
        # Add prediction to accumulator
        full_mask[top:top+patch_h, left:left+patch_w] += pred_mask
        count_mask[top:top+patch_h, left:left+patch_w] += 1
    
    # Average overlapping predictions
    full_mask = np.divide(full_mask, count_mask, where=count_mask > 0)
    
    return full_mask

# PREDICT LOOP FOR LARGE IMAGES
print(f"Looking for images in {input_dir}")
image_files = sorted([f for f in os.listdir(input_dir) 
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])

if not image_files:
    print(f"No images found in {input_dir}")
    exit()

print(f"Found {len(image_files)} images to process")

with torch.no_grad():
    for filename in image_files:
        print(f"\nProcessing {filename}...")
        
        # Load large image
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("RGB")
        original_size = (image.size[1], image.size[0])  # (height, width)
        
        print(f"  Image size: {image.size[0]}x{image.size[1]}")
        
        # Extract patches
        patches = extract_patches(image, PATCH_SIZE, STRIDE)
        print(f"  Extracted {len(patches)} patches")
        
        # Predict on each patch
        predictions = []
        for i, (patch, coords) in enumerate(patches):
            if (i + 1) % 10 == 0:
                print(f"    Processing patch {i+1}/{len(patches)}...")
            
            # Transform and predict
            input_tensor = transform_patch(patch).unsqueeze(0).to(device)
            output = model(input_tensor)['out']
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
            predictions.append((mask, coords))
        
        # Stitch predictions back together
        print(f"  Stitching {len(predictions)} predictions...")
        full_mask = stitch_predictions(predictions, original_size, PATCH_SIZE, STRIDE)
        
        # Threshold mask
        mask_binary = (full_mask > 0.5).astype(np.uint8)
        
        # Save binary mask
        mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8))
        mask_save_path = os.path.join(output_dir, f"mask_{filename}")
        mask_img.save(mask_save_path)
        
        # Create and save overlay visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original: {filename}\n({image.size[0]}x{image.size[1]})")
        axes[0].axis("off")
        
        # Predicted mask
        axes[1].imshow(mask_binary, cmap='gray')
        axes[1].set_title(f"Predicted Mask\n({len(patches)} patches processed)")
        axes[1].axis("off")
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(mask_binary, cmap='Reds', alpha=0.5)
        axes[2].set_title("Overlay (Red = Lesions)")
        axes[2].axis("off")
        
        # Save overlay figure
        overlay_path = os.path.join(output_dir, f"overlay_{filename}")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Mask saved: {mask_save_path}")
        print(f"  ✓ Overlay saved: {overlay_path}")

print(f"\n{'='*60}")
print(f"Prediction complete! All results saved to: {output_dir}")
print(f"{'='*60}")
print(f"\nTo download results to your local machine:")
print(f"scp -r ccdepeel@farm.cse.ucdavis.edu:{output_dir} ~/Desktop/")