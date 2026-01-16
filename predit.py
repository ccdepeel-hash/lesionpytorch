# predict.py
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

# CONFIGS
input_dir = os.path.expanduser("~/kliebengrp/new_images")          # folder with leaf images to predict
output_dir = os.path.expanduser("~/kliebengrp/predicted_masks")   # folder to save the mask images
checkpoint_path = os.path.expanduser("~/kliebengrp/checkpoints/lesion_latest.pth")

os.makedirs(output_dir, exist_ok=True)

# SET UP MODEL ARCHITECTURE (same as buildmodel.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet50(weights=None)  # No pretrained weights

# Adjust classifier for single output class (lesion vs background)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Adjust auxiliary classifier
if model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

model = model.to(device)

# LOAD TRAINED WEIGHTS
print(f"Loading trained model from {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"Model loaded successfully on {device}")

# IMAGE TRANSFORM (same as training)
transform_img = transforms.Compose([
    transforms.Resize((277, 394)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# PREDICT LOOP
print(f"Looking for images in {input_dir}")
image_files = sorted([f for f in os.listdir(input_dir) 
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])

if not image_files:
    print(f"No images found in {input_dir}")
    exit()

print(f"Found {len(image_files)} images to process")

with torch.no_grad():
    for filename in image_files:
        # Load image
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # Save original size for later
        
        input_tensor = transform_img(image).unsqueeze(0).to(device)
        
        # Predict mask
        output = model(input_tensor)['out']
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Threshold mask (0.5 threshold for binary classification)
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Save binary mask image
        mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8))
        mask_save_path = os.path.join(output_dir, f"mask_{filename}")
        mask_img.save(mask_save_path)
        
        # Create and save overlay visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original: {filename}")
        axes[0].axis("off")
        
        # Predicted mask
        axes[1].imshow(mask_binary, cmap='gray')
        axes[1].set_title("Predicted Mask")
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
        
        print(f"✓ Processed {filename}")
        print(f"  - Mask saved: {mask_save_path}")
        print(f"  - Overlay saved: {overlay_path}")

print(f"\nPrediction complete! All results saved to: {output_dir}")
print(f"\nTo download results to your local machine, use scp:")
print(f"scp -r ccdepeel@farm.hpc.ucdavis.edu:{output_dir} ~/Desktop/")