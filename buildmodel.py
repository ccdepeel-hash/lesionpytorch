# buildmodel.py
# In VS Code: Shift+Cmd+P -> Python: Select Interpreter -> choose your Python 3.11 (plantlesion_gpu2 env)

# IMPORTS
import torch
import numpy
import sys
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
# dataset, dataloader - from pytorch 2.2.2. for image loading/"batching for training
# transforms from torchvision - for image processes/converting
# PIL - opens .jpg and .png files (for directory access)
# random - for synchronized augmentation transforms

# PATHING
# Hardcoded for cluster - works for all users with ~ expansion
image_dir = "~/kliebengrp/input/images"
mask_dir = "~/kliebengrp/input/masks"

# Ensure directories expand correctly (important for "~")
image_dir = os.path.expanduser(image_dir)
mask_dir = os.path.expanduser(mask_dir)

# Image size - reduced from 4300x2700 to single leaf size
img_size = (277, 394)

# DEFINING DATASET WITH AUGMENTATION
# Updated class with synchronized transforms to artificially expand dataset
# Images and masks get same geometric transforms (flips, rotations)
# Only images get color transforms (masks stay binary)
class LesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(277, 394), augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        # Filter out hidden files like .DS_Store
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        
        assert len(self.image_filenames) == len(self.mask_filenames), "Images and masks counts do not match"
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        # Open images as RGB and masks as grayscale
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask
        
        # Resize to target size
        image = transforms.Resize(self.img_size)(image)
        mask = transforms.Resize(self.img_size)(mask)
        
        # Apply synchronized augmentations
        # These create random variations each epoch to prevent overfitting
        if self.augment:
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random vertical flip (50% chance)
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # Random rotation (same angle for both image and mask)
            angle = random.uniform(-45, 45)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
            # Color jitter (only on image, not mask!)
            # Varies brightness, contrast, saturation, hue to simulate different lighting
            image = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )(image)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        
        # Normalize image only (using ImageNet stats)
        # These mean/std values are standard for pretrained models
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(image)
        
        return image, mask

# DATALOADER + MODEL SETUP
# importing models sub-package for DeepLabV3
from torchvision import models
import torch.nn as nn

# Initialize dataset and dataloader with augmentation enabled
dataset = LesionDataset(image_dir, mask_dir, img_size=img_size, augment=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # 16 images / 4 batches = 4 steps per epoch

# Check a batch (optional but highly recommended)
images, masks = next(iter(dataloader))
print(f"Image batch shape: {images.shape}")
print(f"Mask batch shape: {masks.shape}")

# MODEL DEFINITION
# Segmentation network. Using pretrained model from torchvision (DeepLabV3 (DL3) with a ResNet-50 backbone)
# creates and loads DeepLabV3 neural network (pretrained weights from ImageNet/COCO)
model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

# Adjust classifier for single output class (lesion vs background)
# The pretrained model has 21 classes (for COCO/VOC), we change it to 1
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Optional: also adjust auxiliary classifier if using it
if model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Move to GPU if available
# For HPC cluster - should detect CUDA GPU automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model moved to: {device}")

# LOSS FUNCTION + OPTIMIZER SETUP
# For segmentation with binary masks, we use BCEWithLogitsLoss (binary cross entropy with sigmoid)
# Model attempting to predict segmentation mask from existing image_dir
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TEST FORWARD PASS
# Checking that everything runs. Send input image/masks to same device as model. (CPU/GPU)
images = images.to(device)
masks = masks.to(device)

# torch.no_grad(): tells pytorch not to compute gradients. Not training yet, just testing (memory save)
# Sending images through DL3; 'out' is main output in DL3
with torch.no_grad():
    outputs = model(images)['out']
    print(f"Output shape: {outputs.shape}")  # should be [batch_size, 1, 277, 394]

# SAVE CHECKPOINT SETUP
# Folder creation if it doesn't exist. Saving model's weights (learned parameters) into file inside folder
# If model crashes during training, can reload model with files in folder... no need to retrain
# .pth file used to give model without re-training it. These are weight values, not output.
# This is just the INITIAL checkpoint - actual training checkpoints saved by runmodel.py
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/lesion_model_init.pth")
print("Initial model saved successfully.")

# make objects importable by other scripts
# runmodel.py imports these to continue training
__all__ = ["model", "dataloader", "criterion", "optimizer", "device"]