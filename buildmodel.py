# In VS Code: Shift+Cmd+P -> Python: Select Interpreter -> choose your Python 3.9 (same as terminal; see install.sh)
# IMPORTS
import torch
import numpy
import sys
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# dataset, dataloader - from pytorch 2.2.2. for image loading/"batching for training
# transforms from torchvision - for image processes/converting
# PIL - opens .jpg and .png files (for directory access)"

# (vvv maybe reformat this vvv)
# make sure these directories below exist for images, masks. 
# mkdir -p dataset/images | mkdir -p dataset/masks
# to see if worked, ls dataset 
# open dataset/images (folder) | drag relevant relevant images in. | open dataset/masks (folder)| put relevant images in. drag/drop

# PATHING **add on additional code here later to make adjusting directories easier
image_dir = "~/dataset/images"
mask_dir = "~/dataset/masks"

# TRANSFORMS
# clarify values and maybe need to adjust? purpose of them
img_size = (2050, 4300)

transform_img = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])
# with established directories, this should "set the target size" of the images to 256x256 pixels and convert them to tensors.
# it is preparing masks as tensors for segmentation. does not run yet, only defines how it will be processed later... (?)

# DEFINING DATASET
class LesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        
        assert len(self.image_filenames) == len(self.mask_filenames), "Images and masks counts do not match"
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask
        
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        
        return image, mask
    
#testing

# DATALOADER + MODEL SETUP
# importing models sub-package. move to top of code?
from torchvision import models
import torch.nn as nn

# Ensure directories expand correctly (important for "~")
#**clarify further; needed?
image_dir = os.path.expanduser(image_dir)
mask_dir = os.path.expanduser(mask_dir)

# Initialize dataset and dataloader
dataset = LesionDataset(image_dir, mask_dir, transform_img, transform_mask)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Check a batch (optional but highly recommended)
images, masks = next(iter(dataloader))
print(f"Image batch shape: {images.shape}")
print(f"Mask batch shape: {masks.shape}")

# MODEL DEFINITION
# Segementation network. Using pretrained model from torchvision (DeepLabV3 (DL3) with a ResNet-50 backbone)
# creates and loads DeepLabV3 neural network (pretrained weights)
model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

# Adjust classifier for single output class (lesion vs background)
# "The pretrained model has 21 classes (for COCO/VOC), we change it to 1"
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Optional: also adjust auxiliary classifier if using it
if model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Move to GPU if available; for HPC cluster... 
# may need to tweak install so it installs GPU version of pytorch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model moved to: {device}")

# LOSS FUNCTION + OPTIMIZER SETUP
# "For segmentation with binary masks, we use BCEWithLogitsLoss (binary cross entropy with sigmoid)"
# model attempting to predict segmentation mask from existing image_dir
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TEST FORWARD PASS 
# checking that everything runs. send input image/masks to same device as model. (CPU/GPU/MPS)
images = images.to(device)
masks = masks.to(device)

# torch.no_grad(): tells pytorch not to compute gradients. Not training yet, just testing (memory save)
# sending images through DL3; out is main output in DL3
with torch.no_grad():
    outputs = model(images)['out']
    print(f"Output shape: {outputs.shape}")  # should be [batch_size, 1, 256, 256]

# SAVE CHECKPOINT SETUP
# folder creation if it doesn't exist. saving model's weights (learned parameters) into file inside folder
# if model crashes during training, can reload model with files in folder... no need to retrain
# .pth file used to give model without re-training it. these are weight values, not output.
# "torch.save(model.state)dict(), f"checkpoints/lesion_epoch{epoch+1}.pth")" used if we want new checkpoint for every epoch/full pass.
# script overwritten every time with new values after each run.
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/lesion_model_init.pth")
print("Initial model saved successfully.")
# ~/python--pytorch/lesionML/checkpoints for location of .pth 

# make objects importable by other scripts
__all__ = ["model", "dataloader", "criterion", "optimizer", "device"]