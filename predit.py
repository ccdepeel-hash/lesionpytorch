# predict.py
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from buildmodel import model, device  # same model setup

# CONFIGS
input_dir = "new_images"          # folder with leaf images to predict
output_dir = "predicted_masks"    # folder to save the mask images
checkpoint_path = "checkpoints/lesion_latest.pth"

os.makedirs(output_dir, exist_ok=True)

# LOADING TRAINED MODEL
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# IMAGE TRANSFORM (same as training)
transform_img = transforms.Compose([
    transforms.Resize((277, 394)), # from 256, 256 to (2050, 4300), to 277 x 394. (avg image size, eyeballed it)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# PREDICT LOOP
with torch.no_grad():
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Load image
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform_img(image).unsqueeze(0).to(device)

        # Predict mask
        output = model(input_tensor)['out']
        mask = torch.sigmoid(output).squeeze().cpu().numpy()

        # Threshold mask
        mask_binary = (mask > 0.5).astype(np.uint8)

        # Save mask image
        mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8))
        save_path = os.path.join(output_dir, f"mask_{filename}")
        mask_img.save(save_path)

        # --- DISPLAY OVERLAY ---
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.imshow(mask_binary, cmap='Reds', alpha=0.5)  # translucent red overlay
        plt.title(f"Predicted Mask - {filename}")
        plt.axis("off")
        plt.show()

        print(f"Saved mask: {save_path}")

print("Prediction complete for all images.")
