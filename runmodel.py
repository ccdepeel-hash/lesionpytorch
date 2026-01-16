# runmodel.py
import torch
from buildmodel import model, dataloader, criterion, optimizer, device
import os

# TRAINING CONFIGURATION
num_epochs = 30
save_dir = os.path.expanduser("~/kliebengrp/checkpoints")
os.makedirs(save_dir, exist_ok=True)

# LOAD PREVIOUS CHECKPOINT IF IT EXISTS
checkpoint_path = os.path.join(save_dir, "lesion_latest.pth")
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    print("Resuming training from previous checkpoint!")
else:
    print("No checkpoint found, starting from pretrained weights")

# TRAINING LOOP
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# SAVE CHECKPOINT (atomic write to avoid corruption)
temp_path = os.path.join(save_dir, "lesion_model_tmp.pth")
final_path = os.path.join(save_dir, "lesion_latest.pth")
torch.save(model.state_dict(), temp_path)
os.replace(temp_path, final_path)
print(f"Model saved to {final_path}")

print("Training complete.")