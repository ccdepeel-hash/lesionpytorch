# runmodel.py
import torch
from buildmodel import model, dataloader, criterion, optimizer, device
import os

# TRAINING CONFIGURATION
num_epochs = 10
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

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

    # inside your training loop
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # after each epoch
    temp_path = os.path.join(save_dir, "lesion_model_tmp.pth")
    final_path = os.path.join(save_dir, "lesion_latest.pth")

    torch.save(model.state_dict(), temp_path)
    os.replace(temp_path, final_path)   


print("Training complete.")

# need to make a file for new images... directories need to be established.