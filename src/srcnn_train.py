import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_dataset import PairedImageDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.srcnn import ModifiedSRCNN
from misc import config
import time 


train_dataset = PairedImageDataset(config.HIGH_RES_TRAIN_FOLDER, config.LOW_RES_TRAIN_FOLDER)
valid_dataset = PairedImageDataset(config.HIGH_RES_VALID_FOLDER, config.LOW_RES_VALID_FOLDER)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_SRCNN, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE_SRCNN, shuffle=False, num_workers=4)

model = ModifiedSRCNN(in_channels = 3, num_blocks = 3, n1 = 64, n2 = 32, f1 = 9, f2 = 5, f3 = 5)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_SRCNN)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_folder, patience=5):
    # Move model to the specified device
    model.to(device)
    
    # Create directories for storing artifacts
    os.makedirs(log_folder, exist_ok=True)
    
    log_file = os.path.join(log_folder, 'logs.log')
    best_weights_file = os.path.join(log_folder, 'best_weights.pth')
    
    best_loss = float('inf')
    patience_counter = 0
    
    with open(log_file, 'w') as log:
        log.write('Epoch,Train Loss,Val Loss,Epoch Time\n')
        
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_losses = []
            for hr_images, lr_images in train_loader:
                hr_images, lr_images = hr_images.to(device), lr_images.to(device)
                optimizer.zero_grad()
                sr_images = model(lr_images)
                loss = criterion(sr_images, hr_images)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for hr_images, lr_images in val_loader:
                    hr_images, lr_images = hr_images.to(device), lr_images.to(device)
                    sr_images = model(lr_images)
                    loss = criterion(sr_images, hr_images)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.2f}s')
            log.write(f'{epoch+1},{train_loss},{val_loss},{epoch_time}\n')
            
            # Check for best validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_weights_file)
            else:
                patience_counter += 1
                        
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model


if __name__ == "__main__":
    model = train_model(model=model, 
                        train_loader=train_loader, 
                        val_loader=valid_loader, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        num_epochs=config.SRCNN_NUM_EPOCHS, 
                        device=config.DEVICE,
                        log_folder="logs/SRCNN", 
                        patience=config.SRCNN_PATIENCE)
    model.save("weights/srcnn_weights.pth")