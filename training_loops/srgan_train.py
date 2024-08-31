from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from tqdm import tqdm 
import time

class ResNetLoss(nn.Module):
    def __init__(self, use_half_precision: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = resnet50(weights = ResNet50_Weights.DEFAULT)
        if use_half_precision:
            self.resnet = self.resnet.half()
        self.loss = nn.MSELoss()

    def forward(self, sr_image, hr_image):
        return self.loss(self.resnet(sr_image), self.resnet(hr_image))
    

def train_srgan(train_loader, valid_loader, generator, discriminator, 
                generator_optim, discriminator_optim, bce_loss, resnet_loss, num_epochs: int = 100, 
                device: str = DEVICE, log_folder: str = "logs/SRGAN", patience: int = 15, 
                use_half_precision: bool = False):
    """ Train the SRGAN model with the given hyperparameters
    train_loader (DataLoader): Training data loader
    valid_loader (DataLoader): Validation data loader
    generator (nn.Module): Generator model
    discriminator (nn.Module): Discriminator model
    generator_optim (Optimizer): Generator optimizer
    discriminator_optim (Optimizer): Discriminator optimizer
    bce_loss (nn.Module): Binary cross entropy loss
    resnet_loss (nn.Module): ResNet loss
    num_epochs (int): Number of epochs
    log_folder (str): Folder to save the logs
    device (torch.device): Device to run the model
    patience (int): Patience for early stopping. Default: 15
    use_half_precision (bool): Use half precision training. Default: False
    """
    # Move model to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Check for half precision training
    if use_half_precision:
        generator = generator.half()
        discriminator = discriminator.half()
    
    # Create the log folder
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "log.log")
    generator_wts_path = os.path.join(log_folder, "generator_weights.pth")
    discriminator_wts_path = os.path.join(log_folder, "discriminator_weights.pth")

    # Initialize the best loss for Early stopping
    best_loss = float("inf")
    patience_counter = 0

    # Start the training loop
    with open(log_file, "w") as log:
        log.write("Epoch,GenTrainLoss,DiscTrainLoss,GenValidLoss,DiscValidLoss,Time\n")

        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            start_time = time.time()    

            # training phase
            generator.train()
            discriminator.train()
            total_gen_loss, total_disc_loss = 0.0, 0.0

            for lr_images, hr_images in train_loader:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)

                # Train the discriminator
                sr_images = generator(lr_images)
                discriminator_real = discriminator(hr_images)
                discriminator_fake = discriminator(sr_images.detach())

                discriminator_real_loss = bce_loss(discriminator_real, 
                                                   torch.ones_like(discriminator_real, dtype=torch.float16 if use_half_precision else torch.float32))
                discriminator_fake_loss = bce_loss(discriminator_fake, 
                                                   torch.zeros_like(discriminator_fake, dtype=torch.float16 if use_half_precision else torch.float32))
                discriminator_loss = discriminator_real_loss + discriminator_fake_loss
                total_disc_loss += discriminator_loss.item()

                discriminator_optim.zero_grad()
                discriminator_loss.backward()
                discriminator_optim.step()

                # Train the generator
                discriminator_fake = discriminator(sr_images)
                generator_loss = 1e-3 * bce_loss(discriminator_fake, 
                                                 torch.ones_like(discriminator_fake, dtype=torch.float16 if use_half_precision else torch.float32))
                resnet_loss_val = 0.006 * resnet_loss(sr_images, hr_images)
                generator_loss += resnet_loss_val
                total_gen_loss += generator_loss.item()

                generator_optim.zero_grad()
                generator_loss.backward()
                generator_optim.step()

            total_disc_loss /= len(train_loader)
            total_gen_loss /= len(train_loader)

            # Validation phase
            generator.eval()
            discriminator.eval()
            total_gen_valid_loss, total_disc_valid_loss = 0.0, 0.0

            with torch.no_grad():
                for lr_images, hr_images in valid_loader:
                    lr_images, hr_images = lr_images.to(device), hr_images.to(device)

                    # Calculate the generator loss
                    sr_images = generator(lr_images)
                    discriminator_fake = discriminator(sr_images)
                    generator_loss = 1e-3 * bce_loss(discriminator_fake, 
                                                     torch.ones_like(discriminator_fake, dtype=torch.float16 if use_half_precision else torch.float32))
                    resnet_loss_val = 0.006 * resnet_loss(sr_images, hr_images)
                    generator_loss += resnet_loss_val
                    total_gen_valid_loss += generator_loss.item()

                    # Calculate the discriminator loss
                    discriminator_real = discriminator(hr_images)
                    discriminator_fake = discriminator(sr_images.detach())
                    discriminator_real_loss = bce_loss(discriminator_real, 
                                                       torch.ones_like(discriminator_real, dtype=torch.float16 if use_half_precision else torch.float32))
                    discriminator_fake_loss = bce_loss(discriminator_fake, 
                                                       torch.zeros_like(discriminator_fake, dtype=torch.float16 if use_half_precision else torch.float32))
                    discriminator_loss = discriminator_real_loss + discriminator_fake_loss
                    total_disc_valid_loss += discriminator_loss.item()

            total_gen_valid_loss /= len(valid_loader)
            total_disc_valid_loss /= len(valid_loader)

            # Log the losses
            end_time = time.time()
            log.write(f"{epoch + 1},{total_gen_loss},{total_disc_loss},{total_gen_valid_loss},{total_disc_valid_loss},{end_time - start_time}\n")
            print(f"Epoch: {epoch + 1}, GenTrainLoss: {total_gen_loss}, DiscTrainLoss: {total_disc_loss}, GenValidLoss: {total_gen_valid_loss}, DiscValidLoss: {total_disc_valid_loss}, Time: {end_time - start_time}")

            # check for early stopping
            if total_gen_valid_loss < best_loss:
                best_loss = total_gen_valid_loss
                patience_counter = 0
                torch.save(generator.state_dict(), generator_wts_path)
                torch.save(discriminator.state_dict(), discriminator_wts_path)
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter} / {patience}: Loss did not improve from {best_loss}")
                if patience_counter >= patience:
                    print("Early stopping after {epoch + 1} epochs")
                    break
        
    generator.load_state_dict(torch.load(generator_wts_path))
    discriminator.load_state_dict(torch.load(discriminator_wts_path))



        
    
                

            


    




if __name__ == "__main__":
    sr_image_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)
    hr_image_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)
    resnet_loss = ResNetLoss().to(DEVICE)
    loss = resnet_loss(sr_image_tensor, hr_image_tensor)
    print(loss)