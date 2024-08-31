import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairedImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        """Initialize the dataset class
        Args:
            hr_dir(str): Path to the high resolution directory
            lr_dir(str): Path to the low resolution directory
            transform: Transformations to apply. By default applies the ToTensor operation"""
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),])
        else: self.transform = transform 

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert("RGB")
        lr_image = Image.open(lr_image_path).convert("RGB")

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return hr_image, lr_image