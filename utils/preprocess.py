import numpy as np 
import torch 
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from misc import config



class Preprocessor:
    def __init__(self, image_folder, high_res_base_folder, low_res_base_folder) -> None:
        self.image_folder = image_folder
        self.high_res_base_folder = high_res_base_folder
        self.low_res_base_folder = low_res_base_folder
        self.high_res_img_counter, self.low_res_img_counter = 0, 0

        self.images = os.listdir(image_folder)
        os.makedirs(high_res_base_folder, exist_ok = True)
        os.makedirs(low_res_base_folder, exist_ok = True)

    
    def break_into_patches(self, img, patch_size = 33, stride = 14):
        """ Breaks an image into patches of size patch_size x patch_size 
        with a stride of stride pixels 
        Args:
            img: np.array of shape (H, W, C)
            patch_size: int, size of the patch
            stride: int, stride of the patch
        Returns:
            np.array of shape (N, patch_size, patch_size, C)
        """
        patches = []
        for i in range(0, img.shape[0] - patch_size + 1, stride):
            for j in range(0, img.shape[1] - patch_size + 1, stride):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        return np.array(patches)
    
    def save_patches(self, patches, base_path, high_res = True):
        """ Saves patches to a directory
        Args:
            patches: np.array of shape (N, patch_size, patch_size, C)
            base_path: str, path to the directory
            high_res: bool, if True, saves the patches to the high_res folder
        """               
        if high_res: 
            for patch in patches:
                cv2.imwrite(os.path.join(base_path, f"{self.high_res_img_counter}.jpg"), patch)
                self.high_res_img_counter += 1
        else: 
            for patch in patches:
                cv2.imwrite(os.path.join(base_path, f"{self.low_res_img_counter}.jpg"), patch)
                self.low_res_img_counter += 1
                
    def blur_image(self, image, scaling_factor: int = 4):
        """Blurr Image by (1) applying Gaussian Blur, (2) Subsample from 
        image buy the scaling factor, (3) Upsample with bicubic interpolation
        Args:
            image (np.ndarray): Image of dimensions (H, W, C)
            scaling factor (int): Scaling factor of the images.
        Returns:
            The low resolution image"""
        # Randomly select Gaussian blur parameters
        kernel_size = np.random.choice([3, 5, 7, 9, 11])
        sigmaX = np.random.uniform(0.1, 10)
        sigmaY = np.random.uniform(0.1, 10)
        
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX, sigmaY)

        # Subsample the image by the scaling factor
        width = int(blurred_image.shape[1] / scaling_factor)
        height = int(blurred_image.shape[0] / scaling_factor)
        subsampled_image = cv2.resize(blurred_image, (width, height), interpolation=cv2.INTER_AREA)

        # Upscale the subsampled image by the same factor using bicubic interpolation
        upscaled_image = cv2.resize(subsampled_image, (blurred_image.shape[1], blurred_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        return upscaled_image
    
    def make_low_res_patches(self, high_res_patches):
        """ Makes low resolution patches from high resolution patches
        Args:
            high_res_patches: np.array of shape (N, patch_size, patch_size, C)
        Returns:
            np.array of shape (N, patch_size, patch_size, C)
        """
        low_res_patches = np.zeros_like(high_res_patches)
        for i, patch in enumerate(high_res_patches):
            low_res_patches[i] = self.blur_image(patch)
        return low_res_patches
    
    def preprocess_images(self):
        """ Saves all patches to the respective directories
        image_folder (filepath): Path where the base images are collected from
        high_res_base_folder (filepath): Path where the high res sub-images are saved
        low_res_base_folder (filepath): Path where the low res sub-images are saved.
        """
        for image in tqdm(self.images, desc="Processing images"):
            if image.endswith((".png", ".jpg")):
                img = cv2.imread(os.path.join(self.image_folder, image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                high_res_patches = self.break_into_patches(img, patch_size=128, stride = 256)
                low_res_patches = self.make_low_res_patches(high_res_patches=high_res_patches)
                self.save_patches(high_res_patches, self.high_res_base_folder, high_res=True)
                self.save_patches(low_res_patches, self.low_res_base_folder, high_res=False)
            else:
                continue



if __name__ == "__main__":
    train_preprocessor = Preprocessor(image_folder=config.TRAINING_IMAGE_FOLDER, 
                                        high_res_base_folder=config.HIGH_RES_TRAIN_FOLDER, 
                                        low_res_base_folder=config.LOW_RES_TRAIN_FOLDER)
    valid_preprocessor = Preprocessor(image_folder=config.VALID_IMAGE_FOLDER, 
                                        high_res_base_folder=config.HIGH_RES_VALID_FOLDER, 
                                        low_res_base_folder=config.LOW_RES_VALID_FOLDER)
        
    train_preprocessor.preprocess_images()
    valid_preprocessor.preprocess_images()
