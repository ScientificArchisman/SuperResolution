from PIL import Image
from torchvision import transforms
import numpy as np
import imageio

def proprocess_to_test(image_path: str, transform = None):
    """Preprocess an image for testing
    Args:
        image_path: str, path to the image
        transform: torchvision.transforms, transformations to apply
    Returns:
        torch.Tensor of shape (C, H, W)
    """
    img = Image.open(image_path).convert("RGB")
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    return img


def save_image(array: np.ndarray, path: str) -> None:
    """
    Save a NumPy array representing an image to a specified path.
    
    Parameters:
    array (np.ndarray): The image data in the form of a NumPy array with shape (H, W, C).
    path (str): The path where the image will be saved.
    """
    # Ensure the array has the shape (H, W, C)
    if len(array.shape) != 3 or array.shape[2] not in [1, 3, 4]:
        raise ValueError("Array must have shape (H, W, C) where C is 1, 3, or 4.")
    
     # If the array is of float type, convert it to uint8
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array * 255).astype(np.uint8)
    
    # Save the image to the specified path
    imageio.imwrite(path, array)