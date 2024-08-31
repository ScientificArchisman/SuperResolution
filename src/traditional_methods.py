from PIL import Image, ImageFilter
import cv2
import numpy as np
import pywt
from scipy.ndimage import zoom

def apply_traditional_editing(image, methods, scale_factor=2):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    for method in methods:
        if method == 'Bilinear Interpolation':
            image = image.resize((new_width, new_height), Image.BILINEAR)
        elif method == 'Bicubic Interpolation':
            image = image.resize((new_width, new_height), Image.BICUBIC)
        elif method == 'Lanczos Resampling':
            image = image.resize((new_width, new_height), Image.LANCZOS)
        elif method == 'Nearest-Neighbor Interpolation':
            image = image.resize((new_width, new_height), Image.NEAREST)
        elif method == 'Gaussian Filtering':
            open_cv_image = np.array(image)
            blurred_image = cv2.GaussianBlur(open_cv_image, (5, 5), 0)
            image = Image.fromarray(blurred_image)
        elif method == 'Median Filtering':
            open_cv_image = np.array(image)
            median_image = cv2.medianBlur(open_cv_image, 5)
            image = Image.fromarray(median_image)
        elif method == 'Unsharp Masking':
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        elif method == 'Fourier-Based Methods':
            open_cv_image = np.array(image.convert('L'))
            dft = cv2.dft(np.float32(open_cv_image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            rows, cols = open_cv_image.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols, 2), np.uint8)
            r = 30
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 0
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            image = Image.fromarray(np.uint8(img_back))
        elif method == 'Wavelet-Based Methods':
            open_cv_image = np.array(image.convert('L'))
            coeffs2 = pywt.dwt2(open_cv_image, 'haar')
            LL, (LH, HL, HH) = coeffs2
            LL = np.clip(LL, 0, 255)
            image = Image.fromarray(np.uint8(LL))
        elif method == 'Spline Interpolation':
            open_cv_image = np.array(image)
            zoomed_image = zoom(open_cv_image, (scale_factor, scale_factor, 1), order=3)
            image = Image.fromarray(np.uint8(zoomed_image))
    
    return image