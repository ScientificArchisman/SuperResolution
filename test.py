import sys
import os 
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.srcnn import ModifiedSRCNN
from utils.feed_to_model import proprocess_to_test, save_image
from misc import config
from tqdm import tqdm


modfified_srcnn_model = ModifiedSRCNN(in_channels = 3, num_blocks = 3, n1 = 64, n2 = 32, f1 = 9, f2 = 5, f3 = 5)
mod_srcnn_weights = torch.load("/Users/archismanchakraborti/Desktop/Final_model/weights/srcnn_weights.pth", map_location=config.DEVICE)
modfified_srcnn_model.load_state_dict(mod_srcnn_weights)

LR_TEST_FOLDER = "conversion/LR"
HR_TEST_FOLDER = "conversion/HR"


with torch.no_grad():
    modfified_srcnn_model.to(config.DEVICE)
    len_folder = len(os.listdir(LR_TEST_FOLDER))
    for idx, img_name in tqdm(enumerate(os.listdir(LR_TEST_FOLDER)), desc="Processing Images", total=len_folder):
        print(f"Processing Image {idx+1}/{len_folder} --------------------> {img_name}")

        img_path = os.path.join(LR_TEST_FOLDER, img_name)
        img = proprocess_to_test(img_path).unsqueeze(0).to(config.DEVICE)
        sr_img = modfified_srcnn_model(img).squeeze(0).permute(1, 2, 0).cpu().numpy()
        save_image(sr_img, os.path.join(HR_TEST_FOLDER, img_name))
