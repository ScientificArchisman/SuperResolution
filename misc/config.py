import torch
TRAINING_IMAGE_FOLDER = "archive/DIV2k/train_hr"
VALID_IMAGE_FOLDER = "archive/DIV2k/valid_hr"

HIGH_RES_TRAIN_FOLDER = "archive/processed_data/high_res_train"
LOW_RES_TRAIN_FOLDER = "archive/processed_data/low_res_train"
HIGH_RES_VALID_FOLDER = "archive/processed_data/high_res_valid"
LOW_RES_VALID_FOLDER = "archive/processed_data/low_res_valid"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessor parameters
PATCH_SIZE: int = 128
STRIDE: int = 256

# SRCNN model parameters
BATCH_SIZE_SRCNN: int = 32
LEARNING_RATE_SRCNN=0.001
SRCNN_NUM_EPOCHS: int = 100
SRCNN_PATIENCE: int = 15