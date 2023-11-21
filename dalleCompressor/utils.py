import cv2
import torch
import random
import numpy as np

from dalle2_pytorch import DALLE2
from dalle2_pytorch.tokenizer import tokenizer

from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig
from dalle2_pytorch.dalle2_pytorch import resize_image_to, cast_tuple, VQGanVAE, maybe, default, exists

def PSNR(x, y):
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(1. / mse)

def read_image(image=None, path=None, size=None):
    if image is None:
        image = cv2.imread(path)[..., ::-1]
        
    max_size = max(image.shape[:2])
    new_image = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    h = new_image.shape[0] - image.shape[0]
    w = new_image.shape[1] - image.shape[1]
    
    h1 = new_image.shape[0] - h // 2
    if image.shape[0] % 2:
        h1 -= 1
    w1 = new_image.shape[1] - w // 2
    if image.shape[1] % 2:
        w1 -= 1
        
    new_image[h // 2: h1, w // 2: w1] = image
    
    new_image = cv2.resize(new_image, (size, size))
    new_image = new_image.astype(np.float64) / 255.
    return new_image

def set_seed(seed: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)