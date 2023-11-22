import argparse
import math
import random
import os

import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import cv2

wandb = None
from network.model_cond import Generator, Discriminator
model = Generator(256, 512, 8).cuda()
ckpt_down = torch.load("/mnt/e/share/weight/smile/150000.pt", map_location=lambda storage, loc: storage)
model.load_state_dict(ckpt_down["g_ema"])
sample_z = torch.randn(64, 512, device='cuda')

with torch.no_grad():
    model.eval()
    sample, _ = model([sample_z])
    imgs = sample.clip(0,1).detach().cpu().numpy()
    for img in imgs:
        img = np.transpose(img, (1,2,0))[...,::-1]*255
        cv2.imshow('img', img.astype(np.uint8))
        cv2.waitKey(0)
    pass