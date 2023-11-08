import cv2
import numpy as np
import random
import math
import torch

def preprocess(img):
    img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    if len(img_resize.shape)==2:
        img_resize = img_resize[:,:,None].repeat(1,1,3)
    im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
    im = torch.from_numpy(im)  # to torch
    im = im.float()  # uint8 to fp16/32
    im /= 255.0  # 0-255 to 0.0-1.0
    return im

class RandomPerspective:

    def __init__(self,
                 degrees=0.0,
                 translate=0.0,
                 scale=0.0,
                 shear=0.0,
                 perspective=1.0,
                 border=(0, 0),
                 pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform
        self.size = (256,256)
    def __call__(self, img):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(-self.translate, self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        # M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        M = T @ S @ R
        # print(M)
        
        # Affine image

        if self.perspective:
            img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(0, 0, 0))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(0, 0, 0))
        return img
