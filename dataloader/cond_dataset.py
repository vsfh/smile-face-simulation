import cv2
import torch
import sys
sys.path.append('.')
sys.path.append('..')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from natsort import natsorted
from dataloader.augment import RandomPerspective, preprocess
import os
import numpy as np

class GeneratedDepth(Dataset):
    def __init__(self, path='/ssd/gregory/smile/out/', mode='train'):
        self.path = path
        
        self.all_files = []
        if mode=='test':
            folder_list = natsorted(os.listdir(self.path))[-9:]
        else:
            folder_list = natsorted(os.listdir(self.path))[:-9]
            
        for folder in folder_list:
            self.all_files.append(os.path.join(self.path, folder,))

        print('total image:', len(self.all_files))
        self.mode = mode
        self.show = False
        self.aug = RandomPerspective(translate=0.05, degrees=5, scale=0.05)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(img_folder, 'smile.png'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        im = preprocess(img)
        cond_im = img.copy()
        cond_im_2 = img.copy()
        
        
        cond = torch.zeros((7,256,256))
        if self.mode=='train':
            tk = cv2.imread(os.path.join(img_folder, 'teeth_mask.png'))
            ed = cv2.imread(os.path.join(img_folder, 'down_edge.png'))
            eu = cv2.imread(os.path.join(img_folder, 'upper_edge.png'))
        else:
            tk = cv2.imread(os.path.join(img_folder, 'step', 'depth.png'))
            ed = cv2.imread(os.path.join(img_folder, 'step', 'down_edge.png'))
            eu = cv2.imread(os.path.join(img_folder, 'step', 'up_edge.png'))
            eu, ed = cv2.dilate(eu, kernel=np.ones((3,3))), cv2.dilate(ed, kernel=np.ones((3,3)))
            tk[tk!=0]=255
        mk = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
        cond[3] = preprocess(mk)[0]
        
        cond_im[tk==0]=0
        cond_im = self.aug(cond_im)
        img[tk!=0]=0
        cond[-3:] = preprocess(cond_im)

        cond_im_2[...,0][mk[...,0]!=0] = ed[...,0][mk[...,0]!=0]
        cond_im_2[...,1][mk[...,0]!=0] = eu[...,0][mk[...,0]!=0]     
        cond_im_2[...,2][mk[...,0]!=0] = tk[...,0][mk[...,0]!=0] 
        if self.show:
            print(img_folder)
            cv2.imshow('cond1', cond_im)
            cv2.imshow('cond2', cond_im_2)
            cv2.waitKey(0)
            
        cond[:3] = preprocess(cond_im_2)
        
        return {'images': im, 'cond':cond}   

def get_loader(path, bs ,mode):
    ds = GeneratedDepth(path, mode)
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=(mode=='train'),
                    num_workers=4,
                    drop_last=True)
    return dl

def get_input(img_folder, step_idx):
    img = cv2.imread(os.path.join(img_folder, 'smile.png'))
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    im = preprocess(img)
    cond_im = img.copy()
    cond_im_2 = img.copy()
    
    
    cond = torch.zeros((7,256,256))

    # tk = cv2.imread(os.path.join(img_folder, f'step{step_idx}.txt', 'depth.png'))
    # ed = cv2.imread(os.path.join(img_folder, f'step{step_idx}.txt', 'down_edge.png'))
    # eu = cv2.imread(os.path.join(img_folder, f'step{step_idx}.txt', 'up_edge.png'))    
    tk = cv2.imread(os.path.join(img_folder, f'step', 'depth.png'))
    ed = cv2.imread(os.path.join(img_folder, f'step', 'down_edge.png'))
    eu = cv2.imread(os.path.join(img_folder, f'step', 'up_edge.png'))
    eu, ed = cv2.dilate(eu, kernel=np.ones((3,3))), cv2.dilate(ed, kernel=np.ones((3,3)))
    tk[tk!=0]=255
    mk = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
    cond[3] = preprocess(mk)[0]
    
    cond_im[mk==0]=0
    img[tk!=0]=0
    cond[-3:] = preprocess(cond_im)

    cond_im_2[...,0][mk[...,0]!=0] = ed[...,0][mk[...,0]!=0]
    cond_im_2[...,1][mk[...,0]!=0] = eu[...,0][mk[...,0]!=0]     
    cond_im_2[...,2][mk[...,0]!=0] = tk[...,0][mk[...,0]!=0] 
        
    cond[:3] = preprocess(cond_im_2)
    
    im = im.unsqueeze(0).cuda()
    cond = cond.unsqueeze(0).cuda()
    
    return {'images': im, 'cond':cond}   