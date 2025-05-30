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

class YangOldNew(Dataset):
    def __init__(self, path, mode='test'):
        self.path = path
        
        self.all_files = []
        if mode=='test':
            folder_list = os.listdir(self.path)[-9:]
        else:
            folder_list = os.listdir(self.path)[:-9]
            
        for folder in folder_list:
            self.all_files.append(os.path.join(self.path, folder,))

        print('total image:', len(self.all_files))
        self.mode = mode
        self.half = False
        self.aug = RandomPerspective(translate=0.05, degrees=5, scale=0.05)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(img_folder, 'Img.jpg'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        im = preprocess(img)

        cond = torch.zeros((7,256,256))
        cond_im = img.copy()
        cond_im_2 = img.copy()
        ed = cv2.imread(os.path.join(img_folder, 'TeethEdgeDown.png'))
        eu = cv2.imread(os.path.join(img_folder, 'TeethEdgeUp.png'))
        mk = cv2.imread(os.path.join(img_folder, 'MouthMask.png'))
        mk_dia = cv2.dilate(mk, kernel=np.ones((7,7)))
        
        tk = cv2.imread(os.path.join(img_folder, 'TeethMasks.png'))
        eu, ed = cv2.dilate(eu, kernel=np.ones((3,3))), cv2.dilate(ed, kernel=np.ones((3,3)))
        
        cond[3] = preprocess(mk_dia)[0]
        
        cond_im[tk==0]=0
        if self.mode == 'train':
            # cond_im = self.aug(cond_im)
            cond_im = cond_im + np.random.normal(0, 100, cond_im.shape)
        cond[-3:] = preprocess(cond_im)
        
        cond_im_2[...,0][mk_dia[...,0]!=0] = ed[...,0][mk_dia[...,0]!=0]
        cond_im_2[...,1][mk_dia[...,0]!=0] = eu[...,0][mk_dia[...,0]!=0]     
        cond_im_2[...,2][mk_dia[...,0]!=0] = tk[...,0][mk_dia[...,0]!=0] 
        cond[:3] = preprocess(cond_im_2)
        
        return {'images': im, 'cond':cond}
    
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
        # cond_im_2 = np.zeros_like(img)
        
        
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
        mk_dia = cv2.dilate(mk, kernel=np.ones((5,5)))
        cond[3] = preprocess(mk)[0]
        
        cond_im[tk==0]=0
        cond_im = self.aug(cond_im)
        cond[-3:] = preprocess(cond_im)

        cond_im_2[mk!=0]=0
        cond_im_2[...,0][mk[...,0]!=0] = ed[...,0][mk[...,0]!=0]
        cond_im_2[...,1][mk[...,0]!=0] = eu[...,0][mk[...,0]!=0]     
        # cond_im_2[...,2][mk[...,0]!=0] = tk[...,0][mk[...,0]!=0] 
        if self.show:
            print(img_folder)
            cv2.imshow('cond1', cond_im)
            cv2.imshow('cond2', cond_im_2)
            cv2.waitKey(0)
            
        cond[:3] = preprocess(cond_im_2)
        
        return {'images': im, 'cond':cond}   

class Pair(Dataset):
    def __init__(self, path='/media/gregory/smile/mono_pair/', mode='train'):
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
        img = cv2.imread(os.path.join(img_folder, 'b.png'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        im = preprocess(img)
        
        cond = torch.zeros((4,256,256))
        mk = cv2.imread(os.path.join(img_folder, 'd.png'))
        cond[3] = preprocess(mk)[0]
        
        img = cv2.imread(os.path.join(img_folder, 'c.png'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        cond[:3] = preprocess(img)
        

        
        return {'images': im, 'cond':cond}   
    
class Pair(Dataset):
    def __init__(self, path='/media/gregory/smile/mono_pair/', mode='train'):
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
        img = cv2.imread(os.path.join(img_folder, 'b.png'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        im = preprocess(img)
        
        cond = torch.zeros((4,256,256))
        mk = cv2.imread(os.path.join(img_folder, 'd.png'))
        cond[3] = preprocess(mk)[0]
        
        img = cv2.imread(os.path.join(img_folder, 'c.png'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        cond[:3] = preprocess(img)
        

        
        return {'images': im, 'cond':cond}   

class FFHQDepth(Dataset):
    def __init__(self, mode, args):
        self.args = args
        self.path = args.path
        self.is_training = mode=='train'
        self.all_files = []
        zero_file = ['10595.txt','10794.txt','06696.txt','12142.txt','08889.txt','13202.txt',\
            '11563.txt','07342.txt','12556.txt','11858.txt','11596.txt','12202.txt','07966.txt',\
                '07468.txt','11818.txt','13339.txt','10085.txt','10486.txt','06185.txt','08692.txt',\
                    '09005.txt','09640.txt','00044.txt','13420.txt','09848.txt','10454.txt','07077.txt',\
            '14031.txt','12129.txt','05355.txt','10855.txt','10686.txt','07612.txt','08986.txt','06474.txt',\
            '11056.txt','07697.txt','13510.txt','06667.txt','07920.txt','11216.txt','10592.txt','09207.txt',\
            '13834.txt','11054.txt','12697.txt','12214.txt','12754.txt','08536.txt','08182.txt','07922.txt',\
            '08998.txt','12613.txt','07385.txt','11290.txt','10456.txt','07818.txt','13801.txt','09739.txt',\
                '07027.txt','07315.txt','09786.txt','09588.txt','13480.txt','11877.txt','09489.txt','12904.txt',\
            '11283.txt','10551.txt','06572.txt','09421.txt','10092.txt','06424.txt','07125.txt','13600.txt','13794.txt',\
            '10861.txt','06059.txt','03782.txt','11188.txt','12794.txt','04371.txt','13144.txt','06902.txt','06409.txt',\
            '11859.txt','11354.txt','06382.txt','06309.txt','07039.txt','10503.txt','06795.txt','06862.txt','07616.txt',\
            '10525.txt','09161.txt','13127.txt','09297.txt','13512.txt','08657.txt','10884.txt','10760.txt','09182.txt',\
            '11177.txt','09011.txt','11192.txt','06340.txt','10062.txt','10343.txt','10742.txt','11340.txt','11971.txt',\
                '13596.txt','13933.txt','13953.txt','13612.txt','14135.txt','06084.txt','09854.txt','10700.txt']
        if not self.is_training:
            all_files = natsorted(os.listdir(args.path))[-9:]
        else:
            all_files = natsorted(os.listdir(args.path))[:-9]
        self.all_files = [f for f in all_files if f.replace('png','txt') not in zero_file]
        print('total image:', len(self.all_files))
        self.show = args.show
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.all_files[index])
        depth_path = img_path.replace('detect', 'depth').replace('changed_image', 'ffhq_gray')
        label_path = img_path.replace('changed_image', 'labels/train').replace('png', 'txt')
        mmask_path = img_path.replace('detect/changed_image', 'mmask/train')

        im = cv2.imread(img_path) # (1024,1024,3)
        mmask = cv2.imread(mmask_path)
        w,h = im.shape[:2]
        depth = cv2.imread(depth_path)
        if isinstance(self.args.size, int):
            size = (self.args.size, self.args.size)
        else:
            size = self.args.size
        re_im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
        crop_size = self.args.crop_size
        if crop_size > 0:
            im = cv2.resize(im[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
            mmask = cv2.resize(mmask[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        
        edge = np.zeros_like(im)
        mask = np.zeros_like(im)
        with open(label_path, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
        segments = [((segment*w-crop_size)/(w-2*crop_size)*size[0]).astype(np.int32) for segment in segments]
        missing_tooth = -1
        if np.random.random()<.3:
            missing_tooth = np.random.choice(len(segments), 1)[0]
            missing_mask = np.zeros_like(im)
        for segment_idx in range(len(segments)):
            segment = segments[segment_idx]
            if segment_idx == missing_tooth:
                segment = np.reshape(segment, (1, -1, 2))
                cv2.fillPoly(missing_mask, segment, color=(255,255,255))
                missing_mask = cv2.GaussianBlur(cv2.dilate(missing_mask,np.ones((3,3))), (5,5), 0)/255
                continue           
            for i in range(len(segment) - 1):
                start_point = tuple(segment[i])    # 当前点
                end_point = tuple(segment[i + 1])  # 下一点
                cv2.line(edge, start_point, end_point, color=(255, 255, 255), thickness=2)
            cv2.line(edge, tuple(segment[-1]), tuple(segment[0]), color=(255, 255, 255), thickness=2)
            
            segment = np.reshape(segment, (1, -1, 2))
            cv2.fillPoly(mask, segment, color=(255, 255, 255))

        if missing_tooth>0:
            inner_region = np.logical_and(missing_mask*255<10, mask<10)
            inner_region = np.logical_and(inner_region, mmask>0)
            depth_inner = depth[inner_region]
            inner_region = np.logical_and(inner_region, depth<depth_inner.mean())
            if inner_region.sum()>10:
                missing_value = np.zeros_like(missing_mask)
                missing_value[...,0] = int(im[...,0][inner_region[...,0]].mean())
                missing_value[...,1] = int(im[...,1][inner_region[...,1]].mean())
                missing_value[...,2] = int(im[...,2][inner_region[...,2]].mean())
                im = im*(1-missing_mask) + missing_value*missing_mask
        
        cond = np.zeros((*size, 9))
        cond[...,:3] = im * (1-mmask/255)
        cond[...,3:6] = depth * (mask/255)
        cond[...,6:9] = edge
        # cond[...,-3:] = re_im
        
        mmask = cv2.dilate(mmask, np.ones((3,3)))
        im = preprocess(im)
        cond = preprocess(cond)
        mmask = preprocess(mmask)
        
        return {'images': im,
                'cond': cond,
                'mmask': mmask,
                'name':img_path}   
    
def get_loader(path, bs ,mode, type='tianshi', args=None):
    if type=='Yang':
        ds = YangOldNew(path, mode)
    elif type=='pair':
        ds = Pair(path, mode)
    elif type=='gen':
        ds = GeneratedDepth(path, mode)
    elif args is not None:
        ds = FFHQDepth(mode, args)
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=(mode=='train'),
                    num_workers=4,
                    drop_last=True)
    return dl

def get_input(img_folder, show=False, ori=False):
    cond = np.zeros((256,256,9))

    if 'data/Teeth' in img_folder:
        depth = cv2.imread(os.path.join(img_folder, 'align/depth.png'))
        edge = cv2.imread(os.path.join(img_folder, 'align/edge.png'))
        img = cv2.imread(os.path.join(img_folder, 'Img.jpg'))
        mk = cv2.imread(os.path.join(img_folder, 'MouthMask.png'))
        # res = np.where(mk>0)[1]
        # size = (256,256)
        # crop_size = min(256-res.max(), res.min())
        # img = cv2.resize(img[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        # depth = cv2.resize(depth[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        # edge = cv2.resize(edge[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_NEAREST)
        # mk = cv2.resize(mk[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_NEAREST)
        
        # mk = cv2.dilate(mk,np.ones((5,5)))
    elif 'detect' in img_folder:
        depth_path = img_folder.replace('detect', 'depth').replace('changed_image', 'ffhq_gray')
        label_path = img_folder.replace('changed_image', 'labels/train').replace('png', 'txt')
        mmask_path = img_folder.replace('detect/changed_image', 'mmask/train')

        im = cv2.imread(img_folder) # (1024,1024,3)
        mmask = cv2.imread(mmask_path)
        w,h = im.shape[:2]
        depth = cv2.imread(depth_path)
        size = (256,256)
        crop_size = 256
        im = cv2.resize(im[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        mmask = cv2.resize(mmask[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        
        edge = np.zeros_like(im)
        mask = np.zeros_like(im)
        with open(label_path, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
        segments = [((segment*w-crop_size)/(w-2*crop_size)*size[0]).astype(np.int32) for segment in segments]
        for segment_idx in range(len(segments)):
            segment = segments[segment_idx]
            for i in range(len(segment) - 1):
                start_point = tuple(segment[i])    # 当前点
                end_point = tuple(segment[i + 1])  # 下一点
                cv2.line(edge, start_point, end_point, color=(255, 255, 255), thickness=2)
            cv2.line(edge, tuple(segment[-1]), tuple(segment[0]), color=(255, 255, 255), thickness=2)
            segment = np.reshape(segment, (1, -1, 2))
            cv2.fillPoly(mask, segment, color=(255, 255, 255))
        mk = mmask
        depth = depth * (mask/255)
        img = im

    else:
        tk = cv2.imread(os.path.join(img_folder, 'step', 'depth.png'))
        ed = cv2.imread(os.path.join(img_folder, 'step', 'down_edge.png'))
        eu = cv2.imread(os.path.join(img_folder, 'step', 'up_edge.png'))
        img = cv2.imread(os.path.join(img_folder, 'smile.png'))
        mk = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
        # ed[2:,:,:] = ed[:-2,:,:]
        eu, ed = cv2.dilate(eu, kernel=np.ones((5,5))), cv2.dilate(ed, kernel=np.ones((5,5)))
        tk[tk!=0]=255
    
    im = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    cond[...,:3] = im * (1-mk/255)
    cond[...,3:6] = depth * (mk/255)
    cond[...,6:9] = edge

    cond = preprocess(cond)
    im = preprocess(img)
    
    im = im.unsqueeze(0).cuda()
    cond = cond.unsqueeze(0).cuda()
    
    return {'images': im, 'cond':cond}   

def get_example(img_folder):
    img = cv2.imread(os.path.join(img_folder, 'smile.png'))
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    im = preprocess(img)
    cond_im = img.copy()
    cond_im_2 = img.copy()
    
    cond = torch.zeros((7,256,256)) 
    tk = cv2.imread(os.path.join(img_folder, 'depth.png'))
    ed = cv2.imread(os.path.join(img_folder, 'down_edge.png'))
    eu = cv2.imread(os.path.join(img_folder, 'up_edge.png'))
    eu, ed = cv2.dilate(eu, kernel=np.ones((5,5))), cv2.dilate(ed, kernel=np.ones((5,5)))
    tk[tk!=0]=255
    mk = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
    mk_dia = cv2.dilate(mk, kernel=np.ones((7,7)))
    
    cond[3] = preprocess(mk_dia)[0]
    
    cond[-3:] = preprocess(cond_im)

    cond_im_2[...,0][mk_dia[...,0]!=0] = ed[...,0][mk_dia[...,0]!=0]
    cond_im_2[...,1][mk_dia[...,0]!=0] = eu[...,0][mk_dia[...,0]!=0]     
    cond_im_2[...,2][mk_dia[...,0]!=0] = tk[...,0][mk_dia[...,0]!=0] 
        
    cond[:3] = preprocess(cond_im_2)
    
    im = im.unsqueeze(0).cuda()
    cond = cond.unsqueeze(0).cuda()
    
    return {'images': im, 'cond':cond}  

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/gregory/smile/data/detect/changed_image/')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    
    ds = FFHQDepth(mode='test', args=args)
    for i in ds:
        print(i['images'].shape)