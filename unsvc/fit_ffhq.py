import torch
import natsort
import os
from para_render import get_target_teeth, join_meshes_as_batch, deepmap_to_edgemap, get_renderer
import json
import cv2
import numpy as np
from natsort import natsorted
import glob
from pytorch3d.transforms.rotation_conversions import *

def init():
    para = {
        'case' : 'C01002721204',
        'R' : torch.tensor([[[ 1.0000,  0.0000,  0.0000],
                            [ 0.0000,  0.1517,  0.9884],
                            [ 0.0000, -0.9884,  0.1517]]]).cuda(),
        'T' : torch.tensor([[ 0.,  2.2,  271.]]).cuda(),
        'fov' : 15,
        'dist' : torch.tensor([0., 0., 0.]).cuda(),
        'teeth_width' : 33,
        'ori_y' : 134,
        'ori_x' : 123
    }
    case = para['case']

    with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
        tid_list = json.load(f)
    best_params = torch.load(f'/window/data/smile/out1/{case}/para.pt')
    step = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][-1]
    up_tensor, down_tensor, f_label = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=False, ratio=1.)

    fov = para['fov']
    R = para['R']
    T = para['T']
    err_list = ['C01003287370','C01003398542','C01002769105','C01002722430']
    for img_path in natsorted(glob.glob('/mnt/gregory/smile/data/Teeth/C01*/Img_mask.png')):
        save_path = img_path.replace('Img_mask.png', 'align')
        # if os.path.exists(os.path.join(save_path, 'depth.png')):
        #     continue
        if img_path.split('/')[-2] in err_list:
            continue
        print(img_path)
        
        os.makedirs(save_path, exist_ok=True)
        # img_path = '/mnt/gregory/smile/data/Teeth/C01002741554/Img_mask.png'
        im = cv2.imread(img_path)
        ori_im = cv2.imread(img_path.replace('Img_mask.png', 'Img.jpg'))
        ori_im = cv2.resize(ori_im, (256, 256))
        extra_mask = cv2.imread(img_path.replace('Img_mask.png', 'MouthMask.png'))
        res = np.where(im==200)
        dist = para['dist'].detach().clone()
        
        if (im==200).sum() < 50:
            continue
        if (im>=216).sum() < 300:
            dist[2] += 10
        x1 = res[0].min()
        x2 = res[0].max()
        y1 = res[1].min()
        y2 = res[1].max()
        width = y2 - y1
        ori_y = para['ori_y']
        ori_x = para['ori_x']
        if (x2 - x1) / (y2 - y1) > 1.2:
            up_tensor_, down_tensor_, f_label_ = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=False, ratio=1.2)
            ori_y += 7
        z = (para['teeth_width'] - width)/0.6
        y = (ori_y - x2)/4
        x = (ori_x - y2)/4
        # x = 0
        # y = 0
        # z = 0
        new_T = T.detach().clone()
        new_T[0,0] += x
        new_T[0,1] += y
        new_T[0,2] += z
        
        if (x2 - x1) / (y2 - y1) > 1.2:
            teeth_mesh = join_meshes_as_batch([up_tensor_, down_tensor_.offset_verts(dist)],
                                                include_textures=True)
            # teeth_mesh = up_tensor_
            
        else:
            teeth_mesh = join_meshes_as_batch([up_tensor, down_tensor.offset_verts(dist)],
                                                include_textures=True)
            # teeth_mesh = up_tensor
        renderer = get_renderer('EdgeAndDepth', fov=fov)

        edge_map, depth_map = renderer(meshes_world=teeth_mesh, R=R, T=new_T, f_labels=f_label, extra_mask=extra_mask[...,0])
        # print(np.where(depth_map>0)[0].min(), np.where(depth_map>0)[0].max(),\
        #     np.where(depth_map>0)[1].min(), np.where(depth_map>0)[1].max())
        # ori_im[...,2][edge_map[0]>0] = 256
        
        kernel = np.ones((2, 2), dtype=np.uint8)
        dilated = cv2.dilate(edge_map, kernel, iterations=1)

        # 2. 高斯模糊（抗锯齿）
        blurred = cv2.GaussianBlur(dilated*255, (3, 3), 0)

        # 3. 重新二值化（可选，取决于是否需要纯黑白线条）
        _, smooth_line = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(img_path.replace('Img_mask.png', 'align/depth.png'), depth_map)
        cv2.imwrite(img_path.replace('Img_mask.png', 'align/edge.png'), smooth_line)
        new_para = {
            'case' : 'C01002721204',
            'R' : torch.tensor([[[ 1.0000,  0.0000,  0.0000],
                                [ 0.0000,  0.1517,  0.9884],
                                [ 0.0000, -0.9884,  0.1517]]]).cuda(),
            'T' : new_T,
            'fov' : 15,
            'dist' : torch.tensor([0., 0., 0.]).cuda(),
            'teeth_width' : 33,
            'ori_y' : 134,
            'ori_x' : 123,
            'ratio' : (x2 - x1) / (y2 - y1)
        }
        torch.save(new_para, img_path.replace('Img_mask.png', 'align/para.pt'))
        # cv2.imshow('img', ori_im)
        # cv2.waitKey(0)
        # break

def control(size=(256,256)):
    case = 'C01002721204'
    with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
        tid_list = json.load(f)
    step = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][-1]
    up_tensor, down_tensor, f_label = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=False, ratio=1.)
    
    d = 100
    a = 97
    w = 119
    s = 115
    j = 106
    k = 107
    n = 110
    m = 109
    u=117
    i=105
    z_ = 122
    x_ = 120
    c = 99
    enter = 13
    angle = 0
    err_list = ['C01003287370','C01003398542','C01002769105','C01002722430']
    for img_path in natsorted(glob.glob('/mnt/gregory/smile/data/Teeth/C01*/Img_mask.png'))[780:]:
        # img_path = '/mnt/gregory/smile/data/Teeth/C01002723453/Img_mask.png'
        save_path = img_path.replace('Img_mask.png', 'align')
        print(img_path)
        if img_path.split('/')[-2] in err_list:
            continue
        
        os.makedirs(save_path, exist_ok=True)
        ori_im = cv2.imread(img_path.replace('Img_mask.png', 'Img.jpg'))
        ori_im = cv2.resize(ori_im, size)
        extra_mask = cv2.imread(img_path.replace('Img_mask.png', 'MouthMask.png'))
        z = 0
        y = 0
        x = 0
        angle = 0
        if not os.path.exists(img_path.replace('Img_mask.png', 'align/para.pt')):
            continue
        para = torch.load(img_path.replace('Img_mask.png', 'align/para.pt'))
        fov = para['fov']
        R = para['R']
        T = para['T']
        dist = para['dist'].detach().clone()
        if para['ratio'] > 1.2:
            up_tensor_, down_tensor_, _ = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=False, ratio=1.2)


        while True:
            renderer = get_renderer('EdgeAndDepth', fov=fov)
            new_T = T.detach().clone()
            new_T[0,0] += x
            new_T[0,1] += y
            new_T[0,2] += z
            if para['ratio'] > 1.2:
                teeth_mesh = join_meshes_as_batch([up_tensor_, down_tensor_.offset_verts(dist)],
                                        include_textures=True)
            else:
                teeth_mesh = join_meshes_as_batch([up_tensor, down_tensor.offset_verts(dist)],
                                                    include_textures=True)
            axis_angles = torch.tensor([[angle,0,0]], dtype=torch.float32).cuda()
            new_R = R.detach().clone()@axis_angle_to_matrix(axis_angles)
            
            with torch.no_grad():
                edge_map, depth_map = renderer(meshes_world=teeth_mesh, R=new_R, T=new_T, f_labels=f_label, extra_mask=extra_mask[...,0])
            
            kernel = np.ones((2, 2), dtype=np.uint8)
            dilated = cv2.dilate(edge_map, kernel, iterations=1)
            blurred = cv2.GaussianBlur(dilated*255, (3, 3), 0)
            _, smooth_line = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
            cond_im = ori_im.copy()
            cond_im[...,0][smooth_line>0] = 1
            cond_im[...,1][smooth_line>0] = 1
            cv2.imshow('img', cond_im)
            key = cv2.waitKey()
            if key==a:
                x += 1
            if key==d:
                x -= 1
            if key==w:
                y += 1
            if key==s:
                y -= 1
            if key==j:
                z += 5
            if key==k:
                z -= 5
            if key==n:
                fov += 1
            if key==m:
                fov -= 1
            if key==u:
                angle += 2/180
            if key==i:
                angle -= 2/180
            if key==z_:
                dist[2] += 1
            if key==x_:
                dist[2] -= 1
            if key==c:
                print('dist', dist, 'T', new_T, 'R', new_R, 'fov', fov)
            if key==enter:
                cv2.imwrite(img_path.replace('Img_mask.png', 'align/depth.png'), depth_map)
                cv2.imwrite(img_path.replace('Img_mask.png', 'align/edge.png'), smooth_line)
                new_para = {
                    'case' : 'C01002721204',
                    'R' : new_R,
                    'T' : new_T,
                    'fov' : fov,
                    'dist' : dist,
                    'teeth_width' : 33,
                    'ori_y' : 134,
                    'ori_x' : 123,
                    'ratio' : para['ratio']
                }
                print(new_para)
                torch.save(new_para, img_path.replace('Img_mask.png', 'align/para.pt'))
                break
        # break
# focal 60
# 90 124 90 123
# 96 125 96 123 z + 10
# 75 108 90 123 y + 4 up
# 90 124 75 107 x + 4 left

if __name__ == '__main__':
    control()