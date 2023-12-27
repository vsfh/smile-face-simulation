import cv2
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
from unsvc.para_fitting import detection_and_segmentation
import os
import shutil

if __name__=='__main__':
    def a():
        case = 'C01002721350'
        eu = cv2.imread(f'/mnt/d/data/smile/out1/{case}/upper_edge.png')
        ed = cv2.imread(f'/mnt/d/data/smile/out1/{case}/down_edge.png')
        mk = cv2.imread(f'/mnt/d/data/smile/out1/{case}/mouth_mask.png')
        tk = cv2.imread(f'/mnt/d/data/smile/out1/{case}/teeth_mask.png')
        img = cv2.imread(f'/mnt/d/data/smile/out1/{case}/smile.png')
        depth = cv2.imread(f'/mnt/d/data/smile/out1/{case}/step/depth.png')
        
        # ed, eu = cv2.dilate(ed, np.ones((3,3))), cv2.dilate(eu, np.ones((3,3)))
        a = np.zeros((256,256,3), dtype=np.uint8)
        a[...,1] += eu[...,0]
        a[...,2] += ed[...,0]
        # a[...,0] += depth[...,0]
        
        a[mk==0] = img[mk==0]
        b = img.copy()
        b[tk==0] = 0
        cv2.imwrite(f'/mnt/d/data/smile/out1/{case}/step/input_style.png', a)
        cv2.imwrite(f'/mnt/d/data/smile/out1/{case}/step/edge.png', eu+ed)
    
    def b():
        path = '/mnt/e/data/classification/ori_data/2023-05-04-fussen_classification'
        spath = '/mnt/e/data/classification/ori_data/2'
        for file in os.listdir(path):
            for img_name in os.listdir(os.path.join(path, file)):
                if img_name.startswith('正位片'):
                    shutil.copy(os.path.join(path,file,img_name), os.path.join(spath, f'{file}.png'))           
        
        face = cv2.imread('/mnt/d/data/smile/TeethSimulation/C01002722687/微笑像.jpg')

    def c():    
        res = detection_and_segmentation(face)
        (y,x) = res['position']
        (width, height, half) = res['width_height']
        mouth = cv2.imread('/mnt/e/paper/smile/figure/mouth_out.png')
        face = cv2.resize(face, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
        face[y:y+256,x:x+256] = mouth
        face = cv2.resize(face, (width, height), cv2.INTER_AREA)
        cv2.imwrite('/mnt/e/paper/smile/figure/face_out.png', face)
        
    def d():
        import cv2
        import numpy as np
        from scipy.interpolate import splprep, splev

        # 读取图像
        image = cv2.imread('/mnt/e/paper/smile/figure/C01002722687/step/up_edge.png', cv2.IMREAD_GRAYSCALE)

        # 二值化处理（如果图像不是二值化的）
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # 找到轮廓
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个空白画布
        smooth_image = np.zeros_like(image)

        # 遍历所有轮廓并平滑
        for cnt in contours:
            # 将轮廓点转换为(x, y)坐标
            points = cnt[:, 0, :]
            x, y = points[:, 0], points[:, 1]

            # 使用样条曲线进行平滑
            tck, u = splprep([x, y], s=20)
            unew = np.linspace(u.min(), u.max(), 1000)
            smooth = splev(unew, tck)

            # 绘制平滑后的曲线
            smooth_points = np.array(smooth).T.round().astype(int)
            cv2.polylines(smooth_image, [smooth_points], False, (255, 255, 255), 1)
            # for p in smooth_points:
            #     cv2.circle(smooth_image, tuple(p), 1, (255, 255, 255), -1)

        # 显示或保存结果
        cv2.imshow('Smooth Curve', smooth_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def e():
        import torch
        import cv2
        from pytorch3d.structures import  Meshes
        from pytorch3d.renderer import (
            BlendParams, 
            PointLights, 
            RasterizationSettings, 
            MeshRenderer, 
            MeshRasterizer,  
            HardPhongShader,
            PerspectiveCameras,
            TexturesUV
        )
        from pytorch3d.utils import torus
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=2e-3,
            faces_per_pixel=25,
            perspective_correct=True,
            cull_backfaces=True
        )
        focal_length = 12
        light=[16.0, -101.0, -33.0]
        color = 0.26
        opt_cameras = PerspectiveCameras(device=device, focal_length=focal_length)
        
        lights = PointLights(device=device, ambient_color=((color, color, color),), location=[light])
        blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
        shader = HardPhongShader(device=device, cameras=opt_cameras, lights=lights,blend_params=blend_params)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=opt_cameras,
                raster_settings=raster_settings
            ),
            shader=shader
        )
        # 设置设备

        # 读取二值图像
        mask_image = cv2.imread('/mnt/e/paper/smile/figure/C01002722687/mouth_mask.png', cv2.IMREAD_GRAYSCALE)

        # 创建透明度通道（0: 不透明, 255: 透明）
        alpha_channel = torch.where(torch.tensor(mask_image) == 0, 1.0, 0.0)

        # 将原始图像和透明度通道合并
        mask_texture = torch.cat([torch.ones_like(alpha_channel).unsqueeze(-1), alpha_channel.unsqueeze(-1)], dim=2)

        # 调整纹理维度以符合PyTorch3D的要求
        mask_texture = mask_texture.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]

        # 创建纹理
        texture = TexturesUV(verts_rgb=mask_texture)

        # 创建平面网格
        verts, faces, _ = torus(r=1, R=2, sides=64, rings=128)
        plane_mesh = Meshes(verts=[verts], faces=[faces], textures=texture)
        img = renderer(plane_mesh)
        pass
    
    def f():
        def concat(case):
            from PIL import Image


            # 读取底层图片和顶层图片
            base_image = cv2.imread(f'/mnt/d/data/smile/out1/{case}/smile.png')
            overlay_image = cv2.imread(f'/mnt/d/data/smile/out1/{case}/step/3d.png')
            a = 0.49
            new_img = np.zeros_like(base_image)
            new_img[overlay_image==0] = base_image[overlay_image==0]
            new_img[overlay_image!=0] = base_image[overlay_image!=0]*a + overlay_image[overlay_image!=0]*(1-a)
            cv2.imwrite(f"/mnt/d/data/smile/out1/{case}/target.png", new_img.astype(np.uint8))
            
            # while True:
            #     new_img = np.zeros_like(base_image)
            #     new_img[overlay_image==0] = base_image[overlay_image==0]
            #     new_img[overlay_image!=0] = base_image[overlay_image!=0]*a + overlay_image[overlay_image!=0]*(1-a)
            #     cv2.imshow('img', new_img.astype(np.uint8))
            #     key = cv2.waitKey()
            #     if key == 97:
            #         a -= 0.05
            #     elif key == 100:
            #         a += 0.05
            #     else:
            #         print(a)
            #         a = 1
                    
            #         break

        for case in os.listdir('/mnt/d/data/smile/out1/'):
            concat(case)
        
    f()