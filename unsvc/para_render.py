import trimesh
import os 
import numpy as np
from scipy.spatial.transform import Rotation as R
from glob import glob
import smile_utils
from pytorch3d.renderer import (
	RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
	PerspectiveCameras, SoftPhongShader,HardPhongShader, TexturesVertex, PointLights,SoftSilhouetteShader
)
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
import cv2
import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import *
import natsort
from tqdm import tqdm
import json
from scipy.interpolate import splprep, splev

class DepthShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, light=None, zfar=200, znear=1):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.light = light if blend_params is not None else PointLights()
        self.zfar = zfar
        self.znear = znear

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        extra_mask = kwargs.get("extra_mask", None)

        mask = fragments.pix_to_face >= 0
        if extra_mask is not None:
            mask = mask * extra_mask

        prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
        prob_map = torch.sum(prob_map, -1) 
                
        zbuf = fragments.zbuf[...,0]
        zbuf_ = zbuf.clone()
        zbuf_[zbuf_==-1] = 1e10
        zbuf_ = torch.cat((torch.ones_like(zbuf_[0][None])*1e10,zbuf_),0)
        zbuf_mask = torch.argmin(zbuf_, 0, keepdim=True)
        
        for i in range(len(prob_map)):
            prob_map[i] = zbuf[i]*(zbuf_mask[0]==i+1)
        prob_map = torch.sum(prob_map[:-2], 0)
        
        out_im = 255*(1-(prob_map-prob_map[prob_map>0].min())/(prob_map.max()-prob_map[prob_map>0].min()))
        out_im[prob_map==0] = 0
        out_im = out_im.detach().cpu().numpy().astype(np.uint8)
        
        return out_im, zbuf_mask

class EdgeShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, zfar=200, znear=1):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.zfar = zfar
        self.znear = znear

    def forward(self, fragments, meshes, **kwargs):
        bg_value = 1e6
        zbuf = fragments.zbuf
        N, H, W, K = zbuf.shape

        zbuf[zbuf == -1] = bg_value

        depth, _ = torch.min(zbuf[:-1], dim=0)
        max_depth, min_depth = depth[depth != bg_value].max(), depth[depth != bg_value].min()
        new_depth = 1 - (depth -min_depth) / (max_depth - min_depth)
        new_depth[depth == bg_value] = 0


        bg = torch.ones((1, H, W), device=zbuf.device) * (bg_value - 1)
        zbuf_with_bg = torch.cat([bg, zbuf[..., 0]], dim=0)
        teeth = torch.argmin(zbuf_with_bg, dim=0)

        return teeth, new_depth.cpu().numpy()
  
def deepmap_to_edgemap(teeth_gray, mid):
    teeth_gray = teeth_gray.astype(np.uint8)
    teeth_gray[teeth_gray==teeth_gray.max()]=0
    teeth_gray[teeth_gray==teeth_gray.max()]=0

    up_teeth = teeth_gray * (teeth_gray <= mid)
    down_teeth = teeth_gray * (teeth_gray > mid)
    
    # up_teeth = cv2.erode(up_teeth, np.ones((3,3)))
    # down_teeth = cv2.erode(down_teeth, np.ones((3,3)))
    
    kernelx = np.array([[1, -1], [0, 0]])
    kernely = np.array([[1, 0], [-1, 0]])

    gradx = cv2.filter2D(up_teeth, cv2.CV_32F, kernelx)
    grady = cv2.filter2D(up_teeth, cv2.CV_32F, kernely)
    grad = np.abs(gradx) + np.abs(grady)
    up_edge = (grad > 0).astype(np.uint8) * 255

    gradx = cv2.filter2D(down_teeth, cv2.CV_32F, kernelx)
    grady = cv2.filter2D(down_teeth, cv2.CV_32F, kernely)
    grad = np.abs(gradx) + np.abs(grady)
    down_edge = (grad > 0).astype(np.uint8) * 255

    return up_edge, down_edge
  
def show_target_teeth(img_folder, tid_list, target_step, type='batch', half=True):
    tooth_dict = smile_utils.load_teeth({int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob(os.path.join(img_folder, 'models', '*._Crown.stl'))},tid_list,half=half)
    step_one_dict = {}

    for arr in np.loadtxt(os.path.join(img_folder, target_step)):
        trans = np.eye(4,4)
        trans[:3,3] = arr[1:4]
        trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
        step_one_dict[str(int(arr[0]))] = trans
        
    up_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='up', add=False, num_teeth=7, export=True)
    down_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='down', add=False, num_teeth=7, export=True)
        
    scene = trimesh.Scene()
    for mesh in up_mesh:
        scene.add_geometry(mesh)
    for mesh in down_mesh:
        scene.add_geometry(mesh)
    scene.add_geometry(trimesh.load(f'{img_folder}/models/final/up.stl'))
    scene.add_geometry(trimesh.load(f'{img_folder}/models/final/down.stl'))
    scene.show()
    return

def get_target_teeth(img_folder, tid_list, target_step, type='batch', half=False):
    tooth_dict = smile_utils.load_teeth({int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob(os.path.join(img_folder, 'models', '*._Crown.stl'))},tid_list,half=half)
    step_one_dict = {}
    up_gum, down_gum = smile_utils.load_gum({'Upper':trimesh.load(f'{img_folder}/models/final/up.stl'),\
                                                'Lower':trimesh.load(f'{img_folder}/models/final/down.stl')})    
    # up_gum, down_gum = smile_utils.load_gum({'Upper':trimesh.load(f'{img_folder}/models/Upper/gum.ply'),\
    #                                             'Lower':trimesh.load(f'{img_folder}/models/Lower/gum.ply')})
    
    for arr in np.loadtxt(os.path.join(img_folder, target_step)):
        trans = np.eye(4,4)
        trans[:3,3] = arr[1:4]
        trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
        step_one_dict[str(int(arr[0]))] = trans
        
    up_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='up', add=False, num_teeth=7)
    up_tensor = smile_utils.meshes_to_tensor(up_mesh,type, device='cuda')
    down_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='down', add=False, num_teeth=7)
    # down_mesh.append(down_gum)
    # down_mesh.append(up_gum)
    down_tensor = smile_utils.meshes_to_tensor(down_mesh,type, device='cuda')
    return up_tensor, down_tensor

def get_renderer(output_type='EdgeAndDepth', device='cuda', focal_length=12, light=[16.0, -101.0, -33.0], color = 0.26):
    opt_cameras = PerspectiveCameras(device=device, focal_length=focal_length)
    if 'Depth' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0,
            faces_per_pixel=25,
            perspective_correct=True,
            cull_backfaces=True
        )
        blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.))
        shader = DepthShader(blend_params=blend_params)
    if 'Edge' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0,
            faces_per_pixel=50,
            perspective_correct=True,
            cull_backfaces=True
        )
        blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.))
        shader = EdgeShader(blend_params=blend_params)
    if 'HardPhong' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=2e-3,
            faces_per_pixel=25,
            perspective_correct=True,
            cull_backfaces=True
        )
        lights = PointLights(device=device, ambient_color=((color, color, color),), location=[light])
        blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
        shader = HardPhongShader(device=device, lights=lights,blend_params=blend_params)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=opt_cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return renderer
   
def interface(case):
    img = cv2.imread(f'/window/data/smile/out1/{case}/smile.png')
    mk = cv2.imread(f'/window/data/smile/out1/{case}/mouth_mask.png')
    with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
        tid_list = json.load(f)
    mk_dia = cv2.dilate(mk, np.ones((7,7)))
    best_params = torch.load(f'/window/data/smile/out1/{case}/para.pt')
    step = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][-1]
    up_tensor, down_tensor = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=False)
    T = best_params['T']
    dist = best_params['dist']
    focal_length=best_params['focal_length']
    lighta, lightb, lightc = 2.0, -60.0, -12.0
    color = 0.9
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
    enter = 13
    angle = 0
    while True:
        with torch.no_grad():
            axis_angles = torch.cat([torch.tensor([angle], dtype=torch.float32), torch.tensor([0], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)],0).cuda()
            R_ = axis_angle_to_matrix(axis_angles[None, :])
            R = R_@best_params['R']
            
            teeth_mesh = join_meshes_as_batch([up_tensor, down_tensor.offset_verts(dist)],
                                                include_textures=True)
            renderer = get_renderer('Edge', focal_length=focal_length, light=[lighta, lightb, lightc], color=color)

            out_im, _ = renderer(meshes_world=teeth_mesh, R=R, T=T, extra_mask=None)
            # out_im = (255*out_im[0,:,:,:3]/out_im[0,:,:,:3].max()).detach().cpu().numpy().astype(np.uint8)

            out_im[mk[...,0]==0]=0  
            eu, ed = deepmap_to_edgemap(out_im.detach().cpu().numpy(), up_tensor._N)
            cond_im = img.copy()
            cond_im[...,0][mk_dia[...,0]!=0] = ed[mk_dia[...,0]!=0]
            cond_im[...,1][mk_dia[...,0]!=0] = eu[mk_dia[...,0]!=0]  
            cv2.imshow('img', cond_im)
            key = cv2.waitKey()
            if key==a:
                T[0,0] += 1
                # lighta += 1
            if key==d:
                # lighta -= 1
                T[0,0] -= 1
            if key==w:
                T[0,1] += 1
                # lightb += 1
            if key==s:
                T[0,1] -= 1
                # lightb -= 1
            if key==j:
                T[0,2] += 5
                # lightc += 1
            if key==k:
                T[0,2] -= 5
                # lightc -= 1
            if key==n:
                focal_length += 10
                # color += 0.02
            if key==m:
                focal_length -= 10
                # color -= 0.02
            if key==u:
                angle += 2/180
            if key==i:
                angle -= 2/180
            if key==enter:
                best_params['T'] = T
                best_params['R'] = R
                best_params['dist'] = dist
                best_params['focal_length'] = focal_length
                print(lighta, lightb, lightc, color)
                break
    torch.save(best_params, f'/window/data/smile/out1/{case}/para.pt')
    return out_im

def render_depth_mask(case, save_path,step_idx=-1, show=False):
    mouth_mask = cv2.imread(f'/window/data/smile/out1/{case}/mouth_mask.png')
    with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
        tid_list = json.load(f)
    best_params = torch.load(f'/window/data/smile/out1/{case}/para.pt')
    step = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][step_idx]
    os.makedirs(f'/window/data/smile/out1/{case}/step', exist_ok=True)
    up_tensor, down_tensor = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=True)
    renderer = get_renderer('Depth', focal_length=best_params['focal_length'])
    T = best_params['T']
    dist = best_params['dist']
    R = best_params['R']
    with torch.no_grad():        
        teeth_mesh = join_meshes_as_batch([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)

        out_im, _ = renderer(meshes_world=teeth_mesh, R=R, T=T)
        
        depth = np.where(mouth_mask[...,0]==0, 0, out_im)
        cv2.imwrite(save_path, depth)
        if show:
            cv2.imshow('mat',depth)
            cv2.imshow('mat2',out_im)
            cv2.waitKey(0)
    return
     
def smooth(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白画布
    smooth_image = np.zeros_like(image)

    # 遍历所有轮廓并平滑
    for cnt in contours:
        # 将轮廓点转换为(x, y)坐标
        points = cnt[:, 0, :]
        x, y = points[:, 0], points[:, 1]

        # 使用样条曲线进行平滑
        tck, u = splprep([x, y], s=5)
        unew = np.linspace(u.min(), u.max(), 100)
        smooth = splev(unew, tck)

        # 绘制平滑后的曲线
        smooth_points = np.array(smooth).T.round().astype(int)
        cv2.polylines(smooth_image, [smooth_points], False, (255, 255, 255), 1)
    cv2.imwrite(path, smooth_image)
    return smooth_image
    
def render_edge(case, save_path1, save_path2, step_idx=-1, show=False):
    mouth_mask = cv2.imread(f'/window/data/smile/out1/{case}/mouth_mask.png')
    
    best_params = torch.load(f'/window/data/smile/out1/{case}/para.pt')
    step = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][step_idx]
    os.makedirs(f'/window/data/smile/out1/{case}/step', exist_ok=True)
    with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
        tid_list = json.load(f)
    up_tensor, down_tensor = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step, half=False)
    renderer = get_renderer('Edge', focal_length=best_params['focal_length'], device='cuda:0')
    T = best_params['T']
    dist = best_params['dist']
    R = best_params['R']
    with torch.no_grad():        
        teeth_mesh = join_meshes_as_batch([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)

        out_im, _ = renderer(meshes_world=teeth_mesh, R=R, T=T, extra_mask=None)

        out_im[mouth_mask[...,0]==0]=0  
        up_edge, down_edge = deepmap_to_edgemap(out_im.detach().cpu().numpy(), up_tensor._N)
        # up_edge, down_edge = smooth(up_edge), smooth(down_edge)
        cv2.imwrite(save_path1, up_edge)
        cv2.imwrite(save_path2, down_edge)
        
        if show:
            cv2.imshow('mat', up_edge)
            cv2.waitKey(0)
            
    return   

def render_3d(case, save_path,step=-1, show=False):
    with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
        tid_list = json.load(f)
    best_params = torch.load(f'/window/data/smile/out1/{case}/para.pt')
    step_idx = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][step]
    os.makedirs(f'/window/data/smile/out1/{case}/step', exist_ok=True)
    up_tensor, down_tensor = get_target_teeth(f'/window/data/smile/TeethSimulation/{case}', tid_list, step_idx, type='scene', half=False)
    renderer = get_renderer('HardPhong', focal_length=best_params['focal_length'])
    T = best_params['T']
    dist = best_params['dist']
    R = best_params['R']
    with torch.no_grad():        
        teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)
        camera = PerspectiveCameras(device='cuda', focal_length=best_params['focal_length'], R=R, T=T)
        out_im = renderer(meshes_world=teeth_mesh, cameras=camera, extra_mask=None)
        out_im = (255*out_im[0,:,:,:3]/out_im[0,:,:,:3].max()).detach().cpu().numpy().astype(np.uint8)
        if show:
            cv2.imshow('mat', out_im)
            cv2.waitKey(0)
        cv2.imwrite(save_path, out_im)
    return  
    
if __name__=='__main__':
    path = '/window/data/smile/out1'
    case = 'C01002721259'
    # step = [file for file in natsort.natsorted(os.listdir(f'/window/data/smile/TeethSimulation/{case}')) if file.endswith('txt')][-1]
    # with open(f'/window/data/smile/TeethSimulation/{case}/models/tid_list.json', 'r')as f:
    #     tid_list = json.load(f)    
    # show_target_teeth(f'/window/data/smile/TeethSimulation/{case}',tid_list, step, half=True)
    # render_3d('C01002722687',f'/window/data/smile/out1/C01002722687/step/3d_0.png', 0)
    for case in tqdm(natsort.natsorted(os.listdir(path))[:1]):
        case = 'C01002721350'
        
        save_path = f'/window/data/smile/out1/{case}/step'
        
        os.makedirs(save_path, exist_ok=True)
        # print(case)
        # try:
        interface(case)
        render_depth_mask(case,f'{save_path}/depth.png')
        render_edge(case,f'{save_path}/up_edge.png',
                    f'{save_path}/down_edge.png')
        # smooth(f'{save_path}/up_edge.png')
        # smooth(f'{save_path}/down_edge.png')
        render_3d(case,f'{save_path}/3d.png')
        # except Exception as e:
        #     print(e)
        break
