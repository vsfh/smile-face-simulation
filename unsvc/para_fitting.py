import torch
import torch.nn as nn
import numpy as np
import cv2
import smile_utils
from pytorch3d.transforms.rotation_conversions import *
import natsort
from PIL import Image
import time
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
import os
import glob
from natsort import natsorted
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import imageio
# rendering components
from pytorch3d.renderer import (
	RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
	PerspectiveCameras,HardPhongShader, PointLights
)
import onnxruntime

is_onnx = False
onnx_opt_dict = {
    'smile_sim_lip_preserve-yolov5':{'input':['images'],'output':['output'],'path':'/window/triton/backup_model/smile_sim_lip_preserve-yolov5/1'},
    'smile_sim_lip_preserve-edge_net':{'input':['data'],'output':['output'],'path':'/window/triton/backup_model/smile_sim_lip_preserve-edge_net/1'},
    'cls-ensemble':{'input':['images'],'output':['output'],'path':'/window/triton/backup_model/cls-yolov5s/1'},
    'new_smile_wo_edge_gan':{'input':['input_image','mask'],'output':['align_img'],'path':'/window/triton/backup_model/new_smile_wo_edge_gan/1'},
}
if is_onnx:
    onnx_sess_dict = {}
    for k,v in onnx_opt_dict.items():
        onnx_sess_dict[k] = onnxruntime.InferenceSession(os.path.join(v['path'],'model.onnx'),providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        
def onnx_infer(data, sess_name):
    input = {name: data[i] for i,name in enumerate(onnx_opt_dict[sess_name]['input'])}
    output_names = onnx_opt_dict[sess_name]['output']
    if len(output_names)==1:
        output = onnx_sess_dict[sess_name].run([], input)[0]
        return output
    else:
        print('error', sess_name)


def _find_objs(image):
    yolo_input_shape = (640, 640)
    original_h, original_w = image.shape[:2]
    resized_image, meta = smile_utils.resize_and_pad(image, yolo_input_shape)
    offsets = meta['offsets']
    scale = meta['scale']
    input_imgs = smile_utils.normalize_img(resized_image)
    output = onnx_infer([input_imgs],'smile_sim_lip_preserve-yolov5', )
    output = output[0]
    xywh = output[:, :4]
    probs = output[:, 4:5] * output[:, 5:]

    objs = []
    num_class = probs.shape[-1]

    for i in range(num_class):
        p = probs[:, i]
        if p.max() < 0.04:
            a = np.array([0,0,0,0])
            objs.append(a)
            continue
        idx = p.argmax()

        x, y, w, h = xywh[idx]
        coords = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

        coords[[0, 2]] -= offsets[0]
        coords[[1, 3]] -= offsets[1]
        coords /= scale

        coords = smile_utils.loose_bbox(coords, (original_w, original_h))
        objs.append(coords)
    objs = np.array(objs, dtype=int)
    return objs

def _seg_mouth(image):
    seg_input_shape = (256, 256)
    resized_image, _ = smile_utils.resize_and_pad(image, seg_input_shape)
    input_imgs = smile_utils.normalize_img(resized_image)
    output = onnx_infer([input_imgs], 'smile_sim_lip_preserve-edge_net')
    output = np.transpose(output[0], (1, 2, 0))
    output = smile_utils.sigmoid(output)
    return output

def apply_style(img, mean, std, mask=None, depth=None):
    if len(mask.shape) == 2:
        mask = mask[..., None]
    if mask.shape[2] == 3:
        mask = mask[..., :1]
    mask = mask.astype(np.uint8)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")

    src_mean, src_std = cv2.meanStdDev(img_lab, mask=mask)
    img_lab = (img_lab - src_mean.squeeze()) / src_std.squeeze() * std.squeeze() + mean.squeeze()
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    img_rgb = img_rgb.astype(np.float32) * (depth[...,None] * 0.3 + 0.7)* (depth[...,None] * 0.3 + 0.7)

    mask = mask.astype(np.float32)

    smooth_mask = cv2.blur(mask, (5, 5))
    smooth_mask = smooth_mask[..., None]

    result = img_rgb * smooth_mask + (1 - smooth_mask) * img
    result = result.astype(np.uint8)

    return result

res_list = ['mouth_mask', 'edge', 'down_edge', 'upper_edge', 'teeth_mask']
def detection_and_segmentation(face_img, save_path=None):
    height, width = face_img.shape[:2]

    # step 1. find mouth obj
    objs = _find_objs(face_img)

    mouth_objs = objs[2]
    x1, y1, x2, y2 = mouth_objs
    if x1==x2 and y1==y2:
        raise Exception('not smile image')

    w, h = (x2 - x1), (y2 - y1)
    
    half = max(w, h) * 1.1 / 2
    # half = max(w, h) / 2
    

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    x1, y1, x2, y2 = smile_utils.loose_bbox([x1, y1, x2, y2], (width, height))
    x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

    template = cv2.resize(face_img, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
    mouth = template[y: y + 256, x: x + 256]

    seg_result = _seg_mouth(mouth)

    mouth_mask = (seg_result[..., 0] > 0.6).astype(np.float32)
    teeth_mask = (seg_result[..., 4] > 0.6).astype(np.uint8)
    edge = (seg_result[..., 1] > 0.6).astype(np.float32)
    up_edge = (seg_result[..., 3] > 0.6).astype(np.float32)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        for i in range(seg_result.shape[-1]):
            a = (seg_result[..., i] > 0.6).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f'{res_list[i]}.png'), 255*a)
            # cv2.imshow(f'img{i}', 255*a)
            # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_path, f'smile.png'), mouth[...,::-1])
        return
        
    contours, _ = cv2.findContours(mouth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours)>1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        idx = np.argmax(np.array(area))
        # print(idx)
        for k in range(len(contours)):
            if k!= idx:
                mouth_mask = cv2.drawContours(mouth_mask, contours, k, 0, cv2.FILLED)
                
    return {'mouth_mask': mouth_mask,
            'teeth_mask': teeth_mask,
            'edge': edge,
            'up_mask': up_edge,
            'mouth_img': mouth,
            'face_img': template,
            'position': (y,x),
            'width_height': (width, height, half),
            }

class SoftEdgeShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, zfar=200, znear=1):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.zfar = zfar
        self.znear = znear

    def forward(self, fragments,  meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        extra_mask = kwargs.get("extra_mask", None)

        mask = fragments.pix_to_face >= 0
        if extra_mask is not None:
            mask = mask * extra_mask

        prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
        prob_map = torch.sum(prob_map, -1)
        alpha = torch.prod((1.0 - prob_map), dim=-1, keepdim=True)
        alpha = 1 - alpha    
                
        zbuf = fragments.zbuf[...,-1]
        zbuf[zbuf==-1] = 1e10
        zbuf = torch.cat((torch.ones_like(zbuf[0][None])*1e10,zbuf),0)
        zbuf_mask = torch.argmin(zbuf, 0, keepdim=True)
        
        for i in range(len(prob_map)):
            prob_map[i] = prob_map[i]*(zbuf_mask[0]==i+1)
        return prob_map, 0

class EdgeShader(nn.Module):
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

class Model(nn.Module):
    def __init__(self, meshes, image_ref, teeth_region, focal_length, axis_angle, translation,
                raster_settings):
        super().__init__()
        self.up_mesh, self.down_mesh = meshes
        self.device = self.up_mesh.device

        image_ref = torch.from_numpy(image_ref[np.newaxis, ...].astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        mouth_mask, up_mask = teeth_region
        mouth_mask = torch.from_numpy(mouth_mask[np.newaxis, ...].astype(np.float32))
        up_mask = torch.from_numpy(up_mask[np.newaxis, ...].astype(np.float32))
        
        self.register_buffer('mouth_mask', mouth_mask)
        self.register_buffer('up_mask', up_mask)
        
        self.focal_length = nn.Parameter(torch.tensor(focal_length, dtype=torch.float32, device=self.device))
        
        self.angle_x = nn.Parameter(torch.tensor(axis_angle[:1], dtype=torch.float32, device=self.device))
        self.angle_y = nn.Parameter(torch.tensor(axis_angle[1:2], dtype=torch.float32, device=self.device))
        self.angle_z = nn.Parameter(torch.tensor(axis_angle[2:3], dtype=torch.float32, device=self.device))

        self.x = nn.Parameter(torch.tensor((translation[0],), dtype=torch.float32, device=self.device))
        self.y = nn.Parameter(torch.tensor((translation[1],), dtype=torch.float32, device=self.device))
        self.z = nn.Parameter(torch.tensor((translation[2],), dtype=torch.float32, device=self.device))

        self.dist_x = nn.Parameter(torch.tensor((-0,), dtype=torch.float32, device=self.device))
        self.dist_y = nn.Parameter(torch.tensor((-0,), dtype=torch.float32, device=self.device))
        self.dist_z = nn.Parameter(torch.tensor((-0,), dtype=torch.float32, device=self.device))

        self.cameras = PerspectiveCameras(device=self.device, focal_length=self.focal_length)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftEdgeShader(
                blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.)))
        )
        # self.seg_model = torch.jit.load('traced_bert.pt')
        # self.seg_model.to(self.device)
        # for param in self.seg_model.parameters():
        #     param.requires_grad = False

    def render(self):
        with torch.no_grad():
            axis_angles = torch.cat([self.angle_x, self.angle_y, self.angle_z])
            R = axis_angle_to_matrix(axis_angles[None, :])

            translation = torch.cat([self.x, self.y, self.z])
            T = translation[None, :]
            image = self.renderer(meshes_world=self.meshes, R=R, T=T, extra_mask=None)
        return image

    def forward(self):
        axis_angles = torch.cat([self.angle_x, self.angle_y, self.angle_z])
        R = axis_angle_to_matrix(axis_angles[None, :])


        translation = torch.cat([self.x, self.y, self.z])
        T = translation[None, :]

        down_offset = torch.cat([self.dist_x, self.dist_y, self.dist_z])
        meshes = join_meshes_as_batch([self.up_mesh, self.down_mesh.offset_verts(down_offset)], include_textures=False)
        
        image, zbuf_mask = self.renderer(meshes_world=meshes.clone(), R=R, T=T, extra_mask=None)  
        # image = torch.where(image != 0, torch.ones_like(image), image)
        # image = image * self.mouth_mask
                    
        im = image.view(image.shape[0],1,*image.shape[1:3])
        
        im = smile_utils.erode(smile_utils.dilate(im))
        im = smile_utils.dilate(im,) -im
        im = smile_utils.dilate(torch.sum(im, 0).clip(0,1),3)
        pred = im.clip(0,1)*self.mouth_mask
        
        label = self.image_ref[None]

        loss = torch.sum(((pred - label)) ** 2)
        return loss, pred, label, R, T, down_offset
    
def fitting(seg_res, tooth_dict, step_list, tid_list, save_path=None, save_gif=False, device='cuda'):
    step1 = step_list['step_0']
    teeth = smile_utils.load_teeth(tooth_dict, tid_list, type='tooth', half=True, sample=True, voxel_size=1.0)
    
    mouth_mask = seg_res['mouth_mask']
    teeth_mask = seg_res['teeth_mask']
    # upper_lower = seg_res['upper_lower']
    up_mask = seg_res['up_mask']

    closed_teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE,
                                         kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)))
    closed_up_teeth_mask = cv2.morphologyEx(up_mask, cv2.MORPH_CLOSE,
                                         kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    image_ref = seg_res['edge']
    up_mesh = smile_utils.apply_step(teeth, step1, mode='up', add=False, num_teeth=5)
    up_tensor = smile_utils.meshes_to_tensor(up_mesh, device=device)
    down_mesh = smile_utils.apply_step(teeth, step1, mode='down', add=False, num_teeth=5)
    down_tensor = smile_utils.meshes_to_tensor(down_mesh, device=device)
    up_bbox = up_mesh[0].get_axis_aligned_bounding_box()
    ref_y, ref_z = up_bbox.min_bound[1], up_bbox.max_bound[2]
    ref = -ref_y * np.sin(np.deg2rad(10)) + ref_z * np.cos(np.deg2rad(10))

    index = np.argwhere(np.max(up_mask[:, 100:150], axis=1) > 0)
    if len(index)==0:
        mouth_mid = 128
    else:
        mouth_mid = np.max(index)

    focal_length = 30
    init_z = 1000
    scale = focal_length * 128 / init_z
    init_y = (128 - mouth_mid) / scale + ref

    translation = (0, init_y, init_z)
    axis_angle = [np.deg2rad(-80), np.deg2rad(0), 0]

    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0,
        faces_per_pixel=1,
        perspective_correct=True,
        cull_backfaces=True
    )

    model = Model(meshes=[up_tensor, down_tensor], image_ref=image_ref,
                  teeth_region=(closed_teeth_mask, closed_up_teeth_mask),
                  focal_length=focal_length,
                  translation=translation,
                  axis_angle=axis_angle,
                  raster_settings=raster_settings).to(device)

    params = [{'params': model.angle_x, 'lr': 0, 'name': 'angle_x'},
              {'params': model.angle_y, 'lr': 0, 'name': 'angle_y'},
              {'params': model.angle_z, 'lr': 0, 'name': 'angle_z'},

              {'params': model.x, 'lr': 3e-4, 'name': 'x'},
              {'params': model.y, 'lr': 3e-4, 'name': 'y'},
              {'params': model.z, 'lr': 0, 'name': 'z'},

              {'params': model.dist_x, 'lr': 0, 'name': 'dist_x'},
              {'params': model.dist_y, 'lr': 0, 'name': 'dist_y'},
              {'params': model.dist_z, 'lr': 0, 'name': 'dist_z'},

              {'params': model.focal_length, 'lr': 1e-5, 'name': 'focal_length'}]

    optimizer = torch.optim.SGD(
        params,
        lr=1e-3, momentum=0.1)

    min_loss = np.Inf

    output_file = 'output.gif'
    out = []
    update = False
    for i in range(200):
        optimizer.zero_grad()
        loss, pred, label, R, T, dist = model()

        if i % 10==0 and save_gif:
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            diff = np.zeros((256, 256, 3), dtype=np.float32)
            diff[..., 0] = pred
            diff[..., 1] = label
            out.append((diff*255).astype(np.uint8))
        if loss.item() < min_loss:
            # T[0,2] += 50
            best_opts = {
                'focal_length': model.focal_length.detach().clone(),
                'R': R.detach().clone(),
                'T': T.detach().clone(),
                'dist': dist.detach().clone(),
            }
            min_loss = loss.item()
            min_step = i
        if i > 50 and not update:
            update = True
            for group in optimizer.param_groups:
                if group['name'] in ['angle_x', 'angle_y', 'angle_z']:
                    group['lr'] = 1e-7
                elif group['name'] in ['x', 'y', 'z']:
                    group['lr'] = 1e-5
                elif group['name'] in ['focal_length']:
                    group['lr'] = 5e-6
                else:
                    pass    
        if (i - min_step) > 100:
            break
        loss.backward()
        optimizer.step()
    
    if save_path is not None:
        if save_gif:
            imageio.mimsave(os.path.join(save_path, output_file), out, duration=100)

        torch.save(best_opts, os.path.join(save_path, 'para.pt'))
    return 0,0,0,False

def render_target(teeth, step):
    device = 'cuda:0'
    best_opts = torch.load('')
    opt_cameras = PerspectiveCameras(device=device, focal_length=best_opts['focal_length'])
    lights = PointLights(device=device, ambient_color=((0.9, 0.9, 0.9),), location=[[2.0, -60.0, -12.0]])
    edge_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=opt_cameras,
            raster_settings=RasterizationSettings(
                image_size=256,
                faces_per_pixel=1,
                perspective_correct=True,
                cull_backfaces=True,
            )
        ),
        shader=HardPhongShader(device=device, cameras=opt_cameras, lights=lights,blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.)))
    )

    # teeth_root = utils.load_teeth(tooth_root_dict, type='root', half=False, sample=False)
    edge_dict = {}
    depth_dict = {}
    id_dict = {}
    start = time.time()

    up_mesh = smile_utils.apply_step(teeth, step, mode='up', add=False, num_teeth=6)
    up_tensor = smile_utils.meshes_to_tensor(up_mesh,'scene', device=device)
    down_mesh = smile_utils.apply_step(teeth, step, mode='down', add=False, num_teeth=6)
    down_tensor = smile_utils.meshes_to_tensor(down_mesh,'scene', device=device)

    mid = len(up_mesh)

    with torch.no_grad():
        R = best_opts['R']
        T = best_opts['T']
        dist = best_opts['dist']
        dist[-1] = dist[-1]
        
        teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)

        out_im = edge_renderer(meshes_world=teeth_mesh, R=R, T=T)
        out_im = (255*out_im[0,:,:,:3]/out_im[0,:,:,:3].max()).detach().cpu().numpy().astype(np.uint8)

    return edge_dict, depth_dict, id_dict, True

def render(params, tooth_dict, step_list, **kwargs):
    teeth_mask = kwargs['teeth_mask']
    return

def _gan(inputs, network_name, tinf=None):
    output = onnx_infer(inputs, network_name, )
    output = output[0].transpose(1, 2, 0)
    output = (output + 1.0) / 2.0
    return output

def smile_generation_based_on_edge(seg_res, edge_list, depth_dict, id_dict):
    ori_mask = seg_res['mouth_mask']
    # ori_mask = cv2.dilate(ori_mask, np.ones((3,3)))
    mouth_img = seg_res['mouth_img']
    face_img = seg_res['face_img']
    wh = seg_res['width_height']
    xy = seg_res['position']
    kernel = np.ones((4,4),np.uint8)

    mouth_lab = cv2.cvtColor(mouth_img, cv2.COLOR_RGB2LAB)
    ori_teeth_mask = cv2.erode(seg_res['teeth_mask'], kernel)
    target_mean, target_std = cv2.meanStdDev(mouth_lab, mask=ori_teeth_mask)
    smile_img_list = {}
    mouth_img = mouth_img/255
    
    for edge_idx in edge_list.keys():

        up_edge = edge_list[edge_idx]
        down_edge = id_dict[edge_idx]
        
        edge = np.logical_or(up_edge, down_edge).astype(np.float32)
        
        depth_ori = depth_dict[edge_idx]
        depth = depth_ori.copy()
        depth[depth_ori>0] = 1
        depth[ori_mask==0] = 0

        edge[edge>0] = 1
        up_edge[up_edge>0] = 1
        
        contour1, _ = cv2.findContours(up_edge.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour2, _ = cv2.findContours(down_edge.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tmask = np.zeros_like(edge)
        cv2.drawContours(tmask, contour1, -1, (1), thickness=cv2.FILLED)
        cv2.drawContours(tmask, contour2, -1, (1), thickness=cv2.FILLED)
        
        edge = edge[...,None].repeat(3,2)
        up_edge = up_edge[...,None].repeat(3,2)
        
        mask = ori_mask[...,None].repeat(3,2)
        tmask = tmask[...,None].repeat(3,2)

        cond = edge*mask*0.1 + up_edge*mask*0.5 + tmask*mask*(1-edge) + mouth_img*(1-mask)
        network_name = 'smile_sim_lip_preserve-up_net'
        aligned_mouth = _gan([smile_utils.normalize_img(cond)], network_name, )
        aligned_mouth = aligned_mouth.clip(0,1)*255
        aligned_mouth = aligned_mouth.astype(np.uint8)
        
        depth_ori = cv2.dilate(depth_ori.repeat(3,2), kernel)
        
        aligned_mouth = apply_style(aligned_mouth.copy(), target_mean, target_std, mask=depth[...,0], depth=depth_ori[...,0])
        
        aligned_mouth = mouth_img*(1-mask)*255 + aligned_mouth*mask
        local_face = face_img.copy()
        local_face[xy[0]: xy[0] + 256, xy[1]: xy[1] + 256] = aligned_mouth
        local_face = cv2.resize(local_face, wh)
        smile_img_list[edge_idx] = local_face
    return smile_img_list

def blend_images(image_a, image_b, opacity):
    blended_image = cv2.addWeighted(image_a, 1 - opacity, image_b, opacity, 0)
    return blended_image

class predictor(object):
    def __init__(self) -> None:
        pass
    def predict(self, face_img: np.array, tooth_dict, step_list, save_path=None):
        seg_res = detection_and_segmentation(face_img, save_path)
        fitting(seg_res, tooth_dict, step_list, save_path, device='cuda')
        return
  
def visualize_teeth(face_img, tooth_dict, step_list):
    import trimesh
    seg_res = detection_and_segmentation(face_img)
    step1 = smile_utils.load_step_file(step_list['step_0'])
    upper_lower = seg_res['upper_lower']
    up_mesh = smile_utils.trimesh_load_apply(tooth_dict, step1)
    up_mesh[11].show()

def load_teeth_step(img_folder):
    tooth_dict = {int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob.glob(os.path.join(img_folder, 'models', '*._Root.stl'))}
    steps_dict = {}
    step_one_dict = {}
    
    step_list = natsorted(glob.glob(img_folder+'/models/*.txt'))
    for arr in np.loadtxt(step_list[0]):
        trans = np.eye(4,4)
        trans[:3,3] = arr[1:4]
        trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
        step_one_dict[str(int(arr[0]))] = trans
    steps_dict['step_0'] = step_one_dict
    return steps_dict, tooth_dict

def infer():
    import json
    path = '/window/data/smile/out1'
    for case in tqdm(natsort.natsorted(os.listdir(path))[:200]):
        case = 'C01002721204'
        img_folder = os.path.join('/window/data/smile/TeethSimulation/',case)
        steps_dict, tooth_dict = load_teeth_step(img_folder)
        save_path = os.path.join(path,case)
        with open(os.path.join(img_folder,'models','tid_list.json'),'r') as f:
            tid_list = json.load(f)
        seg_dict = {}
        seg_dict['mouth_mask'] = cv2.imread(os.path.join(save_path,'mouth_mask.png'))[...,0]/255
        seg_dict['edge'] = cv2.imread(os.path.join(save_path,'edge.png'))[...,0]/255
        seg_dict['teeth_mask'] = cv2.imread(os.path.join(save_path,'teeth_mask.png'))[...,0]/255
        seg_dict['up_mask'] = cv2.imread(os.path.join(save_path,'upper_edge.png'))[...,0]/255
        fitting(seg_dict, tooth_dict, steps_dict, tid_list, save_path, save_gif=True)
        break

def test_single():
    path = '/window/data/smile/out1'
    for img_path in glob.glob(f'{path}/*/smile.png'):
        img = cv2.imread(img_path)
        print(img_path)
        cv2.imshow('img', img)
        cv2.waitKey(0)

def img_process():
    path = '/window/data/smile/out1'
    for img_folder in tqdm(glob.glob(f'{path}/C*')):
        # img_folder = '/window/data/smile/out1/C01002721978'
        img = cv2.imread(os.path.join(img_folder, 'face1.jpg'))
        # img.save('test.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.rot90(img, 1)
        save_path = os.path.join('/window/data/smile/out1', os.path.basename(img_folder))
        detection_and_segmentation(img, save_path)
        
        # try:
        #     detection_and_segmentation(img, save_path)
        # except Exception as e:
        #     print(e)
        # break
if __name__=="__main__":
    # test_single()
    # img_process()  
    infer()