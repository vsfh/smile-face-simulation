import sys
sys.path.append('.')
sys.path.append('..')

from gum_generation.predict_lib import generate_gum
from gum.gum_deformation.deformer import DeformDLL
from gum_generation.test_half_jaw import get_result
from scipy.spatial.transform import Rotation as R

import smile_utils
from glob import glob
import trimesh
import os
import numpy as np
import natsort
import json
import tqdm

def generate_gum_with_deform(teeth_dict, jaw_name):
    jaw_type = (1, 2) if jaw_name == 'Upper' else (3, 4)
    result = generate_gum({fdi: tooth for fdi, tooth in teeth_dict.items() if fdi // 10 in jaw_type}, production=True)

    return DeformDLL(result[0], get_result(result)).get_gum()

def infer(case, step_idx=-1, show=False):
    img_folder = f'/mnt/hdd/data/smile/TeethSimulation/{case}'
    save_path = os.path.join(img_folder, 'models', 'final')
    os.makedirs(save_path, exist_ok=True)
    step_idx = [file for file in natsort.natsorted(os.listdir(img_folder)) if file.endswith('txt')][step_idx]
    with open(os.path.join(img_folder, 'models', 'tid_list.json'), 'r')as f:
        tid_list = json.load(f)
    step_one_dict = {}
    for arr in np.loadtxt(os.path.join(img_folder, step_idx)):
        step_one_dict[int(arr[0])] = arr[1:]
    teeth_dict = smile_utils.trimesh_load_apply({int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob(os.path.join(img_folder, 'models', '*._Crown.stl'))},
                                                step_one_dict, tid_list)
    up_gum = generate_gum_with_deform(teeth_dict, 'Upper')
    down_gum = generate_gum_with_deform(teeth_dict, 'Lower')
    up_gum.export(os.path.join(save_path, 'up.stl'))
    down_gum.export(os.path.join(save_path, 'down.stl'))
    if show:
        scene = trimesh.Scene()
        for i,mesh in teeth_dict.items():
            scene.add_geometry(mesh)
        scene.add_geometry(up_gum)
        scene.add_geometry(down_gum)
        scene.show()
    
if __name__=='__main__':
    for case in tqdm.tqdm(os.listdir(f'/mnt/hdd/data/smile/TeethSimulation/')):
        print(case)
        infer(case)
        # break
    