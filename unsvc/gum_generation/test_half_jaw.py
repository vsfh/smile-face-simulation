import os, click, time, glob, pickle, json, os.path as osp
import numpy as np
import tqdm
import trimesh
from gum_generation.predict_lib import generate_gum
from typing import Dict
from natsort import natsorted
from multiprocessing import Pool


def load_all_teeth(teeth_folder_path, idx=0):
    teeth_dict = {int(os.path.basename(path).split('.')[0]): trimesh.load(path) for path in glob.glob(os.path.join(teeth_folder_path, f'*._Crown.stl'))}
    with open(os.path.join(teeth_folder_path, 'tid_list.json')) as f:
        tid_list = json.load(f)
    teeth_dict = {fdi: tooth for fdi, tooth in teeth_dict.items() if fdi in tid_list}
    step_files = natsorted(glob.glob(os.path.join(teeth_folder_path, 'step*.txt')))
    for line in np.loadtxt(step_files[idx]):
        tid = int(line[0])
        if tid not in teeth_dict:
            continue

        translation_matrix = trimesh.transformations.translation_matrix(line[1:4])
        rotation_matrix = trimesh.transformations.quaternion_matrix(np.hstack([line[-1:], line[4:-1]]))

        teeth_dict[tid] = teeth_dict[tid].apply_transform(translation_matrix @ rotation_matrix)

    return teeth_dict


def read_teeth_mesh(case_dir):
    fps = glob.glob(osp.join(case_dir, "*.stl"))
    return {osp.basename(fp)[-6:-4]: trimesh.load(fp) for fp in fps if len(osp.basename(fp)) == 6}


def get_result(result):
    gum, handle_dict, surf_point_dict, tooth_boundary_dict, gum_faceid_dict, tooth_lingual_side_vid_dict, \
    tooth_buccal_side_vid_dict, gum_distal_side_vid_dict, sorted_tids, gum_params, ori_gum_info = result

    out = {'handle_dict': {str(k): [int(i) for i in t] for k, t in handle_dict.items()},
           'surf_point_dict': {str(k): [int(i) for i in t] for k, t in surf_point_dict.items()},
           'tooth_boundary_dict': {str(tid): t.tolist() for tid, t in tooth_boundary_dict.items()},
           'gum_faceid_dict': {str(k): [int(i) for i in t] for k, t in gum_faceid_dict.items()},
           'tooth_lingual_side_vid_dict': {str(tid): v.tolist() for tid, v in tooth_lingual_side_vid_dict.items()},
           'tooth_buccal_side_vid_dict': {str(tid): v.tolist() for tid, v in tooth_buccal_side_vid_dict.items()},
           'gum_distal_side_vid_dict': {str(tid): v.tolist() for tid, v in gum_distal_side_vid_dict.items()},
           'sorted_tids': sorted_tids,
           'gum_params': gum_params,
           'ori_gum_info': ori_gum_info
           }
    out = json.dumps(out)
    return out


def process(case_dir):
    if len(glob.glob(os.path.join(case_dir, f'*._Crown.stl'))) != len(
            glob.glob(os.path.join(case_dir, f'*._Root.stl'))):
        return

    if os.path.isdir(os.path.join(case_dir, 'Upper')) and os.path.isdir(os.path.join(case_dir, 'Lower')):
        return

    teeth_dict = load_all_teeth(case_dir)

    for jaw_name, jaw_set in (('Upper', (1, 2)), ('Lower', (3, 4))):
        try:
            if os.path.isdir(os.path.join(case_dir, jaw_name)):
                continue

            result = generate_gum({fdi: tooth for fdi, tooth in teeth_dict.items() if fdi // 10 in jaw_set},
                                  production=True)

            save_dir = os.path.join(case_dir, jaw_name)
            os.makedirs(save_dir, exist_ok=True)

            gum = result[0]
            gum.export(save_dir + '/gum.ply')

            json.dump({'gum_vertices': np.asarray(gum.vertices).tolist(),
                       'gum_faces': np.asarray(gum.faces).tolist()}, open(osp.join(save_dir, 'gum.json'), 'w'))

            print(get_result(result), file=open(save_dir + '/result.txt', 'w'))
        except:
            save_dir = os.path.join(case_dir, jaw_name)
            if os.path.isdir(save_dir):
                os.system(f'rm -rf {save_dir}')


@click.command()
@click.option('--data_path', default="/media/wuhuikai/data/data/Teeth_simulation_10K/*/models")
def run(data_path):
    case_dirs = glob.glob(data_path)
    with Pool(4) as p:
        list(tqdm.tqdm(p.imap_unordered(process, case_dirs), total=len(case_dirs)))


if __name__ == '__main__':
    run()
