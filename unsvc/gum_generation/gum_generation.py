import os

from gum_generation.predict_lib import generate_gum
from gum.gum_deformation.deformer import DeformDLL
from gum_generation.test_half_jaw import get_result


def generate_gum_with_deform(teeth_dict, jaw_name):
    jaw_type = (1, 2) if jaw_name == 'Upper' else (3, 4)
    result = generate_gum({fdi: tooth for fdi, tooth in teeth_dict.items() if fdi // 10 in jaw_type}, production=True)

    return DeformDLL(result[0], get_result(result))


if __name__ == '__main__':
    from utils.angle_align_tooth_loader import load, load_matrix, transform

    path = '/media/wuhuikai/data/projects/IntraOralCamEst/datasets/BC01000991177/新病例阶段_2021-11-28_15-24-13'
    teeth_dict = transform(load(path), load_matrix(path))
    for jaw_name, jaw_type in (('Upper', (1, 2)), ('Lower', (3, 4))):
        result = generate_gum({fdi: tooth for fdi, tooth in teeth_dict.items() if fdi//10 in jaw_type}, production=True)

        save_dir = os.path.join(path, jaw_name)
        os.makedirs(save_dir, exist_ok=True)

        result[0].export(save_dir + '/gum.ply')
        print(get_result(result), file=open(save_dir + '/result.txt', 'w'))
