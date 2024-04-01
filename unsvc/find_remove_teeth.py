import json
import os
import glob
import tqdm
import trimesh
import numpy as np


from natsort import natsorted


def load_all_teeth(teeth_folder_path, postfix='Crown', idx=0):
    teeth_path_dict = {int(os.path.basename(path).split('.')[0]): path for path in
                       glob.glob(os.path.join(teeth_folder_path, 'models', f'*._{postfix}.stl'))}

    step_files = natsorted(glob.glob(os.path.join(teeth_folder_path, 'step*.txt')))
    teeth_dict = {}
    for line in np.loadtxt(step_files[idx]):
        tid = int(line[0])

        translation_matrix = trimesh.transformations.translation_matrix(line[1:4])
        rotation_matrix = trimesh.transformations.quaternion_matrix(np.hstack([line[-1:], line[4:-1]]))

        teeth_dict[tid] = (trimesh.load(teeth_path_dict[tid]), translation_matrix, rotation_matrix)

    return teeth_dict
def find_all_normal_teeth_half_jaw(teeth_dict, half_jaw_idx):
    def create_incisor_collision_manager(jaw_idx):
        collision_manager_ = trimesh.collision.CollisionManager()
        collision_manager_.add_object(f'{jaw_idx}', teeth_dict[min([tid_ for tid_ in teeth_dict.keys() if tid_ // 10 == jaw_idx])])
        return collision_manager_

    def calc_collision_depth(collision_managers):
        is_collision, contact_data = collision_managers[0].in_collision_other(collision_managers[1], return_data=True)
        if not is_collision:
            return collision_managers[0].min_distance_other(collision_managers[1])
        return - max([data.depth for data in contact_data])

    def calc_sphere_collision(left_tooth, right_tooth):
        centroid_distance = np.linalg.norm(left_tooth.centroid - right_tooth.centroid)
        left_size = np.min(left_tooth.bounds[1] - left_tooth.bounds[0])
        right_size = np.min(right_tooth.bounds[1] - right_tooth.bounds[0])
        return centroid_distance - (left_size + right_size) / 2

    def find_all_normal_teeth(idx_list):
        def find_normal_tooth(normal_list_):
            if normal_list_[-1] == tid_list[-1]:
                return normal_list_

            current = collision_manager_dict[normal_list_[-1]]
            current_tooth = teeth_dict[normal_list_[-1]]
            distance_array = np.asarray(
                [(tid,
                  calc_collision_depth([current, collision_manager_dict[tid]]),
                  calc_sphere_collision(current_tooth, teeth_dict[tid])) for tid in tid_list[tid_list.index(normal_list_[-1])+1:]]
            )
            distance_array = distance_array[distance_array[:, 1] > -1]
            distance_array = distance_array[distance_array[:, 2] > -2]
            if len(distance_array) == 0:
                return normal_list_
            tid = distance_array[np.argmin(np.abs(distance_array[:, 1])), 0]
            normal_list_.append(int(tid))
            return find_normal_tooth(normal_list_)

        tid_list = sorted([tid_ for tid_ in teeth_dict.keys() if tid_ // 10 == idx_list[0]], reverse=True) + sorted([tid_ for tid_ in teeth_dict.keys() if tid_ // 10 == idx_list[1]])
        tid_list = [tid_ for tid_ in tid_list if tid_%10 != 8]
        # find all normal teeth
        return find_normal_tooth([tid_list[0]])

    def calc_dist(tid_, normal_list_):
        idx = normal_list_.index(tid_)
        distance = []
        if idx - 1 >= 0:
            distance.append(np.abs(calc_collision_depth((collision_manager_dict[tid_], collision_manager_dict[normal_list_[idx - 1]]))))
        if idx + 1 < len(normal_list_):
            distance.append(np.abs(calc_collision_depth((collision_manager_dict[tid_], collision_manager_dict[normal_list_[idx + 1]]))))
        return np.mean(distance)

    # check left/right incisor
    depth = calc_collision_depth([create_incisor_collision_manager(jaw_idx) for jaw_idx in half_jaw_idx])
    assert depth > - 1.0, depth

    # collision_manager
    collision_manager_dict = {}
    for tid in teeth_dict:
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object(f'{tid}', teeth_dict[tid])
        collision_manager_dict[tid] = collision_manager

    # find_all_normal_teeth
    normal_list = find_all_normal_teeth(half_jaw_idx)
    reversed_normal_list = find_all_normal_teeth(half_jaw_idx[::-1])
    if normal_list == reversed_normal_list:
        return normal_list

    # diff
    diff_list = sorted(list(set(normal_list) - set(reversed_normal_list)))
    reversed_diff_list = sorted(list(set(reversed_normal_list) - set(normal_list)))
    for tid, reversed_tid in zip(diff_list, reversed_diff_list):
        dist = calc_dist(tid, normal_list)
        reversed_dist = calc_dist(reversed_tid, reversed_normal_list)
        if dist > reversed_dist:
            normal_list.remove(tid)
            normal_list.append(reversed_tid)
    return normal_list


def find_remove_teeth(teeth_dict):
    teeth_dict = {tid: tooth.apply_transform(translation @ rotation) for tid, (tooth, translation, rotation) in teeth_dict.items()}
    return find_all_normal_teeth_half_jaw(teeth_dict, (1, 2)) + find_all_normal_teeth_half_jaw(teeth_dict, (3, 4))


if __name__ == '__main__':
    def main():
        root_dir = '/media/wuhuikai/data/data/TeethArrangementStepData'
        root_dir = '/mnt/hdd/data/smile/Teeth_simulation_10K/'
        for path in tqdm.tqdm(sorted(glob.glob(os.path.join(root_dir, 'C*')))):
            tooth_list = find_remove_teeth(load_all_teeth(path, 'Root', -1))
            with open(os.path.join(path, 'tid_list.json'), 'w') as f:
                json.dump(tooth_list, f)
    # tooth_list = find_remove_teeth(load_all_teeth('/mnt/hdd/data/smile/Teeth_simulation_10K/C01002721260', 'Root', -1))
    # print(tooth_list)
    main()
