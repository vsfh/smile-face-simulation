import json
import time
import ctypes
import trimesh

import numpy as np
import os.path as osp

from typing import Dict
from ctypes import cdll


def get_random_transform():
    from scipy.spatial.transform import Rotation

    max_degrees = 20
    rand_angle = np.random.rand(3) * max_degrees
    rot = Rotation.from_euler('xyz', rand_angle, degrees=True).as_matrix()

    _max, _min = 1, -1
    trans = np.random.rand(3) * (_max - _min) + _min
    return rot, trans


class ToothTransformation(ctypes.Structure):
    _fields_ = [("tid", ctypes.c_int),
                ("rotation", ctypes.c_double * 9),
                ("translation", ctypes.c_double * 3)]


class DeformDLL:
    lib = cdll.LoadLibrary(osp.join(osp.dirname(__file__),
                                    "libchohotech_gum_deform.so"))

    def __init__(self, gum: trimesh.Trimesh, api_result_str: str):
        json_data = json.loads(api_result_str.replace('\'', '\"'))
        self.tids = [int(tid) for tid in json_data['tooth_boundary_dict'].keys()]
        self.tooth_contour_dict = {tid: np.array(json_data['tooth_boundary_dict'][str(tid)]) for tid in self.tids}
        self.gum_vertices = np.array(gum.vertices)
        self.gum_faces = np.array(gum.faces)

        self.ori_gum_vertices = np.copy(self.gum_vertices)
        self.ori_gum_faces = np.copy(self.gum_faces)

        # for debug
        self.handle_dict = json_data['handle_dict']
        self.json_data = json_data

        # initialize gum deformer
        self.initialize_gum_deformer(self.gum_vertices, self.gum_faces, api_result_str)

    def initialize_gum_deformer(self, gum_vertices, gum_faces, api_result_str: str):
        # create gum deformer
        c_json_file = ctypes.c_char_p(api_result_str.encode('utf-8'))
        self.lib.create_gum_deformer.restype = ctypes.c_void_p

        vs = np.array(gum_vertices).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        fs = np.array(gum_faces).astype('i4').flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.p_gd = ctypes.c_void_p()
        self.lib.create_gum_deformer(
            vs, ctypes.c_uint(len(gum_vertices) * 3),
            fs, ctypes.c_uint(len(gum_faces) * 3),
            c_json_file, ctypes.c_uint(len(api_result_str)),
            ctypes.byref(self.p_gd))

    def deform(self, tt_dict: Dict[int, np.ndarray] = None):
        tt_list = [self.to_c_tooth_transform(tid, T) for tid, T in tt_dict.items()]
        tt_list = (ToothTransformation * len(tt_list))(*tt_list)
        n_teeth = len(tt_list)

        # deform
        deformed_vs = np.zeros(100000 * 3)
        n_deformed_vs = ctypes.c_uint(0)
        deformed_fs = np.zeros(100000 * 3, dtype='i4')
        n_deformed_fs = ctypes.c_uint(0)
        ret_str = ctypes.create_string_buffer(300000)
        n_ret_str = ctypes.c_uint(0)

        self.lib.deform(self.p_gd, tt_list, ctypes.c_int(n_teeth),
                        deformed_vs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.byref(n_deformed_vs),
                        deformed_fs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        ctypes.byref(n_deformed_fs),
                        ret_str,
                        ctypes.byref(n_ret_str))

        # return
        self.gum_vertices = deformed_vs[:n_deformed_vs.value].reshape((-1, 3))
        self.gum_faces = deformed_fs[:n_deformed_fs.value].reshape((-1, 3)).astype('i4')

    def show(self):
        import open3d as o3d
        # deformed gum
        m1 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.gum_vertices),
                                       o3d.utility.Vector3iVector(self.gum_faces))
        m1.compute_vertex_normals()
        m1.paint_uniform_color(np.array([224, 132, 131]) / 255.)

        # original gum
        m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.ori_gum_vertices),
                                       o3d.utility.Vector3iVector(self.ori_gum_faces))
        m2.compute_vertex_normals()

        i = 0
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        mesh_list = [[m1], [m2]]
        for m in mesh_list[0]:
            vis.add_geometry(m)

        def func(vis: o3d.visualization.VisualizerWithKeyCallback):
            nonlocal i, mesh_list
            for m in mesh_list[i]:
                vis.remove_geometry(m, False)

            i = 1 - i
            for m in mesh_list[i]:
                vis.add_geometry(m, False)
                vis.update_geometry(m)

        vis.register_key_callback(ord(' '), func)
        vis.run()

    @staticmethod
    def to_c_tooth_transform(tid, T: np.ndarray):
        rotation, translation = T[:3, :3], T[:3, 3]
        rotation = (ctypes.c_double * 9)(*rotation.flatten())
        translation = (ctypes.c_double * 3)(*translation.flatten())
        tt = ToothTransformation(tid=int(tid), rotation=rotation, translation=translation)
        return tt

    def get_random_tooth_transformation_dict(self):
        tt_dict = {}
        for tid in self.tids:
            contour = self.tooth_contour_dict[tid]
            r, t = get_random_transform()
            c = contour.mean(0)
            t = -c @ r.T + t + c

            T = np.eye(4)
            T[:3, :3] = r
            T[:3, 3] = t
            tt_dict[tid] = T
        return tt_dict

    def get_gum(self):
        return trimesh.Trimesh(self.gum_vertices, self.gum_faces)

    def __del__(self):
        self.lib.destroy_gum_deformer(self.p_gd)



if __name__ == '__main__':
    folder = '/media/wuhuikai/data/projects/IntraOralCamEst/datasets/intra_oral/mesh_data/Upper/gum'

    deformer = DeformDLL(trimesh.load(osp.join(folder, 'gum.ply')), open(osp.join(folder, 'result.txt'), 'r').read())
    deformer.deform(deformer.get_random_tooth_transformation_dict())
    deformer.show()
    deformer.get_gum().show()
