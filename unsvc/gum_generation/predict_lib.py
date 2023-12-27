import numpy as np
import trimesh
from typing import Dict, Tuple, List, Callable, Union
import json

from .lib.gum import GumSolver, GumGeneratorParams, Gum, _canonical_to_input
from .lib.tooth import Tooth


def generate_gum(teeth_dict: Dict[str, trimesh.Trimesh],
                 gum_height=None,
                 inner_curve_out_dist=None,
                 outer_curve_out_dist=None,
                 production=None) \
        -> Tuple[trimesh.Trimesh,
                 Dict[str, np.ndarray],
                 Dict[str, np.ndarray],
                 Dict[str, np.ndarray],
                 Dict[str, np.ndarray],
                 Dict[str, np.ndarray],
                 Dict[str, np.ndarray],
                 Dict[str, np.ndarray],
                 List[str],
                 Dict[str, float],
                 str]:
    """ Generate a lower gum or an upper gum by tooth meshes.
    Input:
        teeth_dict: dict
            Keys are tooth ids (FDI) and values are tooth meshes.

    Output:
        gum_mesh: trimesh.Triemsh
            The generated lower or upper gum mesh.
        gum_handle_dict: Dict
            The indices of lower/upper gum handles.
        gum_surf_point_dict: Dict
            The indices of points in each partial surface. Partial surfaces are components which the gum consists of,
            e.x. each surface in the top of the gum w.r.t each tooth.
        tooth_boundary_dict: Dict, indexed by tid
            values are tooth boundary vertices.
        gum_faceid_dict: Dict
            keys are ['side', 'top', 'bottom', 'other']
            values are indices of faces in the gum (1d array).
        tooth_lingual_side_vid_dict: Dict
            keys are tid, values are indices of vertices with shape u-by-v in the lingual side of the tooth.
        tooth_buccal_side_vid_dict: Dict
            keys are tid, values are indices of vertices with shape u-by-v in the buccal side of the tooth.
        gum_distal_side_vid_dict: Dict
        sorted_tids: sorted tooth ids
        gum_params: Dict, the parameters to generate the gum.
    """
    # initialize
    params = GumGeneratorParams()
    if gum_height is not None:
        params.gum_height = gum_height
    if inner_curve_out_dist is not None:
        params.inner_curve_out_dist = inner_curve_out_dist
    if outer_curve_out_dist is not None:
        params.outer_curve_out_dist = outer_curve_out_dist
    if production is None:
        production = True
    if not production:
        params.gum_side_sample_size = 10
        # params.tooth_mesial_distal_sample_size = 6
    # elif is_upper(list(teeth_dict.keys())[0]):
    #     params.outer_curve_out_dist = -1.
    solver = GumSolver(params)

    # transfer to Tooth
    _teeth_dict = {}
    for tid, mesh in teeth_dict.items():
        tid = int(tid)
        # check if there's water-tight tooth
        t = Tooth(tid, mesh.vertices, mesh.faces, mesh.face_normals)
        if len(t.boundary_vs) > 20:
            _teeth_dict[tid] = t

    if len(_teeth_dict) < 2:
        return trimesh.Trimesh(), {}, {}, {}, {}, {}, {}, {}, [], {}, 'not enough teeth'

    # solve
    gum = solver.generate_single_gum(_teeth_dict)
    gum_mesh = gum.out_mesh
    gum_handle_dict = {str(k): v for k, v in gum.handle_dict.items()}
    gum_handle_dict['n_points_mesial_distal'] = np.array([params.tooth_mesial_distal_sample_size])
    gum_handle_dict['n_points_buccal_lingual'] = np.array([params.tooth_buccal_lingual_sample_size])
    gum_handle_dict['n_points_gum_side_height'] = np.array([params.gum_side_sample_size])
    gum_surf_point_dict = {str(k): v for k, v in gum.surf_point_dict.items()}
    gum_faceid_dict = gum.faceid_dict
    tooth_lingual_side_vid_dict = gum.tooth_lingual_side_vid_dict
    tooth_buccal_side_vid_dict = gum.tooth_buccal_side_vid_dict
    gum_distal_side_vid_dict = gum.distal_side_vid_dict
    sorted_tids = [str(i) for i in gum.tids]

    # get tooth boundary vertices
    tooth_boundary_dict = {}
    is_u = is_upper(list(_teeth_dict.keys())[0])
    for tid, tooth in gum.teeth_dict.items():
        if is_u:
            tooth_boundary_dict[str(tid)] = _canonical_to_input(tooth.boundary_vs)
        else:
            tooth_boundary_dict[str(tid)] = np.array(tooth.boundary_vs)

    gum.ori_gum_info['production'] = production
    ori_gum_info = json.dumps(gum.ori_gum_info)

    return (gum_mesh,
            gum_handle_dict,
            gum_surf_point_dict,
            tooth_boundary_dict,
            gum_faceid_dict,
            tooth_lingual_side_vid_dict,
            tooth_buccal_side_vid_dict,
            gum_distal_side_vid_dict,
            sorted_tids,
            vars(params),
            ori_gum_info)


def is_upper(tid):
    tid = int(tid)
    return tid // 10 in [1, 2, 5, 6] or 91 <= tid <= 94
