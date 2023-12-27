import os
import base64
import time
import json
import urllib
import urllib.parse
import numpy as np
import os.path as osp
import trimesh
import trimesh.exchange.ply
import DracoPy
import requests
from typing import TypedDict, Dict, List


# 朝厚服务请求地址，随api文档发送
base_url = 'https://workflow-api.dev.chohotech.com/workflow'
# 朝厚文件服务地址，随api文档发送
file_server_url = 'https://file-server.dev.chohotech.com'
# 必须传入鉴权 Header。请保护好TOKEN!!! 如果泄露请立即联系我们重置，所有使用该TOKEN的任务都会向您的账户计费
zh_token = 'eyJ0eXBlIjoiYWNjZXNzX3Rva2VuIiwiYWxnIjoiSFMyNTYifQ.eyJpc3MiOiJhdXRoLXNlcnZpY2UiLCJzdWIiOiJ7XCJhY2NvdW50SWRcIjpcIkExLVpIU2VydmljZVwiLFwiYWNjb3VudFR5cGVcIjpcIlpIU2VydmljZVwifSIsImV4cCI6MTY0NjA0NjEzOCwiaWF0IjoxNjQ2MDQ1MjM4fQ.b_kHzp1DGssrs9a83s4f8ObyjAlpgJiJmw9FiRew6uQ'  # 调用所有的API都必须传入token用作鉴权
# 用户组，一般为 APIClient
user_group = "ZHService"
# 贵司user_id, 随api文档发送
user_id = "A1-ZHService"


def encode_mesh(mesh: trimesh.Trimesh, type='drc'):
    if type == 'drc':
        return base64.b64encode(DracoPy.encode(mesh.vertices, mesh.faces)).decode()
    else:
        assert False, "unsupported type"


def submit_task(json_data):
    url = f"{base_url}/run"
    payload = json.dumps(json_data)
    headers = {
        "Content-Type": "application/json",
        "X-ZH-TOKEN": zh_token
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    result = response.json()
    run_id = result['run_id']
    # 打印任务号
    print(run_id)

    return run_id


def wait_for_result(run_id):
    headers = {
        "Content-Type": "application/json",
        "X-ZH-TOKEN": "5ul5UiKnbamfbItLRWgND7OFxk4tMqOvdCoqfdNUJCeXGekP1omRZIjAWWkKClaZ"
    }

    start_time = time.time()
    while True:
        response = requests.request("GET", f"{base_url}/run/{run_id}", headers=headers)
        result = response.json()
        if result['completed'] or result['failed']:
            break
        time.sleep(5)
        print('RUNNING ...')

    if not result['completed']:
        if result['failed']:
            raise ValueError("API运行错误，原因： " + str(result['reason']))
    print("API运行时间： {}s".format(time.time() - start_time))

    response = requests.request("GET", f"{base_url}/data/{run_id}", headers=headers)
    return response.json()


def download(urn, type='drc') -> trimesh.Trimesh:
    resp = requests.get(f"{file_server_url}/file/download?" + urllib.parse.urlencode({"urn": urn}),
                        headers={"X-ZH-TOKEN": zh_token})
    if type == 'drc':
        mesh = DracoPy.decode(resp.content)
        return trimesh.Trimesh(vertices=mesh.points, faces=mesh.faces)
    elif type == 'ply':
        mesh = trimesh.load(trimesh.util.wrap_as_stream(resp.content), file_type=type, process=False)
    elif type == 'stl':
        mesh = trimesh.load(trimesh.util.wrap_as_stream(resp.content), file_type=type, process=False)
    else:
        raise NotImplementedError()

    return mesh


class BaseWorkflow:
    def __init__(self,
                 base_url=base_url,
                 file_server_url=file_server_url,
                 zh_token=zh_token,
                 user_group=user_group,
                 user_id=user_id):
        self.base_url = base_url
        self.file_server_url = file_server_url
        self.zh_token = zh_token
        self.user_group = user_group
        self.user_id = user_id


class GumGenerationResult(TypedDict):
    gum: trimesh.Trimesh
    handle_dict: Dict[str, List[int]]
    surf_point_dict: Dict[str, List[int]]
    tooth_boundary_dict: Dict[str, List[List[float]]]
    msg: str


class GumGeneration(BaseWorkflow):
    def submit(self, teeth_mesh_dict: Dict[int, trimesh.Trimesh],
               gum_height=None,
               inner_curve_out_dist=None,
               outer_curve_out_dist=None) -> str:
        teeth_dict = {str(tid): {'type': 'drc', 'data': encode_mesh(mesh)} for tid, mesh in teeth_mesh_dict.items()}
        run_id = submit_task({
            "spec_group": "mesh-processing",  # 调用的工作流组， 随API文档发送
            "spec_name": "gum-generation",  # 半颌牙齿牙轴预测与邻接面补全 （含FA点、分牙）
            "spec_version": "1.0-snapshot",  # 调用的工作流版本，随API文档发送
            "user_group": user_group,
            "user_id": user_id,
            "input_data": {
                "teeth_dict": teeth_dict,
                "gum_height": gum_height,
                "inner_curve_out_dist": inner_curve_out_dist,
                "outer_curve_out_dist": outer_curve_out_dist,
            },
            "output_config": {
                "gum": {"type": "ply"},
            }
        })
        return run_id

    def run_and_save_for_deformation(self, teeth_mesh_dict: Dict[int, trimesh.Trimesh], save_dir: str, gum_height=None,
                                     inner_curve_out_dist=None, outer_curve_out_dist=None):
        run_id = self.submit(teeth_mesh_dict, gum_height, inner_curve_out_dist, outer_curve_out_dist)
        result = wait_for_result(run_id)
        result = result['result']
        gum = download(result['gum']['data'], 'ply')

        # save
        os.makedirs(save_dir, exist_ok=True)
        gum.export(osp.join(save_dir, 'gum.ply'))
        print(json.dumps(result), file=open(osp.join(save_dir, 'result.txt'), 'w'))
        json.dump({'gum_vertices': np.asarray(gum.vertices).tolist(),
                   'gum_faces': np.asarray(gum.faces).tolist()}, open(osp.join(save_dir, 'gum.json'), 'w'))


if __name__ == '__main__':
    import glob

    folder = '/media/wuhuikai/data/projects/IntraOralCamEst/datasets/intra_oral/mesh_data'
    for jaw_type in ('Upper', 'Lower'):
        mesh_dict = {int(os.path.basename(path).split('.')[0]): trimesh.load(path) for path in glob.glob(os.path.join(folder, jaw_type, '*.stl'))}
        GumGeneration().run_and_save_for_deformation(mesh_dict, os.path.join(folder, jaw_type, 'gum'), gum_height=5.)
