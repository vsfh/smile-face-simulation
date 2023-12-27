import os, click, time, glob
import trimesh
import trimesh.exchange.ply
import DracoPy
from predict_lib import generate_gum
from typing import Dict

def read_all_teeth(data_path: str, prefix='crown_tooth') -> Dict[int, trimesh.Trimesh]:
    fps = glob.glob(os.path.join(data_path, prefix + '??.stl'))
    teeth_dict = {}
    for fp in fps:
        tid = int(os.path.split(fp)[-1][len(prefix):len(prefix) + 2])
        teeth_dict[tid] = trimesh.load(fp)
        data = trimesh.exchange.ply.load_draco(trimesh.util.wrap_as_stream(trimesh.exchange.ply.export_draco(teeth_dict[tid])))
        teeth_dict[tid] = trimesh.Trimesh(data['vertices'], data['faces'])
    return teeth_dict

def check_mesh(mesh:trimesh.Trimesh):
    print('-----------Check mesh-----------')
    print('watertight:', mesh.is_watertight)
    print()

def get_upper_teeth_dict(teeth_dict: Dict[int, trimesh.Trimesh]) -> Dict[str, trimesh.Trimesh]:
    return {str(tid): tooth for tid, tooth in teeth_dict.items() if tid // 10 == 1 or tid // 10 == 2}

def get_lower_teeth_dict(teeth_dict: Dict[int, trimesh.Trimesh]) -> Dict[str, trimesh.Trimesh]:
    return {str(tid): tooth for tid, tooth in teeth_dict.items() if tid // 10 == 3 or tid // 10 == 4}


@click.command()
@click.option('--data_path', default="/mnt/share/shenkaidi/Choho/unisvc/datasets/gum_gen/05")
@click.option('--production', default=False)
def run(data_path, production):
    teeth_dict = read_all_teeth(data_path, prefix='')

    t1 = time.time()
    u_teeth_dict = get_upper_teeth_dict(teeth_dict)
    l_teeth_dict = get_lower_teeth_dict(teeth_dict)
    u_gum = generate_gum(u_teeth_dict, production=production)[0]
    l_gum = generate_gum(l_teeth_dict, production=production)[0]
    print(time.time()-t1)

    check_mesh(u_gum)
    check_mesh(l_gum)

    l_gum.export('l.stl')
    u_gum.export('u.stl')

if __name__ == '__main__':
    run()
