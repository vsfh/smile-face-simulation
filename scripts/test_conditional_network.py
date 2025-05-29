import os
import sys
sys.path.append('.')
sys.path.append('..')
from network.model_cond import pSp
from dataloader.cond_dataset import get_input, get_example
import torch
import torch.nn.functional as F

from scripts.utils import *
from tqdm import tqdm
import natsort
import cv2


data_path = '/data/shenfeihong/smile/out/'
decoder_checkpoint_path = "/data/shenfeihong/smile/ori_style/checkpoint/ori_style_150000.pt"
decoder_checkpoint_path = None
# decoder_checkpoint_path = "/mnt/e/share/weight/smile/ori_style_150000.pt"
save_path = './local_data' # '/data/shenfeihong/smile/weight/'
learning_rate= 1e-5
start_from_avg = False
max_step = 800000
d_reg_every = 16
plot_every = 100
dis_update = 1000
vali_every = 1000

def get_network(checkpoint_path):
    net = pSp(decoder_checkpoint_path=decoder_checkpoint_path,start_from_avg=start_from_avg).cuda()
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    return net

def test_yang(path, net, checkpoint_path=None, save_img_path=None):
    input = get_input(path, False)
    cond_img = input['cond']
    real_img = input['images']
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path))
    with torch.no_grad():
        fake_img,_ = net.forward(cond_img, return_latents=False, concat_img=False)

    img_dict = {}
    sum_img = torch.zeros_like(cond_img[0][:3,:,:])
    sum_img[0] = 1-cond_img[0][2,:,:]
    sum_img[1] = cond_img[0][5,:,:]
    sum_img[2] = cond_img[0][8,:,:]
    if not save_img_path is None:
        save_img = tensor2im(fake_img[0])
        save_img.save(save_img_path)
        im = cv2.imread(save_path)
        cv2.imshow('sim', im)
        cv2.waitKey(0)
    else:
        img_dict['cond'] = sum_img
        img_dict['input'] = cond_img[0][6:9,:,:]
        img_dict['depth'] = cond_img[0][3:6,:,:]
        img_dict['target'] = real_img[0]
        img_dict['output'] = fake_img[0]
        
        fig = plotting_fig(img_dict)
        fig.savefig(f'/mnt/gregory/smile/smile-face-simulation/visualize/{case}.png')
        plt.close(fig)

def test_multi_img(path, checkpoint_path):
    net = get_network(checkpoint_path)
    
    for case in tqdm(natsort.natsorted(os.listdir(path))[:200]):
        input = get_input(os.path.join(path, case), False)
        cond_img = input['cond']
        real_img = input['images']
        with torch.no_grad():
            fake_img, _ = net.forward(cond_img, real_img.clone().detach(), return_latents=False, concat_img=False)
        save_img = tensor2im(fake_img[0])
        save_img.save(f'/mnt/e/paper/smile/orthogan/new/{case}.png')
        # continue
        img_dict = {}
        img_dict['cond'] = cond_img[0][:3,:,:]
        img_dict['input'] = cond_img[0][-3:,:,:]
        img_dict['target'] = real_img[0]
        img_dict['output'] = fake_img[0]
        fig = plotting_fig(img_dict)
        fig.savefig(f'/mnt/e/paper/smile/orthogan/new/all_{case}.png')
        plt.close(fig)

def test_pt():
    date = '11.23'
    path = f'/data/shenfeihong/smile/orthovis/{date}/checkpoint/'
    save_path = f'./local_data/{date}/result'
    os.makedirs(save_path, exist_ok=True)
    for file in os.listdir(path):
        if not file.endswith('0000.pt'):
            continue
        print(file)
        input = get_example('example')
        cond_img = input['cond']
        real_img = input['images']
        net = get_network(os.path.join(path, file))
        with torch.no_grad():
            fake_img, _ = net.forward(cond_img, return_latents=False, concat_img=True)
        img_dict = {}
        img_dict['cond'] = cond_img[0][:3,:,:]
        img_dict['input'] = cond_img[0][-3:,:,:]
        img_dict['target'] = real_img[0]
        img_dict['output'] = fake_img[0]
        fig = plotting_fig(img_dict)
        fig.savefig(os.path.join(save_path, file.replace('.pt','.png')))
        plt.close(fig)
        
def vis():
    for case_path in glob.glob('/mnt/gregory/smile/data/Teeth/C01*'):
        save_path = case_path+'/align/sim.png'
        if not os.path.exists(save_path):
            continue
        im = cv2.imread(save_path)
        cv2.imshow('sim', im)
        cv2.waitKey(0)
        

if __name__=='__main__':
    import glob
    path = '/mnt/e/data/smile/YangNew'
    path = '/mnt/hdd/data/smile/out1'
    # vis()
    net = get_network('/mnt/gregory/smile/weight/cond/5.28/checkpoint/30000.pt')
    for case_path in glob.glob('/mnt/gregory/smile/data/Teeth/C01*'):
        save_path = case_path+'/align/sim.png'
        if not os.path.exists(save_path.replace('sim.png', 'depth.png')):
            continue
        test_yang(case_path, net, save_img_path=save_path)
        # break
