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

data_path = '/data/shenfeihong/smile/out/'
decoder_checkpoint_path = "/data/shenfeihong/smile/ori_style/checkpoint/ori_style_150000.pt"
save_path = '.' # '/data/shenfeihong/smile/weight/'
learning_rate= 1e-5
start_from_avg = True
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

def test_single(path, case, checkpoint_path):
    input = get_input(os.path.join(path, case), '64')
    cond_img = input['cond']
    real_img = input['images']
    net = get_network(checkpoint_path)
    with torch.no_grad():
        fake_img = net.forward(cond_img, return_latents=False)
    img_dict = {}
    img_dict['cond'] = cond_img[0][:3,:,:]
    img_dict['input'] = cond_img[0][-3:,:,:]
    img_dict['target'] = real_img[0]
    img_dict['output'] = fake_img[0]
    fig = plotting_fig(img_dict)
    fig.savefig(os.path.join(path, case, f'test.png'))
    plt.close(fig)

def checkpointme(net, accelerator, step_idx):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(net)  
    accelerator.save(unwrapped_model.state_dict(), os.path.join(save_path,f'{step_idx}.pt'))
     
def test_pt():
    date = '11.16'
    path = f'/data/shenfeihong/smile/orthovis/{date}/checkpoint/'
    save_path = f'/data/shenfeihong/smile/orthovis/{date}/result'
    os.makedirs(save_path, exist_ok=True)
    for file in os.listdir(path):
        if not file.endswith('.pt'):
            continue
        print(file)
        input = get_example('example')
        cond_img = input['cond']
        real_img = input['images']
        net = get_network(os.path.join(path, file))
        with torch.no_grad():
            fake_img, _ = net.forward(cond_img, return_latents=False, concat_img=False)
        img_dict = {}
        img_dict['cond'] = cond_img[0][:3,:,:]
        img_dict['input'] = cond_img[0][-3:,:,:]
        img_dict['target'] = real_img[0]
        img_dict['output'] = fake_img[0]
        fig = plotting_fig(img_dict)
        fig.savefig(os.path.join(save_path, file.replace('.pt','.png')))
        plt.close(fig)
        
if __name__=='__main__':
    path = '/mnt/d/data/smile/out'
    test_pt()
    # for case in tqdm(natsort.natsorted(os.listdir(path))[:15]):
    #     test_single(path,case)  