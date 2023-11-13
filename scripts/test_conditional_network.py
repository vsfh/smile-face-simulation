import os
import sys
sys.path.append('.')
sys.path.append('..')
from network.model_cond import pSp
from dataloader.cond_dataset import get_input
import torch
import torch.nn.functional as F

from scripts.utils import *

data_path = '/data/shenfeihong/smile/out/'
checkpoint_path = '/data/shenfeihong/smile/weight/200000.pt'
save_path = '.' # '/data/shenfeihong/smile/weight/'
learning_rate= 1e-5
start_from_latent_avg = False
max_step = 800000
d_reg_every = 16
plot_every = 100
dis_update = 1000
vali_every = 1000

def get_network(checkpoint_path):
    net = pSp().cuda()
    net.load_state_dict(torch.load(checkpoint_path))
    
    return net

def test_single():
    input = get_input('/data/shenfeihong/smile/out/C01002758248/', '52')
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
    fig.savefig(os.path.join(save_path,f'test.png'))
    plt.close(fig)

def checkpointme(net, accelerator, step_idx):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(net)  
    accelerator.save(unwrapped_model.state_dict(), os.path.join(save_path,f'{step_idx}.pt'))
     
if __name__=='__main__':
    test_single()   