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
checkpoint_path = '/data/shenfeihong/smile/weight/3000.pt'
save_path = '/data/shenfeihong/smile/weight/'
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

def libtorch():
    net = get_network(checkpoint_path)
    input1 = torch.randn(1,7,256,256).cuda()
    module = torch.jit.trace(net, (input1))
    module.save(os.path.join(save_path, 'torchscript.pt'))

     
if __name__=='__main__':
    libtorch()   