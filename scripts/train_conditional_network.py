import os
import sys
sys.path.append('.')
sys.path.append('..')
from network.model_cond import pSp
from dataloader.cond_dataset import get_loader
import torch
import torch.nn.functional as F

from criteria.lpips.lpips import LPIPS
from criteria import w_norm
from criteria.adversial_loss import get_dis_opt
import socket

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs 

from scripts.utils import *

data_path = '/data/shenfeihong/smile/out/'
data_path = '/data/shenfeihong/smile/YangNew/'
decoder_checkpoint_path = '/data/shenfeihong/smile/ori_style/checkpoint/ori_style_150000.pt'
decoder_checkpoint_path = None
if socket.gethostname() == 'vsfh-System-Product-Name':
    save_path = '/mnt/gregory/smile/weight/cond/4.18'
else:
    save_path = '/nas/gregory/smile/weight/cond/5.28'
    
learning_rate= 1e-5
start_from_avg = False
max_step = 200000
d_reg_every = 16
plot_every = 100
dis_update = 1000
vali_every = 1000
concat_img=False


def get_pipeline(args):
    train_loader = get_loader(data_path, 4, 'train', args=args)
    test_loader = get_loader(data_path, 1, 'test', args=args)
    net = pSp(decoder_checkpoint_path, start_from_avg=start_from_avg)
    optimizer = torch.optim.Adam(net.encoder.parameters(), lr=learning_rate)
    discriminator, discriminator_opt = get_dis_opt(decoder_checkpoint_path, learning_rate)
    lpips_loss = LPIPS(net_type='alex')
    # w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=start_from_latent_avg)
    
    return train_loader, test_loader, net, optimizer, discriminator, discriminator_opt, lpips_loss

def train(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # accelerator = Accelerator()
    train_loader, test_loader, net, optimizer, discriminator, discriminator_opt, lpips_loss = get_pipeline(args)
    train_loader, test_loader, net, optimizer, discriminator, discriminator_opt, lpips_loss = accelerator.prepare(train_loader, \
                                                                                                                test_loader, \
                                                                                                                net, \
                                                                                                                optimizer, \
                                                                                                                discriminator, \
                                                                                                                discriminator_opt, \
                                                                                                                lpips_loss)
    print('prepared success')
    net.train()
    requires_grad(discriminator, False)
    step_idx = 0
    plotting_dir = os.path.join(save_path, 'image')
    model_save_dir = os.path.join(save_path, 'checkpoint')
    os.makedirs(plotting_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    while step_idx < max_step:
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            cond_img = batch['cond']
            real_img = batch['images']
            mmask = batch['mmask']
            if start_from_avg:
                fake_img, latent = net.forward(cond_img, real_img.clone().detach(), return_latents=False, concat_img=concat_img)
            else:
                fake_img, _ = net.forward(cond_img, real_img.clone().detach(), return_latents=False, concat_img=concat_img)
        
            # former (1, 0.1, 1) (1, 1, 0.01) (1, 0, 0.01)
            # lpips loss#

            lp_loss = lpips_loss(real_img, fake_img)
            # l2 loss #

            l2_loss = F.mse_loss(real_img, fake_img)
            # adv loss #
            # adv_loss = F.softplus(-discriminator(fake_img)).mean()
            
            loss = lp_loss + l2_loss
            if start_from_avg:
                w_norm_loss = 0.001 * torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
                loss += w_norm_loss
                if step_idx % 50 == 0:
                    print(f'w-norm:{w_norm_loss.item()}')
            if step_idx % 50 == 0:
                print(f'lp_loss:{lp_loss.item()}, l2_loss:{l2_loss.item()}, ')
            accelerator.backward(loss)
            optimizer.step()
            
            # discriminator loss #
            update_discriminator = step_idx > dis_update
            if update_discriminator:
                requires_grad(discriminator, True)
                discriminator_opt.zero_grad()
                real_pred = discriminator(real_img)
                fake_pred = discriminator(fake_img.detach())
                dis_loss = discriminator_loss(real_pred, fake_pred)
                accelerator.backward(dis_loss)
                discriminator_opt.step()
                requires_grad(discriminator, False)
            
            d_regularize = step_idx % d_reg_every == 0
            if d_regularize:
                discriminator_opt.zero_grad()
                real_img = real_img.detach()
                real_img.requires_grad = True
                real_pred = discriminator(real_img)
                r1_loss = discriminator_r1_loss(real_pred, real_img)
                discriminator.zero_grad()
                r1_final_loss = r1_loss
                accelerator.backward(r1_final_loss)
                discriminator_opt.step()
                
            plotting = step_idx % plot_every == 0
            if plotting:
                cond_img = accelerator.gather(cond_img)
                real_img = accelerator.gather(real_img)
                fake_img = accelerator.gather(fake_img)
                img_dict = {}
                img_dict['cond'] = cond_img[0][:3,:,:]
                img_dict['input'] = cond_img[0][6:9,:,:]
                img_dict['depth'] = cond_img[0][3:6,:,:]
                # img_dict['re'] = cond_img[0][9:12,:,:]
                img_dict['target'] = real_img[0]
                img_dict['output'] = fake_img[0]
                fig = plotting_fig(img_dict)
                fig.savefig(os.path.join(plotting_dir, f'{step_idx}.png'))
                plt.close(fig)
                
            is_validate = step_idx % vali_every == 0
            if is_validate:
                validate(net, test_loader, plotting_dir, step_idx)
                checkpointme(net, accelerator, os.path.join(model_save_dir,f'{step_idx}.pt'))
                
            step_idx += 1
             
def validate(net, test_loader, save_path, step_idx):
    net.eval()
    for i, batch in enumerate(test_loader):
        cond_img = batch['cond']
        real_img = batch['images']
        with torch.no_grad():
            if start_from_avg:
                fake_img, _ = net.forward(cond_img, return_latents=False, concat_img=concat_img)
            else:
                fake_img, _ = net.forward(cond_img, return_latents=False, concat_img=concat_img)
        img_dict = {}
        img_dict['cond'] = cond_img[0][:3,:,:]
        img_dict['input'] = cond_img[0][6:9,:,:]
        img_dict['depth'] = cond_img[0][3:6,:,:]
        # img_dict['re'] = cond_img[0][9:12,:,:]
        img_dict['target'] = real_img[0]
        img_dict['output'] = fake_img[0]
        fig = plotting_fig(img_dict)
        fig.savefig(os.path.join(save_path, f'val_{step_idx}_{i}.png'))
        plt.close(fig)
    net.train()
    return None

def checkpointme(net, accelerator, save_path):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(net)  
    accelerator.save(unwrapped_model.state_dict(), save_path)
     
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/nas/gregory/smile/data/detect/changed_image/')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    train(args)   