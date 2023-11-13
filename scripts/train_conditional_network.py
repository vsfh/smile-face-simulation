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

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from scripts.utils import *

data_path = '/data/shenfeihong/smile/out/'
decoder_checkpoint_path = '/data/shenfeihong/smile/ori_style/checkpoint/ori_style_150000.pt'
save_path = '/data/shenfeihong/smile/weight/'
learning_rate= 1e-5
start_from_latent_avg = False
max_step = 800000
d_reg_every = 16
plot_every = 100
dis_update = 1000
vali_every = 1000

def get_pipeline():
    train_loader = get_loader(data_path, 4, 'train')
    test_loader = get_loader(data_path, 4, 'test')
    net = pSp(decoder_checkpoint_path)
    optimizer = torch.optim.Adam(net.encoder.parameters(), lr=learning_rate)
    discriminator, discriminator_opt = get_dis_opt(decoder_checkpoint_path, learning_rate)
    lpips_loss = LPIPS(net_type='alex')
    # w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=start_from_latent_avg)
    
    return train_loader, test_loader, net, optimizer, discriminator, discriminator_opt, lpips_loss

def train():
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # accelerator = Accelerator()
    train_loader, test_loader, net, optimizer, discriminator, discriminator_opt, lpips_loss = get_pipeline()
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
    
    while step_idx < max_step:
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            cond_img = batch['cond']
            real_img = batch['images']
            fake_img = net.forward(cond_img, return_latents=False)
        
            # lpips loss#
            lp_loss = lpips_loss(real_img, fake_img)
            # l2 loss #
            l2_loss = 0.1 * F.mse_loss(real_img, fake_img)
            # adv loss #
            adv_loss = F.softplus(-discriminator(fake_img)).mean()
            
            loss = lp_loss + l2_loss + adv_loss
            accelerator.backward(loss)
            optimizer.step()
            if step_idx % 50 == 0:
                print(f'lp_loss:{lp_loss.item()}, l2_loss:{l2_loss.item()}, adv_loss:{adv_loss.item()}')

            
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
                img_dict = {}
                img_dict['cond'] = cond_img[0][:3,:,:]
                img_dict['input'] = cond_img[0][-3:,:,:]
                img_dict['target'] = real_img[0]
                img_dict['output'] = fake_img[0]
                fig = plotting_fig(img_dict)
                fig.savefig(os.path.join(save_path,f'{step_idx}.png'))
                plt.close(fig)
                
            is_validate = step_idx % vali_every == 0
            if is_validate:
                validate(net, test_loader, step_idx)
                checkpointme(net, accelerator, step_idx)
                
            step_idx += 1
             
def validate(net, test_loader, step_idx):
    net.eval()
    for i, batch in enumerate(test_loader):
        cond_img = batch['cond']
        real_img = batch['images']
        with torch.no_grad():
            fake_img = net.forward(cond_img, return_latents=False)
        img_dict = {}
        img_dict['cond'] = cond_img[0][:3,:,:]
        img_dict['input'] = cond_img[0][-3:,:,:]
        img_dict['target'] = real_img[0]
        img_dict['output'] = fake_img[0]
        fig = plotting_fig(img_dict)
        fig.savefig(os.path.join(save_path,f'val_{step_idx}_{i}.png'))
        plt.close(fig)
    net.train()
    return None

def checkpointme(net, accelerator, step_idx):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(net)  
    accelerator.save(unwrapped_model.state_dict(), os.path.join(save_path,f'{step_idx}.pt'))
     
if __name__=='__main__':
    train()   