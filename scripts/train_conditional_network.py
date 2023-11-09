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

data_path = '/mnt/d/data/smile/out'
decoder_checkpoint_path = '/mnt/d/data/smile/weight/cond_decoder.pt'
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
    train_loader, test_loader, net, optimizer, discriminator, discriminator_opt, lpips_loss = accelerator.prepare(get_pipeline())
    net.train()
    step_idx = 0
    
    while step_idx < max_step:
        for i, batch in enumerate(train_loader):
            cond_img = batch['cond']
            real_img = batch['images']
            fake_img = net.forward(cond_img, return_latents=False)
            
            # lpips loss#
            loss = lpips_loss(real_img, fake_img)
            # l2 loss #
            loss += F.mse_loss(real_img, fake_img)
            # adv loss #
            loss += F.softplus(-discriminator(fake_img)).mean()
            
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
                img_dict = {}
                img_dict['cond'] = cond_img[0][:3,:,:]
                img_dict['input'] = cond_img[0][-3:,:,:]
                img_dict['target'] = real_img[0]
                img_dict['output'] = fake_img[0]
                fig = plotting_fig(img_dict)
                fig.savefig(f'/mnt/e/wsl/code/smile-face-simulation/run/{step_idx}.png')
                plt.close(fig)
                
            is_validate = step_idx % vali_every == 0
            if is_validate:
                validate(net, test_loader, step_idx)
                checkpointme(net, accelerator, step_idx)
            step_idx += 1
            
        #     break
        # break
             
def validate(net, test_loader, step_idx):
    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            cond_img = batch['cond']
            real_img = batch['images']
            fake_img = net.forward(cond_img, return_latents=False)
            img_dict = {}
            img_dict['cond'] = cond_img[0][:3,:,:]
            img_dict['input'] = cond_img[0][-3:,:,:]
            img_dict['target'] = real_img[0]
            img_dict['output'] = fake_img[0]
            fig = plotting_fig(img_dict)
            fig.savefig(f'/mnt/e/wsl/code/smile-face-simulation/run/val_{step_idx}_{i}.png')
            plt.close(fig)

def checkpointme(net, accelerator, step_idx):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(net)  
    accelerator.save(unwrapped_model.state_dict(), f'/mnt/e/wsl/code/smile-face-simulation/run/{step_idx}.pt')  
     
if __name__=='__main__':
    train()   