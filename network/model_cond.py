import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('.')
sys.path.append('..')
from network.encoders import GradualStyleEncoder
from network.layers import *

def replace_batchnorm(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm(module)
        
        if isinstance(module, torch.nn.BatchNorm2d):
            model._modules[name] = torch.nn.GroupNorm(32, module.num_features)

    return model 

class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, dilation=8 ##### modified
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    dilation=max(1, 32 // (2**(i-1))) ##### modified
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, dilation=max(1, 32 // (2**i))  ##### modified
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, dilation=max(1, 32 // (2**(i-1))))) ##### modified

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)
    
    # styles is the latent code w+
    # first_layer_feature is the first-layer input feature f
    # first_layer_feature_ind indicate which layer of G accepts f (should always=0, the first layer)
    # skip_layer_feature is the encoder features sent by skip connection
    # fusion_block is the network to fuse the encoder feature and decoder feature
    # zero_noise is to force the noise to be zero (to avoid flickers for videos)
    # editing_w is the editing vector v used in video face editing
    def forward(
            self,
            styles,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=False,
            first_layer_feature = None, ##### modified
            first_layer_feature_ind = 0,  ##### modified
            skip_layer_feature = None,   ##### modified
            fusion_block = None,   ##### modified
            zero_noise = False,   ##### modified
            editing_w = None,   ##### modified
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if zero_noise:
            noise = [
                getattr(self.noises, f'noise_{i}') * 0.0 for i in range(self.num_layers)
            ]
        elif noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
        # w+ + v for video face editing
        if editing_w is not None:  ##### modified
            latent = latent + editing_w
        
        # the original StyleGAN
        if first_layer_feature is None: ##### modified
            out = self.input(latent)
            out = F.adaptive_avg_pool2d(out, 32) ##### modified
            out = self.conv1(out, latent[:, 0], noise=noise[0])
            skip = self.to_rgb1(out, latent[:, 1])       
        # the default StyleGANEX, replacing the first layer of G
        elif first_layer_feature_ind == 0: ##### modified
            out = first_layer_feature[0] ##### modified
            out = self.conv1(out, latent[:, 0], noise=noise[0])
            skip = self.to_rgb1(out, latent[:, 1])    
        # maybe we can also use the second layer of G to accept f?
        else: ##### modified
            out = first_layer_feature[0] ##### modified
            skip = first_layer_feature[1]  ##### modified    

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            # these layers accepts skipped encoder layer, use fusion block to fuse the encoder feature and decoder feature
            if skip_layer_feature and fusion_block and i//2 < len(skip_layer_feature) and i//2 < len(fusion_block):
                if editing_w is None:
                    out, skip = fusion_block[i//2](skip_layer_feature[i//2], out, skip) 
                else:
                    out, skip = fusion_block[i//2](skip_layer_feature[i//2], out, skip, editing_w[:,i]) 
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip
        if return_latents:
            return image, latent
        return image, None

class pSp(nn.Module):
    def __init__(            
            self,decoder_checkpoint_path=None, start_from_avg=False):
        super(pSp, self).__init__()
        self.encoder = GradualStyleEncoder(50, 'ir_se', use_skip=True, use_skip_torgb=True, input_nc=4)
        self.decoder = Generator(256, 512, 8)
        if not decoder_checkpoint_path is None:
            self.decoder.load_state_dict(torch.load(decoder_checkpoint_path)['g_ema'])
        self = replace_batchnorm(self)
        self.start_from_avg = start_from_avg
        if start_from_avg:
            self.mean_latent = self.decoder.mean_latent(int(1e5))[0].detach().cuda()
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def forward(
            self,
            cond_img,
            styles=None,
            return_latents=False,
            first_layer_feature_ind = 0,  ##### modified
            fusion_block = None,   ##### modified
            input_is_latent = False,
            concat_img = True

    ):
        if styles is None:
            styles = self.encoder(cond_img[:,-3:,:,:]*cond_img[:,2:3,:,:],return_feat=False, return_full=True) ##### modified
            if self.start_from_avg:
                styles = styles + self.mean_latent.repeat(cond_img.shape[0],1,1)
            input_is_latent = True
        feats = self.encoder(cond_img[:,:4,:,:], return_feat=True, return_full=True) ##### modified
        first_layer_feats, skip_layer_feats = None, None ##### modified            

        first_layer_feats = feats[0:2] # use f
        skip_layer_feats = feats[2:] # use skipped encoder feature
        if fusion_block is None:
            fusion_block = self.encoder.fusion # use fusion layer to fuse encoder feature and decoder feature.
        images = self.decoder([styles],
                                input_is_latent = input_is_latent,
                                return_latents=return_latents,
                                first_layer_feature=first_layer_feats,
                                first_layer_feature_ind=first_layer_feature_ind,
                                skip_layer_feature=skip_layer_feats,
                                fusion_block=fusion_block) ##### modified
        if concat_img:
            images = images*(cond_img[:,3:4,:,:])+cond_img[:,:3,:,:]*(1-cond_img[:,3:4,:,:])
        if self.start_from_avg:
            return images, styles - self.mean_latent.repeat(cond_img.shape[0],1,1)
        return images


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], img_channel=3):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(img_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        
        self.size = size ##### modified

    def forward(self, input):
        # for input that not satisfies the target size, we crop it to extract a small image of the target size.
        _, _, h, w = input.shape ##### modified
        i, j = torch.randint(0, h+1-self.size, size=(1,)).item(), torch.randint(0, w+1-self.size, size=(1,)).item() ##### modified
        out = self.convs(input[:,:,i:i+self.size,j:j+self.size]) ##### modified

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

if __name__=='__main__':
    g = Generator(256, 512 ,8)