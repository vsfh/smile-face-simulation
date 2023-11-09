import torch
import sys
sys.path.append('.')
sys.path.append('..')
from network.model_cond import Discriminator

def get_dis_opt(decoder_checkpoint_path=None, learning_rate=1e-5):
    discriminator = Discriminator(256, channel_multiplier=2)
    if decoder_checkpoint_path is not None:
        ckpt = torch.load(decoder_checkpoint_path, map_location='cpu')
        discriminator.load_state_dict(ckpt['d'], strict=False)
    discriminator_optimizer = torch.optim.Adam(list(discriminator.parameters()),
                                                    lr=learning_rate)
    return discriminator, discriminator_optimizer