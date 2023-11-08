import sys
sys.path.append('.')
sys.path.append('..')
from network.model_cond import pSp
from dataloader.cond_dataset import get_loader

train_loader = get_loader('/mnt/d/data/smile/out', 4, 'train')
test_loader = get_loader('/mnt/d/data/smile/out', 4, 'test')
net = pSp()