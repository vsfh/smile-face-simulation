
from torch import nn, autograd  ##### modified
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def discriminator_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()

    return real_loss + fake_loss

def discriminator_r1_loss(real_pred, real_w):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	# var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def plotting_fig(img_dict):
	fig = plt.figure(figsize=(8, 5))
	gs = fig.add_gridspec(1, 5)

	fig.add_subplot(gs[0, 0])
	plt.imshow(tensor2im(img_dict['cond']), cmap="gray")
	plt.title('cond')
	fig.add_subplot(gs[0, 1])
	plt.imshow(tensor2im(img_dict['input']), cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[0, 2])
	plt.imshow(tensor2im(img_dict['depth']))
	plt.title('depth')
	fig.add_subplot(gs[0, 3])
	plt.imshow(tensor2im(img_dict['target']))
	plt.title('Target')
	fig.add_subplot(gs[0, 4])
	plt.imshow(tensor2im(img_dict['output']))
	plt.title('Output')
	plt.tight_layout()
 
	return fig