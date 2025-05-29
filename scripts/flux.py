import torch
from diffusers import FluxFillPipeline, DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
import os
import glob
from natsort import natsorted
import numpy as np
import cv2
# prompt = "aligned teeth with similar texture, light condition.  "
prompt = 'aligned teeth, ensure realistic, aesthetic look, no exaggerated features.'
def test():
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.to(torch.float16)
    for img_path in natsorted(glob.glob('/nas/gregory/smile/data/Teeth/C01*/Img.jpg')):
        img_name = img_path.split('/')[-2]
        mask_path = img_path.replace('Img.jpg', 'MouthMask.png')
        # if not os.path.exists(f'/nas/gregory/smile/data/detect/images/train/{img_name}'):
        #     continue
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        out_im = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=512,
            width=512,
            guidance_scale=10,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        out_im.save(f"/nas/gregory/smile/data/llm/flux2/{img_name}.png")

    
if __name__ == "__main__":
    test()