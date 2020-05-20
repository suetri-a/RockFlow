import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import shutil
import random
import warnings
import argparse

import torch
import torch.utils.data as data

from datasets import get_rock_dataset, postprocess
from model import Glow
from modules import gaussian_sample

######################################### OPTIONS ################################

parser = argparse.ArgumentParser()

parser.add_argument('--name', help='Name of model to load for generating samples')
parser.add_argument('--model', help='Name of model checkpoint to load')


# Stack generation parameters
parser.add_argument('--steps', type=int, default=None, help='Number of intermediate images to interpolate between')
parser.add_argument('--temperature', type=float, default=1, help='Temperature value for sampling distribution')



def main(args):
    # Test loading and sampling
    output_folder = os.path.join('results', args.name)

    with open(os.path.join(output_folder, 'hparams.json')) as json_file:  
        hparams = json.load(json_file)

    model_name = args.model # 'glow_checkpoint_41600.pth'
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    image_shape = (hparams['patch_size'], hparams['patch_size'], 1)
    num_classes = 1
    
    print('Loading model...')
    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                hparams['learn_top'], hparams['y_condition'])

    model_chkpt = torch.load(os.path.join(output_folder, model_name))
    model.load_state_dict(model_chkpt['model'])
    model.set_actnorm_init()
    model = model.to(device)

    stack_dir = os.path.join(output_folder, 'imgs3d')
    if not os.path.exists(stack_dir):
        os.mkdir(stack_dir)

    # Build images 
    model.eval()

    with torch.no_grad():
        temperature = 1
        if args.steps is None:
            steps = 4
        else:
            steps = args.steps

        print('Sampling images...')
        mean, logs = model.prior(None, None)
        alpha = 1-torch.reshape(torch.linspace(0, 1, steps= hparams['patch_size'] // steps),(-1,1,1,1))
        alpha = alpha.to(device)    

        num_imgs = steps + 1
        z = gaussian_sample(mean, logs, temperature)[:num_imgs,...]
        z = torch.cat([alpha*z[i,...] + (1-alpha)*z[i+1,...] for i in range(num_imgs-1)])

        images_raw = model(z=z, temperature=temperature, reverse=True)
        images = postprocess(images_raw).cpu()

    print('Creating video...')
    for i in range(hparams['patch_size']):
        plt.imsave(os.path.join(stack_dir, 'image{}.png'.format(str(i).zfill(3))), 
                np.squeeze(images[i].numpy()), cmap='gray')

    img_path = os.path.join(stack_dir,'')
    mov_path = os.path.join(stack_dir,'stack.mp4')
    if os.path.exists(mov_path):
        os.remove(mov_path)
    os.system('ffmpeg -hide_banner -loglevel panic -framerate 5 -i {} -pix_fmt \
                yuv420p {}'.format(os.path.join(img_path,'image%03d.png'), mov_path))

    print('Finished!')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)