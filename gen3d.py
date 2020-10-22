import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_erosion

import os
import json
import shutil
import random
import warnings
import argparse
from joblib import Parallel, delayed

import torch
import torch.utils.data as data

from datasets import get_rock_dataset, postprocess
from model import Glow
from modules import gaussian_sample
from utils import two_point_correlation

######################################### OPTIONS ################################

parser = argparse.ArgumentParser()

parser.add_argument('--name', help='Name of model to load for generating samples')
parser.add_argument('--model', help='Name of model checkpoint to load')
parser.add_argument('--save_name', default='imgs3d', type=str, help='File extension to save images in folder [save_name]_[stack #]')
parser.add_argument('--n_modalities', type=int, default=1, help='Number of modalities for the entered model')
parser.add_argument('--step_modality', type=int, default=0, help='Index (0-indexed) of modality to calculate step size')
parser.add_argument('--steps', type=int, default=None, help='Number of anchor slices to use in interpolation')
parser.add_argument('--temperature', type=float, default=1, help='Temperature value for sampling distribution')
parser.add_argument('--binary_data', action='store_true', help='Apply processing specifically for binary data')
parser.add_argument('--med_filt', type=int, default=None, help='Save images with median filter applied to x-z and y-z plane images')
parser.add_argument('--save_binary', action='store_true', help='Save images as binaries using Otsu thresholding')
parser.add_argument('--iter', type=int, default=1, help='Generate imgs3d_000 to imgs3d_iter-1 folders')
parser.add_argument('--seed', type=int, default=999, help='Random seed to use for PyTorch')


def write_video(images, prefix, hparams, stack_dir):
    print('Creating video for {} images...'.format(prefix))

    for i in range(hparams['patch_size']):
        plt.imsave(os.path.join(stack_dir, '{}_image{}.png'.format(prefix, str(i).zfill(3))), 
                np.squeeze(images[i].numpy()), cmap='gray')

    img_path = os.path.join(stack_dir,'')
    mov_path = os.path.join(stack_dir, '{}_stack.mp4'.format(prefix))
    if os.path.exists(mov_path):
        os.remove(mov_path)
    os.system('ffmpeg -hide_banner -loglevel panic -framerate 5 -i {} -pix_fmt yuv420p {}'.format(os.path.join(img_path,'{}_image%03d.png'.format(prefix)), mov_path))


def straight_line_at_origin(porosity):
    # From: https://github.com/LukasMosser/PorousMediaGan/blob/master/code/notebooks/covariance/Covariance%20Analysis.ipynb
    def func(x, a):
        return a * x + porosity
    return func


def main(args):
    # torch.manual_seed(args.seed)

    # Test loading and sampling
    output_folder = os.path.join('results', args.name)

    with open(os.path.join(output_folder, 'hparams.json')) as json_file:  
        hparams = json.load(json_file)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    image_shape = (hparams['patch_size'], hparams['patch_size'], args.n_modalities)
    num_classes = 1
    
    print('Loading model...')
    model = Glow(image_shape, 
                hparams['hidden_channels'], 
                hparams['K'], 
                hparams['L'], 
                hparams['actnorm_scale'],
                hparams['flow_permutation'], 
                hparams['flow_coupling'], 
                hparams['LU_decomposed'], 
                num_classes,
                hparams['learn_top'], 
                hparams['y_condition'])

    model_chkpt = torch.load(os.path.join(output_folder, 'checkpoints', args.model))
    model.load_state_dict(model_chkpt['model'])
    model.set_actnorm_init()
    model = model.to(device)

    # Build images
    model.eval()
    temperature = args.temperature

    if args.steps is None:  # automatically calculate step size if no step size

        fig_dir = os.path.join(output_folder, 'stepnum_results')
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        print('No step size entered')

        # Create sample of images to estimate chord length
        with torch.no_grad():
            mean, logs = model.prior(None, None)
            z = gaussian_sample(mean, logs, temperature)
            images_raw = model(z=z, temperature=temperature, reverse=True)
        images_raw[torch.isnan(images_raw)] = 0.5
        images_raw[torch.isinf(images_raw)] = 0.5
        images_raw = torch.clamp(images_raw, -0.5, 0.5)

        images_out = np.transpose(np.squeeze(images_raw[:,args.step_modality,:,:].cpu().numpy()), (1,0,2))

        # Threshold images and compute covariances
        if args.binary_data:
            thresh = 0
        else:
            thresh = threshold_otsu(images_out)
        images_bin = np.greater(images_out, thresh)
        x_cov = two_point_correlation(images_bin, 0)
        y_cov = two_point_correlation(images_bin, 1)

        # Compute chord length
        cov_avg = np.mean(np.mean(np.concatenate((x_cov, y_cov), axis=2), axis=0), axis=0)
        N = 5
        S20, _ = curve_fit(straight_line_at_origin(cov_avg[0]), range(0, N), cov_avg[0:N])
        l_pore = np.abs(cov_avg[0] / S20)
        steps = int(l_pore)
        print('Calculated step size: {}'.format(steps))

    else:
        print('Using user-entered step size {}...'.format(args.steps))
        steps = args.steps


    # Build desired number of volumes
    for iter_vol in range(args.iter):
        if args.iter == 1:
            stack_dir = os.path.join(output_folder, 'image_stacks', args.save_name)
            print('Sampling images, saving to {}...'.format(args.save_name))
        else:
            stack_dir = os.path.join(output_folder, 'image_stacks', args.save_name + '_' + str(iter_vol).zfill(3))
            print('Sampling images, saving to {}_'.format(args.save_name) + str(iter_vol).zfill(3) + '...')
        if not os.path.exists(stack_dir):
            os.makedirs(stack_dir)

        with torch.no_grad():
            mean, logs = model.prior(None, None)
            alpha = 1-torch.reshape(torch.linspace(0,1,steps=steps),(-1,1,1,1))
            alpha = alpha.to(device)

            num_imgs = int(np.ceil(hparams['patch_size'] / steps) + 1)
            z = gaussian_sample(mean, logs, temperature)[:num_imgs,...]
            z = torch.cat([alpha*z[i,...] + (1-alpha)*z[i+1,...] for i in range(num_imgs-1)])
            z = z[:hparams['patch_size'], ...]

            images_raw = model(z=z, temperature=temperature, reverse=True)
        
        images_raw[torch.isnan(images_raw)] = 0.5
        images_raw[torch.isinf(images_raw)] = 0.5
        images_raw = torch.clamp(images_raw, -0.5, 0.5)

        # apply median filter to output
        if args.med_filt is not None or args.binary_data:
            for m in range(args.n_modalities):
                if args.binary_data:
                    SE = ball(1)
                else:
                    SE = ball(args.med_filt)
                images_np = np.squeeze(images_raw[:,m,:,:].cpu().numpy())
                images_filt = median_filter(images_np, footprint=SE)
                
                # Erode binary images
                if args.binary_data:
                    images_filt = np.greater(images_filt, 0)
                    SE = ball(1)
                    images_filt = 1.0*binary_erosion(images_filt, selem=SE) - 0.5

                images_raw[:,m,:,:] = torch.tensor(images_filt, device=device)

        images1 = postprocess(images_raw).cpu()
        images2 = postprocess(torch.transpose(images_raw, 0, 2)).cpu()
        images3 = postprocess(torch.transpose(images_raw, 0, 3)).cpu()

        # apply Otsu thresholding to output
        if args.save_binary and not args.binary_data:
            thresh = threshold_otsu(images1.numpy())
            images1[images1<thresh] = 0
            images1[images1>thresh] = 255
            images2[images2<thresh] = 0
            images2[images2>thresh] = 255
            images3[images3<thresh] = 0
            images3[images3>thresh] = 255


        # # erode binary images by 1 px to correct for training image transformation
        # if args.binary_data:
        #     images1 = np.greater(images1.numpy(), 127)
        #     images2 = np.greater(images2.numpy(), 127)
        #     images3 = np.greater(images3.numpy(), 127)

        #     images1 = 255*torch.tensor(1.0*np.expand_dims(binary_erosion(np.squeeze(images1), selem=np.ones((1,2,2))), 1))
        #     images2 = 255*torch.tensor(1.0*np.expand_dims(binary_erosion(np.squeeze(images2), selem=np.ones((2,1,2))), 1))
        #     images3 = 255*torch.tensor(1.0*np.expand_dims(binary_erosion(np.squeeze(images3), selem=np.ones((2,2,1))), 1))

        # save video for each modality
        for m in range(args.n_modalities):
            if args.n_modalities > 1:
                save_dir = os.path.join(stack_dir, 'modality{}'.format(m))
            else:
                save_dir = stack_dir

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            write_video(images1[:,m,:,:], 'xy', hparams, save_dir)
            write_video(images2[:,m,:,:], 'xz', hparams, save_dir)
            write_video(images3[:,m,:,:], 'yz', hparams, save_dir)
    
    print('Finished!')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)