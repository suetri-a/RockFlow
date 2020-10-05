import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import perimeter 
from skimage.filters import median

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
parser.add_argument('--steps', type=int, default=None, help='Number of anchor slices to use in interpolation')
parser.add_argument('--n_trials', type=int, default=20, help='Number of trials to estimate likelihood for each step number')
parser.add_argument('--temperature', type=float, default=1, help='Temperature value for sampling distribution')
<<<<<<< HEAD
parser.add_argument('--save_med_filt', action='store_true', help='Save images with median filter applied to x-z and y-z plane images')
=======
parser.add_argument('--iter', type=int, default=0, help='Generate imgs3d_000 to imgs3d_iter folders')
>>>>>>> 3d8cf0afe841587e7aa5df1304094d9b6c5c4c6e


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

    for iter_vol in range(args.iter+1):
        stack_dir = os.path.join(output_folder, 'imgs3d_' + str(iter_vol).zfill(3))
        if not os.path.exists(stack_dir):
            os.mkdir(stack_dir)

        # Build images
        model.eval()

        with torch.no_grad():
            temperature = 1

            if args.steps is None:  # automatically calculate step size if no step size
                fig_dir = os.path.join(output_folder, 'stepnum_results')
                if not os.path.exists(fig_dir):
                    os.mkdir(fig_dir)

                print('No step size entered')

                if os.path.exists(os.path.join(fig_dir, 'result.json')):
                    with open(os.path.join(fig_dir, 'result.json'), 'r') as fp:
                        results = json.load(fp)
                    steps = results['steps']

                else:
                    print('Pre-computed optimal step size not found. Computing optimal step size...')
                    mean, logs = model.prior(None, None)

                    n_trials = args.n_trials
                    step_data_xy = {}
                    step_data_xz = {}
                    step_data_yz = {}
                    steps_list = [i for i in range(3, 33, 2)]
                    for steps in steps_list:

                        step_data_xy[steps] = []
                        step_data_xz[steps] = []
                        step_data_yz[steps] = []

                        for i in range(n_trials):
                            # Create volume of images
                            model.eval()
                            with torch.no_grad():
                                alpha = 1-torch.reshape(torch.linspace(0,1,steps=steps),(-1,1,1,1))
                                alpha = alpha.to(device)

                                num_imgs = int(np.ceil(hparams['patch_size'] / steps) + 1)
                                z = gaussian_sample(mean, logs, temperature)[:num_imgs,...]
                                z = torch.cat([alpha*z[i,...] + (1-alpha)*z[i+1,...] for i in range(num_imgs-1)])
                                z = z[:hparams['patch_size'], ...]
                                images_raw = model(z=z, temperature=temperature, reverse=True)
                                images_raw[torch.isnan(images_raw)] = 0.0

                                images1 = postprocess(images_raw).cpu()
                                images2 = postprocess(torch.transpose(images_raw, 0, 2)).cpu()
                                images3 = postprocess(torch.transpose(images_raw, 0, 3)).cpu()

                                _, xy_likelihood, _ = model(x=images_raw, temperature=temperature)
                                _, xz_likelihood, _ = model(x=torch.transpose(images_raw, 0, 2), temperature=temperature)
                                _, yz_likelihood, _ = model(x=torch.transpose(images_raw, 0, 3), temperature=temperature)

                            step_data_xy[steps].append(torch.mean(xy_likelihood).item())
                            step_data_xz[steps].append(torch.mean(xz_likelihood).item())
                            step_data_yz[steps].append(torch.mean(yz_likelihood).item())

                    means_xy = [np.mean(np.array(step_data_xy[steps])[~np.isnan(step_data_xy[steps])]) for steps in range(3,33,2)]
                    stds_xy = [np.std(np.array(step_data_xy[steps])[~np.isnan(step_data_xy[steps])]) for steps in range(3,33,2)]
                    plt.errorbar(np.arange(3,33,2), means_xy, yerr=stds_xy)
                    plt.title('Likelihood for x-y plane')
                    plt.xlabel('Steps between anchor slices')
                    plt.ylabel('Average obj. over all slices')
                    plt.savefig(os.path.join(fig_dir, 'x_y_likelihood.png'))
                    plt.close()

                    means_xz = [np.mean(np.array(step_data_xz[steps])[~np.isnan(step_data_xz[steps])]) for steps in range(3,33,2)]
                    stds_xz = [np.std(np.array(step_data_xz[steps])[~np.isnan(step_data_xz[steps])]) for steps in range(3,33,2)]
                    plt.errorbar(np.arange(3,33,2), means_xz, yerr=stds_xz)
                    plt.title('Likelihood for x-z plane')
                    plt.xlabel('Steps between anchor slices')
                    plt.ylabel('Average obj. over all slices')
                    plt.savefig(os.path.join(fig_dir, 'x_z_likelihood.png'))
                    plt.close()

                    means_yz = [np.mean(np.array(step_data_yz[steps])[~np.isnan(step_data_yz[steps])]) for steps in range(3,33,2)]
                    stds_yz = [np.std(np.array(step_data_yz[steps])[~np.isnan(step_data_yz[steps])]) for steps in range(3,33,2)]
                    plt.errorbar(np.arange(3,33,2), means_yz, yerr=stds_yz)
                    plt.title('Likelihood for y-z plane')
                    plt.xlabel('Steps between anchor slices')
                    plt.ylabel('Average obj. over all slices')
                    plt.savefig(os.path.join(fig_dir, 'y_z_likelihood.png'))
                    plt.close()

                    ll_sum = np.array(means_xz) + np.array(means_yz)
                    valid_idx = np.where(ll_sum > 0)[0]
                    steps = int(steps_list[valid_idx[ll_sum[valid_idx].argmin()]])
                    print('Optimal step size: {}'.format(steps))

                    with open(os.path.join(fig_dir, 'result.json'), 'w') as fp:
                        json.dump({'steps': steps}, fp)

            else:
                print('Using user-entered step size {}...'.format(args.steps))
                steps = args.steps

            print('Sampling images...')
            mean, logs = model.prior(None, None)
            alpha = 1-torch.reshape(torch.linspace(0,1,steps=steps),(-1,1,1,1))
            alpha = alpha.to(device)

            num_imgs = int(np.ceil(hparams['patch_size'] / steps) + 1)
            z = gaussian_sample(mean, logs, temperature)[:num_imgs,...]
            z = torch.cat([alpha*z[i,...] + (1-alpha)*z[i+1,...] for i in range(num_imgs-1)])
            z = z[:hparams['patch_size'], ...]

            images_raw = model(z=z, temperature=temperature, reverse=True)
            images_raw[torch.isnan(images_raw)] = 0.0
            images1 = postprocess(images_raw).cpu()
            images2 = postprocess(torch.transpose(images_raw, 0, 2)).cpu()
            images3 = postprocess(torch.transpose(images_raw, 0, 3)).cpu()

        write_video(images1, 'xy', hparams, stack_dir)
        write_video(images2, 'xz', hparams, stack_dir)
        write_video(images3, 'yz', hparams, stack_dir)
    
    print('Finished!')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)