import math
import torch
import os
import numpy as np
from numba import jit


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size),\
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2**n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def create_image_slideshow(image_dir, picture_format=None):
    '''
    Create image slideshow using FFMPEG:
        https://www.ffmpeg.org/

    
    '''
    if picture_format is None:
        raise Exception('Please enter a valid string for \'picture format\'.')

    mov_path = os.path.join(image_dir,'slideshow.mp4')
    if os.path.exists(mov_path):
        os.remove(mov_path)
    os.system('ffmpeg -framerate 5 -i {} -pix_fmt yuv420p {}'.format(os.path.join(image_dir,picture_format), mov_path))


@jit
def two_point_correlation(im, dim, var=0):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.

    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction

    var should be set to the pixel value of the pore-space. (Default 0)

    The input image im is expected to be three-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[2] #y-axis
        dim_2 = im.shape[1] #z-axis
        dim_3 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        dim_3 = im.shape[2] #y-axis
    elif dim == 2: #z-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[2] #y-axis
        dim_3 = im.shape[1] #z-axis

    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3 - r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a + r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a + r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a + r, n2]

                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r] / (float(lmax))
    return two_point