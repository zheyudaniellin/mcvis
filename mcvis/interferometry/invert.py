"""
invert.py

Taken from pdspy invert.py

Use this to simply invert the data from the visibility plane to the image plane

"""
import numpy as np
from scipy.fftpack import ifft2, fftshift, ifftshift, fftfreq
from .. natconst import arcsec
import pdb
import matplotlib.pyplot as plt

# ==== convolution schemes ====
def pillbox(u, v, delta_u, delta_v):

    m = 1

    arr = np.ones(u.shape, dtype=float)*u.size

    arr[np.abs(u) >= m * delta_u / 2] = 0
    arr[np.abs(v) >= m * delta_v / 2] = 0

    return arr

def exp_sinc(u, v, delta_u, delta_v):

    alpha1 = 1.55
    alpha2 = 2.52
    m = 6

    arr = np.sinc(u / (alpha1 * delta_u)) * \
            np.exp(-1 * (u / (alpha2 * delta_u))**2)* \
            np.sinc(v / (alpha1 * delta_v)) * \
            np.exp(-1 * (v / (alpha2 * delta_v))**2)

    arr[np.abs(u) >= m * delta_u / 2] = 0
    arr[np.abs(v) >= m * delta_v / 2] = 0

    arr = arr/arr.sum() * arr.size

    return arr

# ==== functions ====
def vis_to_img(vis2d, convolution='pillbox'):
    """
    I made some modifications to suit how my data is

    the vis2d quantities have their first index along dec and second index along ra, which corresponds to v and u

    Parameters
    ----------
    vis2d : visibility object
        the data should already be gridded
    """
    # determine bin size
    # dec
    v1d = vis2d.v[:,0]

    # ra
    u1d = vis2d.u[0,:]

    # calculate beam size
    binsize = u1d[1] - u1d[0]

    imsize = len(u1d)

    # set up axis information

    # dec
    x = fftshift(fftfreq(imsize, binsize)) / arcsec

    # ra
    y = fftshift(fftfreq(imsize, binsize)) / arcsec


    f = vis2d.freq

    image = np.zeros([imsize, imsize, len(f)])

    # determine convolution scheme
    if convolution == "pillbox":
        conv_func = pillbox
    elif convolution == "expsinc":
        conv_func = exp_sinc
    else:
        raise ValueError('convolution scheme unknown')

    # loop through the frequency
    for i in range(len(f)):
        inp = vis2d.real + 1j * vis2d.imag
#        im = fftshift(ifft2(ifftshift(inp))).real * imsize**2
        im = fftshift(ifft2(ifftshift(inp))).real

        convolve = fftshift(ifft2(ifftshift(conv_func(vis2d.u, vis2d.v, binsize, binsize)))).real

        # the second index corresponds to RA
        # but we need to invert it, because increasing the index decreases RA
        # we also need to invert the first axis
        # not sure why......
        image[:,:,i] = (im/convolve)[::-1,::-1]

    return x, y, f, image


def get_beam(imsize, binsize, convolution='pillbox'):
    """
    just to get the beam
    """
    real = np.ones([imsize, imsize])
    imag = np.zeros_like(real)

    for i in range(len(f)):
        inp = 0

