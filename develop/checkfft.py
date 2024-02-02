"""
check the fft

Some conclusions
- fft doesn't care about which direction is north, etc. As long as the indexing of the matrix are the same, the input image, visibility, and output image will maintain the same matrix dimensions and same direction

- np.fft algorithm requires the zero frequency to be at the corner (a very specific format). 

"""
from scipy.fft import ifftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import numpy as np
rad = np.pi / 180

# ==== brightness profiles ====
def brightness_gaussian(xx, yy, inc, pa, peak, width):
    """
    calculate a 2d gaussian profile
    xx, yy : 2d ndarray
        the dec and ra coordinates relative to the center
    inc : float
        inclination in radians
    pa : float
        position angle, east-of-north, in radians
    width : float
        the width of the gaussian
    peak : float
        the peak value of the gaussian
    """
    # calculate the principal frame
    xx_p = np.cos(pa) * xx + np.sin(pa) * yy
    yy_p = - np.sin(pa) * xx + np.cos(pa) * yy

    # calculate the disk midplane frame
    xx_d = xx_p
    yy_d = yy_p / np.cos(inc)

    # now calculate the disk gaussian
    img = peak * np.exp(-0.5*(xx_d**2 + yy_d**2) / width**2)

    return img

# ==== functions ====
def ft_image_to_visibility(img):
    """
    fourier tranform the image to the visibility 
    """
    pass

# ==== visualizations ====
def add_colormap(ax, x, y, img, north='right', invert_ra=True):
    """
    plot the image
    """
    if north == 'right':
        plth = x
        pltv = y
        labh = 'Dec'
        labv = 'RA'
    elif north == 'up':
        plth, pltv = y, x
        labh = 'RA'
        labv = 'Dec'
    else:
        raise ValueError('north unknown')
    extent = (plth[0], plth[-1], pltv[0], pltv[-1])

    if north == 'right':
        inp = img
    elif north == 'up':
        inp = img.T
    else:
        raise ValueError('north unknown')

    ax.imshow(inp.T, extent=extent, origin='lower')
    ax.set_xlabel(labh)
    ax.set_ylabel(labv)

    # if the north='up', then we can invert the ra
    if (north == 'up') and invert_ra:
        ax.invert_xaxis()

def plot1(x, y, tru, img, north='up'):
    """
    compare the input true image and the result after sampling
    """
    fig, axgrid = plt.subplots(1,2,sharex=False,sharey=False,squeeze=False,
            figsize=(9,6))
    axes = axgrid.flatten()

    ax = axes[0]
    add_colormap(ax, x, y, tru, north=north)
    ax.set_title('true sky brightness')

    ax = axes[1]
    add_colormap(ax, x, y, img, north=north)
    ax.set_title('after sampling visibilities')

    fig.tight_layout()
    plt.show()

# ==== pipelines ====
def main():
    # ===== settings ====
    # the dec
    x = np.linspace(-1, 1, 100)

    # the ra
    y = np.linspace(-1, 1, 80)

    xx, yy = np.meshgrid(x,y, indexing='ij')

    # ==== calulations ====
    tru = brightness_gaussian(xx, yy, 80*rad, 30*rad, 1, x.max()*0.2)

    # ==== ft ====
    # ft using numpy
    raw = np.fft.fft2(tru)

    # the matrix is shifted after the numpy fft implementation, so let's shift it back to what we would expect from observations where the zero frequency is at the center
    vis = np.fft.fftshift(raw)

    amp = np.abs(vis)
    phase = np.angle(vis)

    # sample the visibilities
    random_points = np.random.rand(len(x), len(y))
    reg = random_points>0.2
    obs = vis * 1
    obs[~reg] = 0
    obs_amp = np.abs(obs)
    obs_phase = np.angle(obs)

    plt.imshow(obs.real)
    plt.show()
    plt.imshow(obs.imag)
    plt.show()

    # ifft to image plane, but we need to reorder the matrix first
    inp = np.fft.ifftshift(obs)
    img = np.fft.ifft2(inp).real

    # ==== plotting ====
    ax = plt.gca()
    add_colormap(ax, x, y, img, north='up')
    plt.show()

    plot1(x, y, tru, img, north='up')

    pdb.set_trace()

if __name__ == '__main__':
    main()

