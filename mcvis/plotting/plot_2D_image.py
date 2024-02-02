"""
basic functions to plot the continuum in the image domain
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.visualization as vsl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .. import natconst
au = natconst.au

def axis_scale(axis_unit, dpc=1):
    """
    record the scaling and labels for the default axis
    """
    # axis unit
    if axis_unit == 'cm':
        scale = 1
        unit = 'cm'
    elif axis_unit == 'au':
        scale = au
        unit = 'au'
    elif axis_unit == 'arcsec':
        scale = au * dpc
        unit = 'arcsec'
    else:
        raise ValueError('axis_unit unknown')

    return {'scale':scale, 'unit':unit}

def default_quant_scale(quant):
    """
    record the scaling and labels 
    We only have Stokes I to consider
    """
    if quant == 'I':
        scale = 1e-3
        tag = 'Stokes I'
        unit = 'mJy/beam'
    else:
        raise ValueError('quant unknown')
    return {'scale':scale, 'tag':tag, 'unit':unit}

def add_colormap(ax, im, quant, north='right', axis_unit='au', cmap=None,
    vmin=None, vmax=None):
    """
    default colormap plot
    """
    qscale = default_quant_scale(quant)

    # pick the color map
    if quant == 'I':
        im2d = im.I / qscale['scale']

        def_vmin = 0
        def_vmax = None

        def_cmap = plt.cm.gist_yarg

        stretch = vsl.LinearStretch()

    else:
        raise ValueError('quant unknown: %s'%quant)

    if vmin is None:
        vmin = def_vmin
    if vmax is None:
        vmax = def_vmax

    norm = vsl.ImageNormalize(im2d, stretch=stretch, 
        interval=vsl.ManualInterval(vmin=vmin, vmax=vmax))

    # axis unit
    axis_arg = axis_scale(axis_unit, dpc=im.grid.dpc)

    if axis_unit == 'cm':
        unitfac = 1
    elif axis_unit == 'au':
        unitfac = au
    elif axis_unit == 'arcsec':
        unitfac = au * im.grid.dpc
    else:
        raise ValueError('axis_unit unknown')

    # determine north 
    if north == 'right':
        pltx = im.grid.x / unitfac
        plty = im.grid.y / unitfac
        labh, labv = r'$\Delta$ Dec', r'$\Delta$ RA'
    elif north == 'up':
        pltx = im.grid.y / unitfac
        plty = im.grid.x / unitfac
        im2d = im2d.T
        labh, labv =r'$\Delta$ RA', r'$\Delta$ Dec'
    else:
        raise ValueError('north unknown')

    extent = (pltx[0], pltx[-1], plty[0], plty[-1])

    out = {}
    out['pc'] = ax.imshow(im2d.T, origin='lower', extent=extent,
        cmap=cmap, norm=norm)

    # add axis labels by default
    ax.set_xlabel('%s [%s]'%(labh, axis_arg['unit']))
    ax.set_ylabel('%s [%s]'%(labv, axis_arg['unit']))

    return out

def add_contour(ax, im, quant, clevs, north='right', axis_unit='au', **kwargs):
    """ add a 2d contour
    """
    im2d = getattr(im, quant)

    # axis unit
    if axis_unit == 'cm':
        unitfac = 1
    elif axis_unit == 'au':
        unitfac = au
    elif axis_unit == 'arcsec':
        unitfac = au * im.grid.dpc
    else:
        raise ValueError('axis_unit unknown')

    # determine north 
    if north == 'right':
        pltx = im.grid.x / unitfac
        plty = im.grid.y / unitfac
    elif north == 'up':
        pltx = im.grid.y / unitfac
        plty = im.grid.x / unitfac
        im2d = im2d.T
    else:
        raise ValueError('north unknown')

    extent = (pltx[0], pltx[-1], plty[0], plty[-1])

    out = {}
    out['cc'] = ax.contour(im2d.T, clevs, origin='lower', extent=extent,
        **kwargs)

    return out

def add_length(ax, dpc, len_au=100, loc=None, axis_unit='au', color='k',
    **kwargs):
    """
    add a line segment in au to represent the length scale of the image
    Parameters
    ----------
    len_au : float
        the length in au
    loc : tuple of 2 floats
        the location of the center of the line segment in units of axis_unit
    """
    if axis_unit == 'cm':
        fac = au
    elif axis_unit == 'au':
        fac = 1
    elif axis_unit == 'arcsec':
        fac = 1 / dpc
    else:
        raise ValueError('axis_unit unknown')

    # default location
    if loc is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        loc = (xlim[0] + 0.1 * np.diff(xlim), ylim[0] + 0.1 * np.diff(ylim))

    pltx = len_au * fac * np.array([-0.5, 0.5]) + loc[0]
    plty = pltx * 0 + loc[1]
    ax.plot(pltx, plty, color=color, **kwargs)

    txt = '%d au'%len_au
    ax.text(loc[0], loc[1], txt, va='bottom', ha='center', color=color)


# ==== some default combo ====
