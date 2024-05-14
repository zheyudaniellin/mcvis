"""
plot_2D_visibilities.py

Plot the visibilities in 2D, i.e., u-v plane

The original file is based on pdspy.plotting.plot_2D_visibilities.py
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pdb
import astropy.visualization as vsl

def axis_scale(axis_unit):
    """
    record the scaling and labels for the default axis
    """
    if axis_unit == 'lambda':
        scale = 1
        unit = r'$\lambda$'
    elif axis_unit == 'klambda':
        scale = 1e3
        unit = r'k$\lambda$'
    else:
        raise ValueError('axis_unit unknown')

    return {'scale':scale, 'unit':unit}

def default_quant_scale(quant):
    """
    record the scaling and labels for the default quantity
    """
    if quant == 'real':
        scale = 1
        tag = 'Real Part'
    elif quant == 'imag':
        scale = 1
        tag = 'Imaginary Part'
    elif quant == 'amp':
        scale = 1
        tag = 'Amplitude [Jy]'
    elif quant == 'phase':
        scale = 1.0
        tag = 'Phase [rad]'
    else:
        raise ValueError('quant unknown')
    return {'scale':scale, 'tag':tag}

def add_colormap(ax, vis, quant, axis_unit='klambda', north='right', 
        vmin=None, vmax=None, default_range=True, 
        cmap=None, **kwargs):
    """
    add the plot to the existing ax

    Add the x,y axes labels by default, so it's easier to make them consistent with the image

    Parameters
    ----------
    vis : visibility object
        this object should have u, v, real, imag as a 2D array already
    quant : str
        any of the quantities from the visibility object. 
        realistically, it should be 'real', 'image', 'amp', 'phase'
    default_range : bool
        If True, then apply interpret vmin and vmax based on quant
        real, amp will set vmin=0 and vmax based on input
        imag will use set the range between -vmax and vmax
        phase will be in between 0 and 2 pi
    cmap : 
        matplotlib colormap
    """

    # determine the x and y axis

    # we know that u and v are 2D arrays, but they each should be a 1D grid
    u = vis.u[0,:]
    v = vis.v[:,0]

    axis_arg = axis_scale(axis_unit)

    # determine the horizontal and vertical axes
    if north == 'right':
        plth, pltv = v, u
        labh, labv = 'v', 'u'
    elif north == 'up':
        plth, pltv = u, v
        labh, labv = 'u', 'v'
    else:
        raise ValueError('north unknown')
    extent = np.array([plth[0], plth[-1], pltv[0], pltv[-1]])/axis_arg['scale']

    # fetch the 2d image
    scale = default_quant_scale(quant)['scale']
    if quant == 'real':
        im2d = vis.real
        def_cmap = plt.cm.viridis
        def_vmin = 0
        def_vmax = None
        stretch = vsl.AsinhStretch()
    elif quant == 'imag':
        im2d = vis.imag
        def_cmap = plt.cm.RdBu_r
        def_vmin = None
        def_vmax = None
        stretch = vsl.LinearStretch()
    elif quant == 'amp':
        im2d = vis.amp
        def_cmap = plt.cm.viridis
        def_vmin = 0
        def_vmax = None
        stretch = vsl.AsinhStretch()
    elif quant == 'phase':
        im2d = vis.phase
        def_cmap = plt.cm.RdBu_r
        def_vmin = None
        def_vmax = None
        stretch = vsl.LinearStretch()
    else:
        raise ValueError('quant unknown')

    if north == 'up':
        im2d = im2d.T

    inp = im2d / scale

    # determine the value limits
    if vmin is None:
        vmin = def_vmin
    if vmax is None:
        vmax = def_vmax

    # determine the colormap
    if cmap is None:
        cmap = def_cmap

    # determine the stretch
    norm = vsl.ImageNormalize(inp, stretch=stretch,
        interval=vsl.ManualInterval(vmin=vmin, vmax=vmax))

    # make the plot
    pc = ax.imshow(inp.T, origin='lower', extent=extent, 
            norm=norm, cmap=cmap)

    # add axis labels by default
    ax.set_xlabel('%s [%s]'%(labh, axis_arg['unit']))
    ax.set_ylabel('%s [%s]'%(labv, axis_arg['unit']))

    # don't invert the axis, because it messes up plotting multiple images

    return pc

def plot_double(ax, o2d, m2d, quant, ticks=None):
    """
    show the 2D visibilities
    compare on a single plot

    axes : list of ax
        there should be 2 ax
    o2d : visibilities object
        visibilities for the observations
    m2d : visibilities object
        visibilities for the model
    """
    dim = o2d.real.shape
    npix = dim[0]

    # ===== real part ====
    ax = axes[0]

    # How to scale the real part.
    vmin = min(0, o2d.real.min())
    vmax = o2d.real.max()

    # plot observed
    ax.contour(o2d.u/1e3, o2d.v/1e3, o2d.real, colors='k')

    # plot model
    ax.contour(m2d.u/1e3, m2d.v/1e3, m2d.real, colors='C0')


    c2d = m2d.real.reshape((npix,npix))[xmin:xmax,xmin:xmax][:,::-1]
    ax.contour(c2d, cmap='jet')




def plot_2D_visibilities_old(axes, o2d, m2d, ticks=None):
    """
    OBSOLETE
    show the 2D visibilities

    axes : list of ax
        there should be 2 ax
    o2d : visibilities object
        visibilities for the observations
    m2d : visibilities object
        visibilities for the model
    """
    if ticks is None:
        ticks = np.array([-250,-200,-100,0,100,200,250])

    dim = o2d.real.shape
    npix = dim[0]

    # How to scale the real part.
    vmin = min(0, o2d.real.min())
    vmax = o2d.real.max()

    pdb.set_trace()

    # Calculate the pixel range to show.
    xmin = int(round(npix/2 + ticks[0]/(binsize/1e3)))
    xmax = int(round(npix/2 + ticks[-1]/(binsize/1e3)))
    ymin, ymax = xmin, xmax

    # Show the real component.
    ax = axes[0]
#    c2d = o2d.real.reshape((npix, npix))[xmin:xmax,xmin:xmax][:,::-1]
    c2d = o2d.real[xmin:xmax,xmin:xmax][:,::-1]
    ax.imshow(c2d, origin='lower', interpolation='nearest', 
        vmin=vmin, vmax=vmax, cmap='jet')

    c2d = m2d.real.reshape((npix,npix))[xmin:xmax,xmin:xmax][:,::-1]
    ax.contour(c2d, cmap='jet')


    # scale the imaginary part
    vmax = o2d.real.max()
    vmin = - vmax 

    # show the imaginary part
    ax = axes[1]
    c2d = o2d.imag.reshape((npix,npix))[xmin:xmax,xmin:xmax][:,::-1]
    ax.imshow(c2d, origin='lower', interpolation="nearest", 
        vmin=vmin, vmax=vmax, cmap='jet')

    c2d = m2d.imag.reshape((npix,npix))[xmin:xmax,xmin:xmax][:,::-1]
    ax.contour(c2d, cmap='jet')

    # adjust the axes ticks
    transform1 = ticker.FuncFormatter(Transform(xmin,xmax, binsize/1e3, '%.0f'))


    ax = axes[0]
    for ax in axes:
        ax.set_xticks(npix/2+ticks[1:-1]/(binsize/1e3) - xmin)
        ax.set_yticks(npix/2+ticks[1:-1]/(binsize/1e3) - ymin)
        ax.get_xaxis().set_major_formatter(transform1)
        ax.get_yaxis().set_major_formatter(transform1)

        ax.set_xlabel('U [k$\lambda$]')
        ax.set_xlabel('V [k$\lambda$]')



