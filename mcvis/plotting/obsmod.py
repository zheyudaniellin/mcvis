"""
obsmod.py

A convenient default script to plot the observations vs model results
"""
import numpy as np
import matplotlib.pyplot as plt
from . import plot_1D_visibilities, plot_2D_visibilities, plot_2D_image, tools
import pdb

def plot1dvis(obs, ftm, figsize=(6, 6)):
    """
    plot the 1D visibility profile from the obs and model
    
    Arguments
    ---------
    obs : observations.manager
    ftm : ft.manager
        This should have the vis and vis1d attributes
    """
    ndata = len(ftm.vis1d)
    nrow = 1
    ncol = ndata
    fig, axgrid = plt.subplots(nrow,ncol,squeeze=False, figsize=figsize)
    axes = axgrid.flatten()

    for i in range(ndata):
        # plot the 1D visibilities
        ax = axes[i]
        plot_1D_visibilities.plot_1D_visibilities(ax, obs.vis1d[i], ftm.vis1d[i])

    fig.tight_layout()

    return fig, axes

def plot2dvis(obs, ftm, quant=['real'], inx=0, 
        north='right', axis_unit='klambda', 
        figsize=(8, 6)):
    """
    compare the 2d quantities in the visibility 
    only for the n-th data, because there's too much info involved

    Arguments
    ---------
    obs : observations.manager
        This should have the vis2d attribute
    mod : modeling.manager
        This should already have the images attribute
    ftm : ft.manager
        This should have the vis2d attribute
    """
    nrow = len(quant)
    ncol = 2
    fig, axgrid = plt.subplots(nrow,ncol, sharex=False, sharey=False, 
            squeeze=False, figsize=figsize)
    axes = axgrid.flatten()

    # get the n-th visibility
    ovis = obs.vis2d[inx]
    mvis = ftm.vis2d[inx]

    # iterate through different quantities 
    for i in range(len(quant)):
        tag = plot_2D_visibilities.default_quant_scale(quant[i])['tag']

        # determine the common range
        o2d = getattr(ovis, quant[i])
        if quant[i] in ['real', 'amp']:
            vmin = 0
            vmax = np.nanmax(o2d)
        elif quant[i] in ['imag']:
            vmax = np.nanmax(abs(o2d))
            vmin = -vmax
        elif quant[i] in ['phase']:
            vmin = - np.pi
            vmax = np.pi
        else:
            raise ValueError('quant unknown')

        # observations
        ax = axgrid[i,0]
        pc = plot_2D_visibilities.add_colormap(ax, ovis, quant[i], 
                axis_unit=axis_unit, north=north, 
                vmin=vmin, vmax=vmax
                )

        cb, cax = tools.set_colorbar_right(pc, ax, fig)
        cb.set_label(tag)
        del pc, cb, cax

        # model
        ax = axgrid[i,1]
        pc = plot_2D_visibilities.add_colormap(ax, mvis, quant[i],
                axis_unit=axis_unit, north=north,
                vmin=vmin, vmax=vmax
                )
        cb, cax = tools.set_colorbar_right(pc, ax, fig)

        cb.set_label(tag)
        del pc, cb, cax

    for ax in axes:
        ax.axhline(y=0, color='grey', alpha=0.3)
        ax.axvline(x=0, color='grey', alpha=0.3)

    if north == 'up':
        for ax in axes:
            ax.invert_xaxis()

    # label the obs and model
    axgrid[0,0].set_title('Observation')
    axgrid[0,1].set_title('Model')

    fig.tight_layout()

    return fig, axgrid

def plot2dimg(obs, ftm, 
        north='right', axis_unit='au',
        figsize=(8, 8)):
    """
    compare the 2d quantities in the image plane
    since each data only has one image, we can show across all the different data

    Arguments
    ---------
    obs : observations.manager
        This should have the vis2d attribute
    mod : modeling.manager
        This should already have the convs attribute
    """
    nrow = len(obs.im)
    ncol = 2
    fig, axgrid = plt.subplots(nrow,ncol, sharex=False, sharey=False,
            squeeze=False, figsize=figsize)
    axes = axgrid.flatten()

    quant = 'I'
    qscale = plot_2D_image.default_quant_scale(quant)

    ascale = plot_2D_image.axis_scale(axis_unit, dpc=obs.im[0].grid.dpc)

    tag = qscale['tag'] + '[%s]'%qscale['unit']

    # iterate through different datasets
    for i in range(len(obs.im)):
        # common range
        vmin = 0
        vmax = np.nanmax(obs.im[i].I) / qscale['scale']

        xlim = np.array([-1,1]) * ftm.img_from_vis2d[i].grid.x.max() / ascale['scale']

        # observations
        ax = axgrid[i,0]
        out = plot_2D_image.add_colormap(ax, obs.im[i], quant, 
                axis_unit=axis_unit, north=north,
                vmin=vmin, vmax=vmax
                )
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        pc = out['pc']

        cb, cax = tools.set_colorbar_right(pc, ax, fig)
        cb.set_label(tag)
        del pc, cb, cax

        # ==== model ====
        ax = axgrid[i,1]
        out = plot_2D_image.add_colormap(ax, ftm.img_from_vis2d[i], quant, 
                axis_unit=axis_unit, north=north,
                vmin=vmin, vmax=vmax
                )
        pc = out['pc']
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)

        cb, cax = tools.set_colorbar_right(pc, ax, fig)
        cb.set_label(tag)
        del pc, cb, cax

    for ax in axes:
        ax.axhline(y=0, color='grey', alpha=0.3)
        ax.axvline(x=0, color='grey', alpha=0.3)

    if north == 'up':
        for ax in axes:
            ax.invert_xaxis()

    # label the obs and model
    axgrid[0,0].set_title('Observation')
    axgrid[0,1].set_title('Model')

    fig.tight_layout()

    return fig, axgrid


