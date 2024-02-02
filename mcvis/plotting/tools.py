"""
tools.py
"""
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_colorbar_bottom(pc, ax, fig, nbins=5):
    """
    add a colorbar with tick labels towards the bottom
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cb = fig.colorbar(pc, cax=cax, orientation='horizontal')
    cb.ax.locator_params(nbins=nbins)

    return cb, cax

def set_colorbar_top(pc, ax, fig, nbins=5):
    """
    add a colorbar with tick labels towards the bottom
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cb = fig.colorbar(pc, cax=cax, orientation='horizontal')
    cb.ax.locator_params(nbins=nbins)

    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    return cb, cax

def set_colorbar_right(pc, ax, fig, nbins=5):
    """
    add the coloarbar with tick labels to the right
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(pc, cax=cax, orientation='vertical')
    cb.ax.locator_params(nbins=nbins)

    return cb, cax


