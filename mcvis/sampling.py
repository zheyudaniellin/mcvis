"""
sampling.py

some common functions for executing sampling techniques that sampling_emcee and sampling_dynesty requires

"""
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

from . import ft

def evaluate(ptheta, obs, mod, parm):
    """
    calculate the model images and its FT given some parameter

    By separating this out from logl(), we can use this function to get additional info than just the probability


    The gaussian probability function is
        p = 1/sqrt(2 pi err^2) exp(-0.5 (obs - mod)^2 / err^2)

    The log of a gaussian probability function is simply
        ln p = -0.5 * (obs-mod)^2 / err^2 - 0.5 ln(2 pi err^2)

    The weights is simply
        w = 1 / err^2

    Arguments
    ---------
    ptheta : list of floats
        the values of each parameter for a particular instant
    obs : observations.manager
    mod : modeling.manager
    parm : parameters.manager
    """
    config = mod.config # makes things easier

    # produce the parameter dictionary
    par = parm.get_value_dict(ptheta)

    # calculate all the images
    mod.run(par)

    # calculate the fourier transform
    ftm = ft.manager(config)

    # calculate the RA, Dec offset
    dRA = np.array(config['visibilities']['D-RA']) + par['dRA']
    dDec = np.array(config['visibilities']['D-Dec']) + par['dDec']
    ftm.match(obs, mod, dRA, dDec, par['PA'])

    # calculate the chi-squared
    ndata = len(obs.vis)
    lnp = 0
    for i in range(ndata):
        lnp += np.sum(
            -0.5*(obs.vis[i].real-ftm.vis[i].real)**2 * obs.vis[i].weights
            -0.5*(obs.vis[i].imag-ftm.vis[i].imag)**2 * obs.vis[i].weights
            + np.log(obs.vis[i].weights) - np.log(2*np.pi)
            )

    return {'ftm':ftm, 'images':mod.images, 'lnp':lnp}

