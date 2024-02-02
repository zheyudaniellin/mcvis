"""
sampling_emcee.py

Follow the sampling process using emcee
includes how to select parameter, evaluate best-fit, and produce results

This should be similar to pipeline_dynesty.py

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import emcee
import pickle
import corner

from . import interferometry as uv
from . import observations
from . import user
from . import modeling
from . import parameters
from . import sampling
from . import ft

# ==== helper functions ====
def spread_gaussian_ball(parC, parW, nwalkers):
    """ produce starting positions for different walkers
    parC : 1d list or ndarray
    parW : 1d list or ndarray
        the standard deviation of the distribution
    nwalkers : int
    """
    ndim = len(parC)
    pos = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        pos[:,i] = np.random.normal(parC[i], parW[i], nwalkers)

    return pos

def spread_uniform(limits, nwalkers):
    """
    produce starting positions for different walkers. Assume a uniform distribution, rather than a gaussian distribution

    Parameters
    ----------
    limits : 1d list of tuples
        the limits for each parameter
    nwalkers : int
    """
    ndim = len(limits)
    pos = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        lim = limits[i]
        pos[:,i] = (lim[1] - lim[0]) * np.random.uniform(size=nwalkers) + lim[0]

    return pos

def spread_uniform2(vals, limits, nwalkers, f=1):
    """
    produce starting positions for different walkers. Assume a uniform distribution, rather than a gaussian distribution

    Sometimes I don't want to go all the way to the limits, but only up to so
me fraction from my first guess. Use f to do that

    Parameters
    ----------
    vals : list
    limits : 1d list of tuples
        the limits for each parameter
    nwalkers : int
    """
    # check if f is in between 0 and 1
    if (f < 0) | (f >= 1):
        raise ValueError('f should be (0, 1]')

    ndim = len(limits)
    pos = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        val = vals[i]
        lim = limits[i]

        # check if val is within the limits
        if (val - lim[0]) * (val - lim[1]) >= 0:
            raise ValueError('the value should be within the limits')

        lo = lim[0] + (val - lim[0]) * f
        up = val + (lim[1] - val) * f

        # calculate the position 
        pos[:,i] = (up - lo) * np.random.uniform(size=nwalkers) + lo

    return pos

# ==== likeliehood functions ====
def logl(ptheta, obs, mod, parm):
    """
    The log likelihood function to be called that only returns a value
    specifically for emcee, since it needs to take the priors here and not as a separate function

    Arguments
    ---------
    ptheta : list of floats
        the values of each parameter for a particular instant
    obs : observations.manager
    mod : modeling.manager
    parm : parameters.manager
    """

    # check if we fail the prior or not
    fail_prior = False

    for i, ikey in enumerate(parm.free_parameters):
        limits = parm.limits(ikey)
        if (ptheta[i] < limits[0]) | (ptheta[i] > limits[1]):
            fail_prior = True

    # if we don't fail the prior, then we are good to evaluate the probability
    if fail_prior:
        lnp = - np.inf
    else:
        res = sampling.evaluate(ptheta, obs, mod, parm)
        lnp = res['lnp']

    return lnp

# ==== sampler ====
def run(config, obs, mod, parm):
    """
    use emcee to conduct the sampling
    """
    # determine the dimensions
    ndim = len(parm.free_parameters)
    nwalkers = config['emcee']['nwalkers']
    nsteps = config['emcee']['nsteps']

    # initialize the locations
    # produce the inputs for spread_uniform
    lim = []
    for ikey in parm.free_parameters:
        lim.append( parm.limits(ikey) )
#    pos = spread_uniform(lim, nwalkers)
    vals = parm.value_of_free_parameters()
    pos = spread_uniform2(vals, lim, nwalkers, f=0.5)

    # prepare the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, logl, args=(obs, mod, parm)
    )

    # create a directory for where the calculations will be done

    # run the sampling
    sampler.run_mcmc(pos, nsteps, progress=False)

    # return the results
    return sampler

def save_output(sampler, free_parameters, fname='emcee_samples.npy', datdir=''):
    """
    save the output of the sampler
    """
    # ==== output ====
    samples = sampler.get_chain(flat=False)

    out = {'samples':samples, 'free_parameters':free_parameters}

    fname = os.path.join(datdir, 'emcee_samples.json')
    with open(fname, 'wb') as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

def read_output(fname='emcee_samples.npy', datdir=''):
    """
    read the default output of the sampler
    """
    # ==== read the sample results ====
    fname = os.path.join(datdir, 'emcee_samples.json')
    with open(fname, 'rb') as f:
        out = pickle.load(f)

    samples = out['samples']
    free_parameters = out['free_parameters']

    # check the dimension
    dim = samples.shape
    if dim[2] != len(free_parameters):
        raise ValueError('internal error: the number of dimensions from the samples should be the same as the number of elements in free_parameters')

    return samples, free_parameters

# ==== visualizations ====
def flatten_samples(samples, burnsteps=0):
    """
    flatten the samples after removing some steps as burn-in
    """
    nsteps, nwalkers, ndim = samples.shape

    if burnsteps >= nsteps:
        raise ValueError('number of steps to discard for burn-in is larger than the number of steps')

    s = samples[burnsteps:,:,:]
    dim = s.shape
    flat = np.reshape(s, (dim[0]*dim[1], dim[2]))
    return flat

def measure_parameter(samples, burnsteps=0,):
    """
    obtain the measurement of the parameter, including the median and uncertainty, based on the samples

    Parameters
    ----------
    samples : numpy ndarray
        before flattening it
    """
    flat = flatten_samples(samples, burnsteps=burnsteps)
    ndim = len(flat[0,:])

    # Get the best fit parameters.
    median = np.median(flat, axis=0)
    sigma = np.std(flat, axis=0)

    # it might be useful to get the 16, 84 percentile too
    p16 = np.zeros([ndim])
    p84 = np.zeros_like(p16)
    for i in range(ndim):
        p16[i], p84[i] = np.percentile(flat[:,i], [16, 84])

    # output bestfit parameters
    out = {
        'median':median,
        'sigma':sigma,
        'p16':p16, 'p84':p84,
    }

    return out

def get_bestfit(config, burnsteps=0):
    """
    Use this function as a formal way to define the bestfit based on measurements of the sampling. 
    Make it convenient, so we only need to know the name of the config file

    We're interested in the bestfit value and the upper limit and lower limit

    Depending on the sampling technique, we would have different definitions
    """
    # ==== read the sample results ====
    samples, free_parameters = read_output(datdir=config['outdir'])

    # measure the parameter
    measurement = measure_parameter(samples, burnsteps=burnsteps)

    best = measurement['median']
    up = measurement['p84']
    lo = measurement['p16']

    return best, lo, up

def plot_trace(samples, tags, figsize=(10,8), title_fmt=".2f"):
    """
    plot the samples and how it varies through each step
    The second column is the marginalized distribution
    """
    nsteps, nwalkers, ndim = samples.shape
    nrow, ncol = ndim, 1
    fig, axgrid = plt.subplots(nrow,ncol,sharex=True, sharey=False, 
            squeeze=False, figsize=figsize)
    axes = axgrid.flatten()

    meas = measure_parameter(samples, burnsteps=0)

    for i in range(ndim):
        # parameter values through each step
        ax = axgrid[i,0]
        ax.plot(samples[:,:,i], 'k', alpha=0.3)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(tags[i])

        # marginalized distribution
        lo, mi, hi = meas['p16'][i], meas['median'][i], meas['p84'][i]
        """
        ax = axgrid[i,1]
        inp = samples[:,:,i].flatten()
        ax.hist(inp, bins=20, fill=False)
        ax.axvline(x=lo, color='k', linestyle='--')
        ax.axvline(x=mi, color='k')
        ax.axvline(x=hi, color='k', linestyle='--')
        """
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(mi), fmt(mi-lo), fmt(hi-mi))
#        title = '{0} = {1}'.format(tags[i], title)
        ax.text(1.02, 0.5, title, va='center', ha='left', transform=ax.transAxes)

    fig.tight_layout()
    return fig, axes

# ==== execution ====
def plot(config, parm):
    """
    a set of default plots if we used emcee

    Parameters
    ----------
    """
    # default is to not throw away any samples
    burnsteps = 0

    # ==== read the sample results ====
    samples, free_parameters = read_output(datdir=config['outdir'])

    # calculate the best fit
    bf, lo, up = get_bestfit(config, burnsteps=burnsteps)

    # ==== plotting ====
    tags = [parm.tag(ikey) for ikey in free_parameters]

    # plot the trace
    fig, axes = plot_trace(samples, tags)
    fname = os.path.join(config['outdir'], 'trace_emcee.png')
    fig.savefig(fname)
    plt.close()

    # plot the corner plot
    flat = flatten_samples(samples, burnsteps=burnsteps)
    fig = corner.corner(flat, labels=tags, 
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, )
    fname = os.path.join(config['outdir'], 'corner_emcee.png')
    fig.savefig(fname)
    plt.close()

