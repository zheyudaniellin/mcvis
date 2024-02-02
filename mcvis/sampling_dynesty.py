"""
sampling_dynesty.py

"""
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import dynesty
import dynesty.utils as dyfunc
import dynesty.results as dyres
import dynesty.plotting as dyplot

from . import interferometry as uv
from . import observations
from . import user
from . import modeling
from . import parameters
from . import ft

# ================================
# ==== probability related  ====
# ================================
def logl(ptheta, obs, mod, parm):
    """
    The log likelihood function to be called that only returns a value

    Arguments
    ---------
    ptheta : list of floats
        the values of each parameter for a particular instant
    obs : observations.manager
    mod : modeling.manager
    parm : parameters.manager
    """

    res = evaluate(ptheta, obs, mod, parm)

    return res['lnp']

def ptform(utheta, parm=None):
    """
    priors
    Arguments
    ---------
    utheta : tuple of floats
        the unit cube?  

    parm : parameters.manager
        somehow, this has to be a keyword argument
    """
    p = []
    for i, ikey in enumerate(parm.free_parameters):
        limits = parm.get_limits(ikey)
        val = (limits[1] - limits[0]) * utheta[i] + limits[0]
        p.append(val)

    return p

# ================================
# ==== run sampler ====
# ================================
# Functions for saving the state of the Dynesty Sampler and loading a saved
# state.

def save_sampler(name, sampler, pool=None, dynamic=False):

    # Clear the random state, as it cannot be pickled.
    sampler.rstate = None

    # Clear the MPI pool, as it also cannot be pickled.
    sampler.pool = None
    sampler.M = map

    # If this is a DynamicNestedSampler object, also do the sub-sampler.
    if dynamic:
        sampler.sampler.rstate = None
        sampler.sampler.pool = None
        sampler.sampler.M = map

    # Save
    pickle.dump(sampler, open(name, "wb"))

    # Restore everything to the way it was before.
    sampler.rstate = np.random
    sampler.pool = pool
    if pool != None:
        sampler.M = pool.map
    else:
        sampler.M = map

    # Again, repeat for the sub-sampler, if DynamicNestedSampler.

    if dynamic:
        sampler.sampler.rstate = np.random
        sampler.sampler.pool = pool
        if pool != None:
            sampler.sampler.M = pool.map
        else:
            sampler.sampler.M = map

def load_sampler(name, pool=None, dynamic=False):
    # Load the sampler from the pickle file.
    sampler = pickle.load(open(name,"rb"))

    # Add back in the random state.
    sampler.rstate = np.random

    # Add the pool correctly.
    sampler.pool = pool
    if pool != None:
        sampler.M = pool.map
        sampler.queue_size = pool.size
    else:
        sampler.M = map

    # Add pool/random state correctly for the sub-sampler as well for
    # the DynamicNestedSampler class.
    if dynamic:
        sampler.sampler.rstate = np.random
        sampler.sampler.pool = pool
        if pool != None:
            sampler.sampler.M = pool.map
            sampler.sampler.queue_size = pool.size
        else:
            sampler.sampler.M = map

    return sampler

# A function to make useful plots as the sampling is running.
def plot_status(res, fdir='', ptform=None, labels=None, periodic=None):
    # Generate a plot of the trace.

    try:
        fig, ax = dyplot.traceplot(res, show_titles=True, trace_cmap="viridis",\
                connect=True, connect_highlight=range(5), labels=labels)
    except:
        # If it hasn't converged enough...
        fig, ax = dyplot.traceplot(res, show_titles=True, trace_cmap="viridis",\
                connect=True, connect_highlight=range(5), labels=labels, \
                kde=False)

    fname = os.path.join(fdir, 'in_process_tracplot.png')
    fig.savefig(fname)
    plt.close(fig)

    # Generate a bounds cornerplot.

    fig, ax = dyplot.cornerbound(res, it=res.niter-1, periodic=periodic, \
            prior_transform=ptform, show_live=True, labels=labels)

    fname = os.path.join(fdir, 'in_process_boundplot.png')
    fig.savefig(fname)
    plt.close(fig)


def run(config, obs, mod, parm, checkpoint_file):
    """
    the default way to run a sampler
    """

    # create the sampler or resume
    if config['dynesty']['resume'] is False:
        sampler = dynesty.DynamicNestedSampler(
            logl, ptform,
            ndim=len(parm.free_parameters),
            logl_args = (obs, mod, parm),
            ptform_kwargs = {'parm':parm},
            bound='multi',
            sample='rwalk',
            walks=config['dynesty']['walks'],
            )

        # normally, we should simply call this function
        sampler.run_nested(checkpoint_file=checkpoint_file, dlogz_init=0.1)

    else:
        sampler = dynesty.DynamicNestedSampler.restore(checkpoint_file)
        sampler.run_nested(resume=True, )

    res = sampler.results

    return res

def run_patrick(config, obs, mod, parm, checkpoint_file):
    """
    use patrick's way of calling the sampler
    The sampler is broken down to allow plotting and saving the status
    """
    periodic = None # don't handle periodic parameters for now
    labels = [parm.get_tag(ikey) for ikey in parm.free_parameters]

    # create the sampler or resume
    if config['dynesty']['resume'] is False:
        sampler = dynesty.DynamicNestedSampler(
            logl, ptform,
            ndim=len(parm.free_parameters),
            logl_args = (obs, mod, parm),
            ptform_kwargs = {'parm':parm},
            bound='multi',
            sample='rwalk',
            walks=config['dynesty']['walks'],
            )

    else:
        sampler = dyhelp.load_sampler(checkpoint_file, pool=None, dynamic=True)
        res = sampler.results

    if not sampler.base:
        for it, results in enumerate(
            sampler.sample_initial(
                dlogz=config['dynesty']['dlogz'],
                nlive=config['dynesty']['nlive_init'],
                save_samples=True,
                resume=config['dynesty']['resume'])
            ):

            # save the state of the sampler
            dyhelp.save_sampler(checkpoint_file, sampler, pool=None, dynamic=True)

            # Print out the status of the sampler.
            dyres.print_fn(results, sampler.it - 1, sampler.ncall,
                    dlogz=config['dynesty']['dlogz'],
                    logl_max = np.inf)

            # manually calculate the stopping criterion
#            logz_remain = np.max(sampler.sampler.live_logl) + \
#                    sampler.sampler.saved_logvol[-1]
            logz_remain = np.max(sampler.sampler.live_logl) + \
                    sampler.sampler.saved_run['logvol'][-1]

#            delta_logz = np.logaddexp(sampler.sampler.saved_logz[-1], \
#                    logz_remain) - sampler.sampler.saved_logz[-1]

            delta_logz = np.logaddexp(sampler.sampler.saved_run['logz'][-1], \
                    logz_remain) - sampler.sampler.saved_run['logz'][-1]



            # Every 1000 steps stop and make plots of the status.
            if (sampler.it - 1) % 1000 == 0 and delta_logz >= config['dynesty']['dlogz']:
                # Add the live points and get the results.

                sampler.sampler.add_final_live()

                res = sampler.sampler.results

                # Make plots of the current status of the fit.
                dyhelp.plot_status(res, fdir=config['outdir'],
                    ptform=sampler.prior_transform,
                    labels=labels, periodic=periodic)

                # If we haven't reached the stopping criteria yet, remove the
                # live points.

                sampler.sampler._remove_live_points()

        # Gather the results and make one final plot of the status.
        res = sampler.results
        dyhelp.plot_status(res, ptform=sampler.prior_transform, \
                labels=labels, periodic=periodic)

    for i in range(sampler.batch, config['dynesty']['maxbatch']):
        # Get the correct bounds to use for the batch.

        logl_bounds = dynesty.dynamicsampler.weight_function(sampler.results)
        lnz, lnzerr = sampler.results.logz[-1], sampler.results.logzerr[-1]

        # Sample the batch.

        for results in sampler.sample_batch(logl_bounds=logl_bounds, \
                nlive_new=config['dynesty']['nlive_batch']):

            # Print out the results.
            (worst, ustar, vstar, loglstar, nc,
                     worst_it, boundidx, bounditer, eff) = results

            results = (worst, ustar, vstar, loglstar, np.nan, np.nan,
                    lnz, lnzerr**2, np.nan, nc, worst_it, boundidx,
                    bounditer, eff, np.nan)

            dyres.print_fn(results, sampler.it - 1, sampler.ncall,\
                    nbatch=sampler.batch+1, stop_val=5, \
                    logl_min=logl_bounds[0], logl_max=logl_bounds[1])

        # Merge the new samples in.
        sampler.combine_runs()

        # Save the status of the sampler after each batch.
        dyhelp.save_sampler(checkpoint_file, sampler,
            pool=None,
            dynamic=True)

        # Get the results.
        res = sampler.results

        # Make plots of the current status of the fit.
        dyhelp.plot_status(res, ptform=sampler.prior_transform, \
                labels=labels, periodic=periodic)

    return res

""" obsolete
def run():
    # ==== read the configuration file ====
    cwd = os.getcwd()
    config = user.load_config(fdir=cwd)

    # ==== read data ====
    obs = observations.load_data(config)

    # ==== prepare the modeling ====
    mod = modeling.manager(config)
    mod.prepare_modeling()

    # ==== parameters ====
    parm = parameters.manager()
    parm.add_plan(config['parameters'])
    parm.fill_default(mod.model.default_par())

    # ==== begin dynesty sampling ====
    if config['sampler'] == 'dynesty':
        # file to save checkpoints
        checkpoint_file = os.path.join(config['outdir'], 'dynesty_sampler.p')

        res = do_sampler_dynesty_default(config, obs, mod, parm, checkpoint_file)
#        res = do_sampler_dynesty_patrick(config, obs, mod, parm, checkpoint_file)

        # save results as a dynesty results object
        # we need this to access dynesty's plotting routines
        fname = os.path.join(config['outdir'], 'dynesty_results.pickle')
        dyhelp.write_dynesty_results(res, fname)
        
    elif config['sampler'] == 'emcee':
        res = do_sampler_emcee(config, obs, mod, parm)
        
        # ==== output ====
        samples = sampler.get_chain(flat=False)
        fname = os.path.join(datdir, 'emcee_samples.npy')
        np.save(fname, samples)
    else:
        raise ValueError('sampler unknown')
"""

def write_results(res, fname):
    """
    output the dynesty.results object to a file
    """
    resdict = res.asdict()
    with open(fname, 'wb') as f:
        pickle.dump(resdict, f, pickle.HIGHEST_PROTOCOL)

def read_results(fname):
    """
    read the file and organize it into a dynesty.results object
    """
    with open(fname, 'rb') as f:
        resdict = pickle.load(f)

    res = dyfunc.Results(resdict)

    return res

# ================================
# ==== default visualizations ====
# ================================
def get_bestfit(res):
    """
    calculate some basic quantities
    res : dynesty.results object
    Returns
    -------
    out : dict
    """
    # Extract sampling results.
    samples = res.samples  # samples
    weights = res.importance_weights()

    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)

    # Resample weighted samples.
    samples = res.samples_equal()

    # Get the best fit parameters.
    median = np.median(samples, axis=0)
    sigma = samples.std(axis=0)

    out = {
        'mean':mean, 'cov':cov,
        'median':median, 'sigma':sigma
        }

    return out

def plot(config, obs, parm):
    """
    default results if we used dynesty
    """
    # ==== read the sample results ====
    fname = os.path.join(config['outdir'], 'dynesty_results.pickle')
    res = dyhelp.read_dynesty_results(fname)
    bf = dyhelp.get_bestfit(res)

    truths = bf['median']

    # ==== plotting ====
    labels = [parm.get_tag(ikey) for ikey in parm.free_parameters]

    # Plot a summary of the run.
    fig, axes = dyplot.runplot(res)
    fig.savefig(os.path.join(config['outdir'], 'summary_dynesty.png'))
    plt.close()

    # Plot traces and 1-D marginalized posteriors.
    # values through each step?
    fig, axes = dyplot.traceplot(res,  
        truths=truths,  
        labels=labels,
        fig=plt.subplots(3, 2, figsize=(16, 12)))
    fig.tight_layout()
    fig.savefig(os.path.join(config['outdir'], 'trace_dynesty.png'))
    plt.close()

    # corner plot
    """
    fig, axes = dyplot.cornerplot(res, 
        truths=truths, show_titles=True,
        title_kwargs={'y': 1.04}, labels=labels,
        fig=plt.subplots(3, 3, figsize=(15, 15)))
    fig.savefig(os.path.join(config['outdir'], 'corner_dynesty.png'))
    plt.show()
    """

    return truths

