"""
pipeline.py

The main pipeline to interface with various codes. The main flow is:
    - read user inputs
    - read observations
    - construct sampler
    - determine parameters (dynesty, emcee)
    - produce radiation transfer
    - fourier transform
    - evaluate level of fit
    - determine new parameters (dynesty, emcee)
    - produce results
        - The results can depend on the sampler code, but overall it should be the same. 
        - Because of this, it makes more sense to dedicate a main pipeline and only call dynesty or emcee when sampling

This utlizes sampling_emcee.py or sampling_dynesty.py for sampling

"""
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

from . import interferometry as uv
from . import observations
from . import user
from . import modeling
from . import parameters
from . import ft
from . import plotting
from . import sampling_emcee
from . import sampling_dynesty
from . import sampling

# ========
def run():
    """
    run the fitting pipeline
    """
    # ==== read the configuration file ====
    cwd = os.getcwd()
    config = user.load_config(fdir=cwd)

    # ==== create output directory ====
    os.system('rm -rf %s'%config['outdir'])
    os.system('mkdir %s'%config['outdir'])

    # ==== read data ====
    obs = observations.load_data(config)

    # ==== prepare the modeling ====
    mod = modeling.manager(config)
    mod.prepare_modeling()

    # ==== parameters ====
    parm = parameters.manager()
    parm.add_plan(config['parameters'])
    parm.fill_default(mod.model.default_par())

    # ==== go to running directory ====
    # use this directory for conducting the radiation transfer calculations 
    # we will move to that directory to make things easy
    os.chdir(mod.rundir)

    # ==== begin dynesty sampling ====
    print('Conducting parameter space sampling')
    if config['sampler'] == 'dynesty':
        # file to save checkpoints
        checkpoint_file = os.path.join(config['outdir'], 'dynesty_sampler.p')

        res = sampling_dynesty.run(config, obs, mod, parm, checkpoint_file)

        print('Finished dynesty')

        # save results as a dynesty results object
        # we need this to access dynesty's plotting routines
        fname = os.path.join(config['outdir'], 'dynesty_results.pickle')
        sampling_dynesty.write_results(res, fname)

    elif config['sampler'] == 'emcee':
        res = sampling_emcee.run(config, obs, mod, parm)

        print('Finished emcee')

        # ==== output ====
        sampling_emcee.save_output(res, parm.free_parameters, datdir=config['outdir'])
    else:
        raise ValueError('sampler unknown')

    # ==== return to working directory ====
    os.chdir(cwd)

    # we can delete the modeling directory
    os.system('rm -rf %s'%mod.rundir)

# ================================
# ==== default visualizations ====
# ================================
def default_results():
    """
    calculate some default results
    """
    # ==== read the configuration file ====
    cwd = os.getcwd()
    config = user.load_config(fdir=cwd)

    # ==== read data ====
    obs = observations.load_data(config)
    obs.get_vis1d()
    obs.get_vis2d()

    # ==== prepare the modeling ====
    mod = modeling.manager(config)
    mod.prepare_modeling(mname='bestmodel')

    # ==== parameters ====
    parm = parameters.manager()
    parm.add_plan(config['parameters'])
    parm.fill_default(mod.model.default_par())

    # ==== plot depending on the sampler ====
    if config['sampler'] == 'dynesty':
        truths = sampling_dynesty.do_results_dynesty(config, obs, parm)
    elif config['sampler'] == 'emcee':

        # plot diagnostics of emcee
        sampling_emcee.plot(config, parm)

        # determine the bestfit values
        bf = sampling_emcee.get_bestfit(config, burnsteps=0)
        truths = bf[0]

    else:
        raise ValueError('samper unknown')

    # ==== run a final model ====
    # remember to do it at in the dedicated directory
    os.chdir(mod.rundir)
    final = sampling.evaluate(truths, obs, mod, parm)
    os.chdir(cwd)

    ftm = final['ftm']

    # calculate the 1D visibilitiy for the model
    ftm.get_vis1d()
    ftm.get_vis2d()

    # calculate the image from the 2d visibility of the model
    ftm.get_img_from_vis2d()
    # convert the units
    for i in range(len(ftm.vis2d)):
        # quick hack
        ftm.img_from_vis2d[i].psfarg = obs.im[i].psfarg
        ftm.img_from_vis2d[i].convert_stokes_unit('cgs')
        ftm.img_from_vis2d[i].convert_stokes_unit('jypbeam')

    # let's convolve the image for easier comparison
    #obs.get_psfarg()
    #mod.get_convs(obs.psfarg['bmaj'], obs.psfarg['bmin'], obs.psfarg['bpa'])

    # ==== plot comparisons ====
    # compare 1d visibility profiles
    fig, axgrid = plotting.obsmod.plot1dvis(obs, ftm)
    fig.savefig(os.path.join(config['outdir'], 'obsmod_vis1d.png'))
    plt.close()

    # compare 2d visibilities
    figsize = (9, 6)
    for i in range(len(config['visibilities']['file'])):
        # real and imaginary part
        fig, axgrid = plotting.obsmod.plot2dvis(obs, ftm, 
                quant=['real', 'imag'],
                inx=i, figsize=figsize, north='up')
        fig.savefig(os.path.join(config['outdir'], 'obsmod_vis2d_%d_complex.png'%i))
        plt.close()

        # amplitude and phase part
        fig, axgrid = plotting.obsmod.plot2dvis(obs, ftm, 
                quant=['amp', 'phase'],
                inx=i, figsize=figsize, north='up')
        fig.savefig(os.path.join(config['outdir'], 'obsmod_vis2d_%d_ampphase.png'%i))
        plt.close()

    # compare 2d images
    figsize = (9, 4*len(obs.im))
    fig, axgrid = plotting.obsmod.plot2dimg(obs, ftm, 
            north='up', axis_unit='au', figsize=figsize)
    fig.savefig(os.path.join(config['outdir'], 'obsmod_img2d.png'))
    plt.close()

    print('default results finished')

