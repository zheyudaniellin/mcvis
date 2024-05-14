"""
midassess.py
"""
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import importlib
import dynesty
import dynesty.plotting as dyplot
import sys
sys.path.append('../..')
import mcvis
sys.path.remove('../..')

def main():
    # ==== settings ====
    datdir = '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcvis_results/project1'

    # ==== read the configuration file ====
    config = mcvis.user.load_config()

    # ==== load in the model ====
    # we need this in order to restore model
    sys.path.append(config['model']['path'])
    model_module = importlib.import_module(config['model']['name'])
    model = model_module.maker()
    sys.path.remove(config['model']['path'])

    # ==== read the samper ====
    checkpoint_file = os.path.join(datdir, 'dynesty_sampler.p')
    sampler = dynesty.DynamicNestedSampler.restore(checkpoint_file)

    res = sampler.results
    bf = mcvis.dyhelp.get_bestfit(res)

    # ==== parameters ====
    parm = mcvis.parameters.manager()
    parm.add_plan(config['parameters'])
    parm.fill_default(model.default_par())

    # ==== plotting ====
    truths = bf['median']
    labels = [parm.get_tag(ikey) for ikey in parm.free_parameters]

    # Plot a summary of the run.
    fig, axes = dyplot.runplot(res)
    fig.savefig(os.path.join(config['outdir'], 'summary.png'))
    plt.show()

    # Plot traces and 1-D marginalized posteriors.
    # values through each step?
    fig, axes = dyplot.traceplot(res,
        truths=truths,
        labels=labels,
        fig=plt.subplots(3, 2, figsize=(16, 12)))
    fig.tight_layout()
    fig.savefig(os.path.join(config['outdir'], 'trace.png'))
    plt.show()

    pdb.set_trace()

if __name__ == '__main__':
    main()

