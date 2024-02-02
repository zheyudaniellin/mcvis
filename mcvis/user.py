"""
user.py

Anything related to the user-interface
write the configuration file as a readable json file

"""
import sys
import json
import os

def default_config():
    c = {
        # output directory
        "outdir": '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcvis_results/project1', 
        # visibilities info
        "visibilities":{
            # the hdf5 file
            "file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/IRAS04302/IRAS04302_SBLB_continuum.uv.hdf5"],
            "pixelsize":[0.003],
            "freq":[225.0], # [GHz]
            "npix":[2000],
            "weight":[1], 

            # The coordinate center with respect to the center of the image
            # units in arcsec
            "D-RA":[0],
            "D-Dec":[0],

            # the image file for comparison
            "image_file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/IRAS04302/IRAS04302_SBLB_continuum_robust_0.5.image.tt0.fits"],
            "image_noise":[1.45e-5], # image noise in Jy/beam 
        }, 
        # settings
        "sampler" : "dynesty", # dynesty or emcee
        "dynesty":{
            "nlive_init": 250, 	# The number of live points to use for Dynesty.
            "nlive_batch": 250, # Number of live points per batch for dynamic nested sampling
            "maxbatch" : 0,	# Maximum number of batches to use.
            "dlogz" : 0.05,	# Stopping threshold for nested sampling.
            "walks" : 25, 	# Number of random walk steps to use to generate a sample
            "resume" : False,	# resume the fitting if there was already a file 
        }, 
        "emcee" : {
            "nwalkers" : 20, 
            "nsteps" : 1000, 
        }, 
        
        "radmc3d":{
            "nx":[64], 
            "xbound":[5, 300], # [au]
            "ny":[64, 64], 
            "ybound":[85, 90, 95], # degrees
            "nphot_scat":1e3,
            "scattering_mode_max": 1, 
            }, 
        # the file to produce the model
        "model":{
            "path":"/home/zdl3gk/mainProjects/edisk_edgeon/mcvis/models", 
            "name":"Qconst"
            }, 
        # parameters for the physical model and others that needs to be fitted
        "parameters": {
            'dRA':{'free':True, 'value':0, 'limits':[-0.1,0.1], 'tag':r'$\Delta$ RA'}, 

            'dDec':{'free':True, 'value':0, 'limits':[-0.1,0.1], 'tag':r'$\Delta$ Dec'},
            'PA':{'free':True, 'value':0, 'limits':[-10, 10], 'tag':'PA'},
            'incl':{'free':True, 'value':45, 'limits':[0, 90], 'tag':r'$i$'},
            'mstar':{'free':False, 'value':1, 'limits':[1, 5], 'tag':r'$M_{*}$'}, 
            'R0':{'free':True, 'value':300, 'limits':[200, 400], 'tag':r'$R_{0}$'},
            'Rc':{'free':False, 'value':100, 'limits':[0, 200], 'tag':r'$R_{c}$'},
            }, 
        "dpc":100, 
    }

    return c

def write_config(config, fdir=''):
    """
    config : dict
    """
    fname = os.path.join(fdir, 'config.json')

    with open(fname, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(fdir=''):
    """
    read the configuration json file
    """
    fname = os.path.join(fdir, 'config.json')
    with open(fname, 'r') as f:
        config = json.load(f)

    return config


