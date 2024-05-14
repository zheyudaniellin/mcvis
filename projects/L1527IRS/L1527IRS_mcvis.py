"""
L1527IRS_mcvis.py

command script
"""
import os
import numpy as np
import sys
sys.path.append('../..')
import mcvis

def main():
    # ==== settings ====
    config = {
        # output directory
        "outdir": '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcvis_results/L1527IRS',
        # visibilities info
        "visibilities":{
            # the hdf5 file
            "file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/L1527IRS/L1527IRS_SBLB_continuum.uv.hdf5"],
            "pixelsize":[0.003],
            "freq":[225],
            "npix":[700],
            "weight":[1],

            # The coordinate center with respect to the center of the image
            # units in arcsec
            "D-RA":[0],
            "D-Dec":[0],

            # the image file for comparison
            "image_file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/L1527IRS/L1527IRS_SBLB_continuum_robust_-0.5.image.tt0.fits"],
            "image_noise":[2.374e-5], # image noise in Jy/beam 
            },
        # settings
        "sampler" : "emcee", 
        "dynesty":{
            "nlive_init": 250,  # The number of live points to use for Dynesty.
            "nlive_batch": 250, # Number of live points per batch for dynamic nested sampling
            "maxbatch" : 0,     # Maximum number of batches to use.
            "dlogz" : 0.5,     # Stopping threshold for nested sampling.
            "walks" : 25,       # Number of random walk steps to use to generate a sample
            "resume" : False,   # resume the fitting if there was already a file 
            },
        "emcee" : {
            "nwalkers" : 20,
            "nsteps" : 1000,
            }, 
        "radmc3d":{
            "nx" : [32],
            "xbound" : [5, 150], # [au]
            "ny" : [32],
            "ybound" : [75, 90], # degrees
            "nphot_scat" : 1e3,
            "scattering_mode_max" : '0',
            },
        # the file to produce the model
        "model":{
            "path" : "/home/zdl3gk/mainProjects/edisk_edgeon/mcvis/models",
            "name" : "Qconst", 
            },
        # parameters for the physical model and others that needs to be fitted
        "parameters": {
            'dRA':{'free':True, 'value':0, 'limits':[-0.5,0.5], 'tag':r'$\Delta$ RA'},
            'dDec':{'free':True, 'value':0, 'limits':[-0.5,0.5], 'tag':r'$\Delta$ Dec'},
            'PA':{'free':True, 'value':0, 'limits':[-30, 30], 'tag':'PA'},
            'incl':{'free':True, 'value':90, 'limits':[80, 110], 'tag':r'$i$'},
            'R0':{'free':True, 'value':80, 'limits':[10, 150], 'tag':r'$R_{0}$'},
            'Rc':{'free':False, 'value':50, 'limits':[0, 200], 'tag':r'$R_{c}$'},
            'Hc':{'free':True, 'value':5, 'limits':[1, 9], 'tag':r'$H_{c}$'},
            'hs':{'free':False, 'value':1.25, 'limits':[1, 1.5], 'tag':r'$h_{s}$'}, 
            'tau0':{'free':True, 'value':0.35, 'limits':[0.01,2.0], 'tag':r'$\tau_{0}$'}, 
            'T0':{'free':True, 'value':40, 'limits':[1, 100], 'tag':r'$T_{0}$'}, 
            'q':{'free':False, 'value':0.5, 'limits':[0, 1], 'tag':r'$q$'},
            'mstar':{'free':False, 'value':0.5, 'limits':[0.1, 1.0], 'tag':r'$M_{*}$'}, 
            },
        "dpc":140,
        }

    # ==== write ====
    os.system("rm -rf config.json")
    mcvis.user.write_config(config)

    # ==== run code ====
    mcvis.pipeline.run()

def plot():
    mcvis.pipeline.default_results()

if __name__ == '__main__':
#    main()
    plot()
