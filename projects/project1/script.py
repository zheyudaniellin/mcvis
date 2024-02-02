"""
script.py

command script
"""
import numpy as np
import sys
sys.path.append('../..')
import mcvis

def main():
    # ==== settings ====
    config = {
        # output directory
        "outdir": '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcvis_results/project1',
        # visibilities info
        "visibilities":{
            # the hdf5 file
            "file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/IRAS04302/IRAS04302_SBLB_continuum.uv.hdf5"],
            "pixelsize":[0.003],
            "freq":[225],
            "npix":[1200],
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
            "nsteps" : 800,
            }, 
        "radmc3d":{
            "nx" : [32],
            "xbound" : [5, 330], # [au]
            "ny" : [32],
            "ybound" : [80, 90], # degrees
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
            'dRA':{'tag':r'$\Delta$ RA', 
                'free':True, 'value':0, 'limits':[-0.5,0.5]},
            'dDec':{'tag':r'$\Delta$ Dec', 
                'free':True, 'value':0, 'limits':[-0.5,0.5]},
            'PA':{'tag':'PA', 
                'free':True, 'value':175, 'limits':[160, 190]},
            'incl':{'tag':r'$i$', 
                'free':True, 'value':87, 'limits':[80, 90]},
            'R0':{'tag':r'$R_{0}$', 
                'free':True, 'value':300, 'limits':[200, 400]},
            'Rc':{'tag':r'$R_{c}$', 
                'free':False, 'value':100, 'limits':[0, 200]},
            'Hc':{'tag':r'$H_{c}$', 
                'free':True, 'value':6, 'limits':[2, 9]},
            'hs':{'tag':r'$h_{s}$', 
                'free':False, 'value':1.25, 'limits':[1, 1.5]},
            'tau0':{'tag':r'$\tau_{0}$', 
                'free':True, 'value':0.35, 'limits':[0.2,1.0]},
            'T0':{'tag':r'$T_{0}$', 
                'free':True, 'value':7.5,'limits':[1,50]},
            'q':{'tag':r'$q$', 
                'free':False, 'value':0.5, 'limits':[0, 1]},
            'mstar':{'tag':r'$M_{*}$', 
                'free':False, 'value':1.6, 'limits':[1.5, 1.7]}, 
            },
        "dpc":160,
        }

    # ==== write ====
    mcvis.user.write_config(config)

    # ==== run code ====
    mcvis.pipeline.run()

def plot():
    mcvis.pipeline.default_results()

if __name__ == '__main__':
    main()
    plot()
