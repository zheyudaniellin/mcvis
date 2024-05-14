"""
GSS30IRS3_mcvis.py

GSS30IRS3
"""
import sys
sys.path.append('../..')
import mcvis

def main():
    # ==== settings ====
    config = {
        # output directory
        "outdir": '/scratch/zdl3gk/mainProjects/edisk_edgeon/mcvis_results/GSS30IRS3',
        # visibilities info
        "visibilities":{
            # the hdf5 file
            "file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/GSS30IRS3/GSS30IRS3_SBLB_continuum.uv.hdf5"], 
            "pixelsize":[0.003],
            "freq":[225],
            "npix":[800],
            "weight":[1],

            # The coordinate center with respect to the center of the image
            # units in arcsec
            "D-RA":[0],
            "D-Dec":[0],

            # the image file for comparison
            "image_file":["/scratch/zdl3gk/mainProjects/edisk_edgeon/data/GSS30IRS3/GSS30IRS3_SBLB_continuum_robust_0.0.image.tt0.fits"],
            "image_noise":[1.685e-5], # image noise in Jy/beam 
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
            "nwalkers" : 12, 
            "nsteps" : 2,
            }, 
        "radmc3d":{
            "nx" : [32],
            "xbound" : [1, 120], # [au]
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
            'dRA':{'free':False, 'value':0, 'limits':[-0.5,0.5], 'tag':r'$\Delta$ RA'},

            'dDec':{'free':False, 'value':0, 'limits':[-0.5,0.5], 'tag':r'$\Delta$ Dec'},
            'PA':{'free':True, 'value':-70, 'limits':[-100, -40], 'tag':'PA'},
            'incl':{'free':True, 'value':70, 'limits':[50, 90], 'tag':r'$i$'},
            'R0':{'free':True, 'value':50, 'limits':[20, 120], 'tag':r'$R_{0}$'},
            'Rc':{'free':False, 'value':50, 'limits':[0, 200], 'tag':r'$R_{c}$'},
            'Hc':{'free':True, 'value':5, 'limits':[2, 8], 'tag':r'$H_{c}$'},
            'hs':{'free':False, 'value':1.25, 'limits':[1, 1.5], 'tag':r'$h_{s}$'}, 
            'tau0':{'free':True, 'value':0.35, 'limits':[0.1,2.0], 'tag':r'$\tau_{0}$'}, 
            'T0':{'free':True, 'value':20, 'limits':[1,200], 'tag':r'$T_{0}$'}, 
            'q':{'free':False, 'value':0.5, 'limits':[0, 1], 'tag':r'$q$'},
            'mstar':{'free':False, 'value':0.46, 'limits':[0.3, 0.7], 'tag':r'$M_{*}$'}, 
            },
        "dpc":138.4,
        }

    # ==== write ====
    mcvis.user.write_config(config)

    # ==== run code ====
    mcvis.pipeline.run()

def plot():
    mcvis.pipeline.default_results()

if __name__ == '__main__':
#    main()
    plot()
