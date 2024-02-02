"""
model_template.py

Only use function, and not do object oriented calulations
To be user friendly, use as few hidden options in the coding as possible

The functions that are necessary
    default_par()
    run()
"""
import numpy as np
import pdb
import sys
import natconst
au = natconst.au

# leave some constants here
gas_to_dust = 100

def default_par():
    """
    the default parameter dictionary
    """
    par = {
        # edge of the disk [au]
        'R0': {'tag':r'$R_{0}$', 'value':300}, 

        # characeristic radius [au]
        'Rc': {'tag':r'$R_{c}$', 'value':100},

        # dust scale height [au]
        'Hc': {'tag':r'$H_{c}$', 'value':8}, 

        # flaring index
        'hs': {'tag': r'$h_{s}$', 'value':1.25 }, 

        # characertistic optical depth
        'tau0': {'tag':r'$\tau_{0}$', 'value':0.01},

        # temperature at R0 [K]
        'T0': {'tag':r'$T_{0}$', 'value':20},

        # power-law index for temperature
        'q': {'tag':r'$q$', 'value':0.5}, 

        # stellar mass [msun]
        'mstar': {'tag':r'$M_{*}$', 'value':1.0}, 

        }

    return par

def run(par, r, theta, phi, w):
    """
    get the necessary dust density, dust temperature, dust opacity
    sometimes the structures are correlated, that's why we should just use a single encapsing function

    Parameters
    ----------
    par : dict
    r : 1d ndarray
    theta : 1d ndarray
    phi : 1d ndarray
    w : 1d ndarray
        for wavelength
    """
    rr, tt, pp = np.meshgrid(r, theta, phi, indexing='ij')

    # dust opacity 
    kabs, ksca = get_dustopacity(par, w)

    # density
    dustdensity = get_dustdensity(par, rr, tt, pp)

    # temperature
    temperature = get_dusttemperature(par, rr, tt, pp)

    return {
        'dustdensity':dustdensity[...,None],
        'dusttemperature':temperature[...,None], 
        'kabs':kabs, 
        'ksca':ksca}

def get_rho0(R0, mstar):
    Q = 1.0
    return mstar / np.pi / np.sqrt(2*np.pi) / R0**3 / Q

def get_dustdensity(par, rr, tt, pp):
    """
    par : dictionary
        include the necessary parameters
    """
    # read in the parameters
    Rc = par['Rc'] * au
    R0 = par['R0'] * au
    Hc = par['Hc'] * au
    hs = par['hs'] 
    mstar = par['mstar'] * natconst.ms

    cyrr = rr * np.sin(tt)
    zz = rr * np.cos(tt)

    # scale height
    hh = Hc * (cyrr / Rc)**hs

    # characeristic gas density
    rho0 = get_rho0(R0, mstar)

    # dust density
    rho = rho0 * (rr / R0)**(-3) * np.exp(-0.5*(zz/hh)**2) / gas_to_dust

    # density cut-off
    reg = cyrr >= R0
    rho[reg] = 0.

    reg = rho <= 1e-30
    rho[reg] = 1e-30
    rho = rho[...,None]

    return rho

def get_dusttemperature(par, rr, tt, pp):
    """
    """
    T0 = par['T0']
    R0 = par['R0'] * au
    q = par['q']

    temp = T0 * (rr / R0)**(-q)
    return temp[...,None]

def get_dustopacity(par, w):
    """
    par : dictionary
        kabs = absorption opacity
    """
    tau0 = par['tau0']
    mstar = par['mstar'] * natconst.ms
    R0 = par['R0'] * au

    rho0 = get_rho0(R0, mstar)

    kabs = tau0 / (rho0 / gas_to_dust) / R0 

    # output
    kabs = kabs + np.zeros_like(w)
    ksca = np.zeros_like(kabs)
    return kabs, ksca

    

class maker(object):
    def __init__(self):
        pass

    def default_par(self):
        return default_par()

    def run(self, par, r, theta, phi, w):
        return run(par, r, theta, phi, w)


