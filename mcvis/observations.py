"""
keep all relevant observation data
including visibility and images
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
from . import interferometry as uv
from . import image
arcsec = 4.84813681e-6      # radians

def load_data(config):
    """
    read the visibility data and default image file
    """
    # read visibilities
    nvis = len(config['visibilities']['file'])
    obs = manager()

    obs.add_header(config['visibilities'])

    for i in range(nvis):
        data = uv.Visibilities()
        data.read(filename=config['visibilities']['file'][i])

        # Set any weights < 0 to be equal to zero to prevent odd fits.
        data.weights[data.weights < 0] = 0

        # scale the weights
        data.weights *= config['visibilities']['weight'][i]

        # go ahead and take out zero weights, since those don't matter anyway
        reg = data.weights > 0
        if False in reg:
            data = uv.Visibilities(data.u[reg], data.v[reg], data.freq, 
                data.real[reg], data.imag[reg], data.weights[reg])

        obs.add_vis(data)

        # read corresponding image
        im = image.intensity()
        dpc = config['dpc']
        im.read_fits(config['visibilities']['image_file'][i], dpc)

        f = config['visibilities']['freq'][i] 

        im.grid.set_frequency(np.array([f]) * 1e9)
        im.grid.get_wavelength()
        rms = config['visibilities']['image_noise'][i]
        im.set_rms(np.array([rms]))

        obs.add_im(im)

    return obs

class manager():
    """
    class to manage the observations and all its derivatives
    """
    def __init__(self):
        self.vis = []
        self.im = []

    def add_header(self, hdr):
        """
        this is the component that corresponds to config['visibilities']
        """
        self.hdr = hdr

    def add_vis(self, data):
        """
        Add the visibility object
        Note that the baselines u, v are in lambda

        Arguments
        ---------
        interferometry.Visibilities
        """
        self.vis.append(data)

    def add_im(self, im):
        self.im.append(im)

    def get_vis1d(self, gridsize1D=20, ):
        """
        calculate a 1D profile of the visibilities for plotting
        """
        self.vis1d = []
        for ivis in self.vis:
            g = uv.average(ivis, gridsize=gridsize1D, 
                    radial=True, log=True, 
                    logmin=ivis.uvdist[np.nonzero(ivis.uvdist)].min()*0.95,
                    logmax=ivis.uvdist.max()*1.05
                )

            self.vis1d.append(g)

    def get_vis2d(self, gridsize=256):
        """
        calculate the 2D image of the visibilities
        it's easier for plotting

        We simply want a 2D distribution of the complex amplitude on a grid of u and v

        Parameters
        ----------
        gridsize : int
            argument for interferometry.grid
        """
        self.vis2d = []
        self.vis2d_binsize = []
        for i, ivis in enumerate(self.vis):
            # calculate the binsize
            max_uvdist = ivis.uvdist.max() * 1.05
            binsize = 2 * max_uvdist / gridsize
#            binsize = 1. / (self.hdr['npix'][i]*self.hdr['pixelsize'][i]*arcsec)

            # grid the data
            g = uv.grid(self.vis[i], gridsize=gridsize, binsize=binsize)

            # reshape the quanities, since we know it's 2D anyways
            for ikey in ['u', 'v', 'real', 'imag', 'weights', 'uvdist', 'amp', 'phase']:
                setattr(g, ikey, getattr(g, ikey).reshape(gridsize, gridsize))

            self.vis2d_binsize.append(binsize)
            self.vis2d.append(g)

    def get_psfarg(self):
        """
        get the bmaj, bmin, bpa each as a list and organize it into a dict
        """
        ndata = len(self.im)
        bmaj = [0] * ndata
        bmin = [0] * ndata
        bpa = [0] * ndata
        for i, im in enumerate(self.im):
            bmaj[i] = im.psfarg['bmaj']
            bmin[i] = im.psfarg['bmin']
            bpa[i] = im.psfarg['bpa']

        self.psfarg = {'bmaj':bmaj, 'bmin':bmin, 'bpa':bpa}


