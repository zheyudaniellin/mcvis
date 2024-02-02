"""
modeling.py

manage the modeling

- create a directory to store the temperature radmc3d calculations
- write inputs for radmc3d
- read calculations from radmc3d
- FFT using galario
- calculate chi-squared
"""
import numpy as np
import pdb
import os
import sys
import importlib
from . import radmc3d
from . import natconst
from . import image
au = natconst.au
rad = np.pi / 180

class manager(object):
    """
    general manager to produce the visibilities
    """
    def __init__(self, config):
        self.config = config

    def prepare_modeling(self, mname='running'):
        """
        Attributes
        ----------
        radmc3d : radmc3d_employee
        """
        self.rundir = os.path.join(self.config['outdir'], mname)

        # create the directory
        os.system('rm -rf %s'%self.rundir)
        os.system('mkdir %s'%self.rundir)

        # delete the previous calculation if there were any
        try: 
            delattr(self, 'images')
        except:
            pass

        # ==== model ====
        # determine the model that was supplied by the user

        # add the path to the module
        path = self.config['model']['path']
        sys.path.append(path)

        # object can't be pickled if we store a module
#        self.model = importlib.import_module(self.config['model']['name'])

        model_module = importlib.import_module(self.config['model']['name'])
        self.model = model_module.maker()

        # now dislodge the path
        sys.path.remove(path)

        # ==== prerun some settings for radmc3d ====
        self.radmc3d = radmc3d_employee(self.rundir)

        # create the grid
        xbound = [ival * au for ival in self.config['radmc3d']['xbound']]
        ybound = [ival * rad for ival in self.config['radmc3d']['ybound']]
        self.radmc3d.get_grid(
            nx=self.config['radmc3d']['nx'], 
            xbound=xbound, 
            ny=self.config['radmc3d']['ny'], 
            ybound=ybound, 
            act_dim=[1,1,0],
            )

        # write the radmc3d.inp file
        self.radmc3d.get_inp(scattering_mode_max=self.config['radmc3d']['scattering_mode_max']
                )

    def run(self, par):
        """
        calculate a particular model from the user, and calculate the image
        Arguments
        ---------
        par : dict
            the dictionary of parameters for the user model to use and other calculations
        """
        # calculate inputs for radmc3d
        res = self.model.run(par,
            self.radmc3d.grid.x,
            self.radmc3d.grid.y,
            self.radmc3d.grid.z,
            self.radmc3d.grid.w
            )

        self.radmc3d.write_dust(res['dustdensity'], res['dusttemperature'])
        self.radmc3d.write_opac(res['kabs'], res['ksca'])

        # create the star
        self.radmc3d.get_stars(mstar=par['mstar']*natconst.ms)

        # create the image for radmc3d for each wavelength
        ndata = len(self.config['visibilities']['file'])
        ims = []
        for i in range(ndata):
            # imaging wavelength
            freq = self.config['visibilities']['freq'][i] * 1e9
            wav = np.array([natconst.cc / freq]) * 1e4
            image.write_camera_wavelength(wav, fdir=self.rundir)

            # settings for the image
            npix = self.config['visibilities']['npix'][i]
            dpc = self.config['dpc']
            pixelsize_au = self.config['dpc'] * self.config['visibilities']['pixelsize'][i]
            imsizeau = int(npix * pixelsize_au)
            """
            imsizeau_y = self.config['aspect_ratio'][i]

            zoomau = '%d %d %d %d'%(-imsizeau/2, imsizeau/2, -imsizeau_y/2, imsizeau_y/2)
            """

            im = self.radmc3d.make_image(
                npix=npix, 
                sizeau=imsizeau,
                stokes=False, circ=False, secondorder=True,
                nphot_scat=self.config['radmc3d']['nphot_scat'], 
                incl=par['incl'], 
                dpc=dpc)

            # convert units
            im.convert_stokes_unit('jyppix')

            ims.append(im)

        self.images = ims

    def get_convs(self, bmaj, bmin, bpa):
        """
        for convenience, calculate the images using convolution. The arguments require the beam info, which should be lists with the same length as self.images. 

        After convolution, the default is to convert the stokes unit from jyppix to jypbeam. 

        Parameters
        ----------
        bmaj : list
        bmin : list
        bpa : list
        """
        self.convs = [None] * len(self.images)
        for i in range(len(self.images)):
            self.convs[i] = self.images[i].convolve_gaussian(bmaj[i], bmin[i], bpa[i])
            self.convs[i].convert_stokes_unit('cgs')
            self.convs[i].convert_stokes_unit('jypbeam')

# ======================================================
# parts on radmc3d 
# ======================================================
class radmc3d_employee(object):
    """
    handle the modeling specific to radmc3d
    """
    def __init__(self, outdir):
        """
        Attributes
        ----------
        outdir : str
            The directory to write the files and run radmc3d
        """
        self.outdir = outdir

    def get_grid(self, xbound=[5*au, 300*au], nx=[128],
        ybound=[80*rad, 90*rad, 100*rad], ny=[64, 64],
        zbound=[0, 2*np.pi], nz=[64],
        wbound=[100, 1e4], nw=[3], 
        act_dim=[1,1,0],):
        """
        create and write the grid
        """
        # setup the grid object
        grid = radmc3d.grid.regularGrid()
        grid.make_spatial(crd_sys='sph', act_dim=act_dim,
            xbound=xbound, nx=nx,
            ybound=ybound, ny=ny)

        grid.get_cell_center()
        grid.write_spatial(fdir=self.outdir)

        # wavelength
        grid.make_wavelength(wbound=wbound, nw=nw)
        grid.write_wavelength(fdir=self.outdir)

        self.grid = grid

    def get_stars(self, mstar=1):
        """ produce the stars.inp file
        """
        src = radmc3d.radiation.discrete_stars()
        src.set_grid(self.grid)

        src.add_star(natconst.rs, mstar*natconst.ms, natconst.ts, [0,0,0])

        src.write_stars(fdir=self.outdir)
        self.src = src

    def get_inp(self, scattering_mode_max='0', mc_scat_maxtauabs='30'):
        """
        write the radmc3d.inp file
        """
        radpar = {
            'istar_sphere': '0',
            'setthreads': '1',
            'mc_scat_maxtauabs': mc_scat_maxtauabs,
            'scattering_mode_max' : scattering_mode_max,
            'alignment_mode' : 0
            }

        radmc3d.utils.write_radpar(radpar, self.outdir)

    def write_dust(self, density, temperature):
        """
        """
        # field data
        dat = radmc3d.data.fieldData()
        dat.set_grid(self.grid)

        dat.dustrho = density
        dat.write_dustrho(fdir=self.outdir, binary=True)

        dat.dusttemp = temperature
        dat.write_dusttemp(fdir=self.outdir, binary=True)

    def write_opac(self, kabs, ksca):
        """
        calculate the opacity
        """
        # opacity
        op = radmc3d.dust.opacity()
        op.set_ext('manual')
        op.w = self.grid.w

        op.kabs =np.zeros_like(op.w) + kabs

        op.ksca = np.zeros_like(op.w) + ksca

        # prepare manager
        mop = radmc3d.dust.opacityManager()
        mop.add_opac(op)
        mop.write_dustopac(fdir=self.outdir)
        mop.write_opac(fdir=self.outdir)

        self.mop = mop

    def make_image(self, npix=200, sizeau=100, 
        stokes=False, circ=False, secondorder=False, 
        nphot_scat=0, incl=45, zoomau=None, dpc=100):
        """
        call radmc3d to calculate the image

        Make sure you're in the directory with the data
        """

        arg = {
            'npix':npix, 
            'sizeau':sizeau, 
            'stokes':stokes, 
            'circ':circ, 
            'secondorder':secondorder, 
            'nphot_scat':nphot_scat, 
            'incl':incl, 
            'zoomau':zoomau
            }
        com = image.commandImage()
        for ikey in arg.keys():
            setattr(com, ikey, arg[ikey])
        com.set_wavelength(loadlambda=True)

        com.get_command()

        com.make(fname='myimage.out')

        # read image
        fname = os.path.join(self.outdir, 'myimage.out')
        im = image.intensity()
        im.read(fname)
        im.grid.set_dpc(dpc)

        return im

