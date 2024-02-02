""" image.py
there are three main classes
- spectra
	the spectra produced by radmc3d
- cuts
	1d profiles of an image
- image
	2d data, like the stokes I, stokes Q, tau surface, tau3d
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pdb
import copy
import subprocess
from astropy.io import fits 
from scipy import signal
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.ndimage import map_coordinates
from . import natconst

# ====
# spectra
# ====
class spectrum(object):

    def __init__(self):
        pass

    def get_frequency(self):
        self.f = natconst.cc * 1e4 / self.w

    def read(self, fname):
        """
        reads the spectrum.out file for the spectrum. note the data structure is different from ordinary images, but this will still use the image object
        
        Parameters
        ----------

        fname   : str, optional
                 File name of the radmc3d output image (if omitted 'spectrum.out' is used)

        binary  : bool, optional
                 False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)
        """
        if binary:
            raise ValueError('binary file for spectrum is not done yet')

        with open(fname, 'r') as rfile:
            # Format number
            iformat = int(rfile.readline())

            nwav = int(rfile.readline())

            self.w = np.zeros([self.nwav], dtype=np.float64)

            self.image = np.zeros([self.nwav], dtype=np.float64)

            # Blank line
            dum = rfile.readline()

            for iwav in range(self.nwav):
                # actual data
                dum = rfile.readline()
                dum = dum.split()
                self.w[iwav] = float(dum[0])
                self.image[iwav] = float(dum[1])

    def plot(self, ax=None, **kwargs):
        """ plot the spectrum as a function of wavelength
        """
        if ax is None:
            ax = plt.gca()

        ax.plot(self.w, self.image, **kwargs)

# ====
# cut object 
# ====
class imageCut(object):
    """ an object to record the info of 
    a spatial cut through an image
    """
    def __init__(self):
        self.quantity = []

    def set_quantity(self, quantname, quant):
        """ sets a  2d quantity. the quantity can be anything
        as long as it's in nl by nw
        Parameters
        ----------
        quant : 2d np.ndarray
            the array
        quantname : str
            name of the quantity
        """
        self.quantity.append(quantname)
        setattr(self, quantname, quant)

    def set_laxis(self, laxis):
        self.laxis = laxis

    def set_xy(self, xpnt, ypnt):
        """ xpnt, ypnt in cm
        """
        self.xpnt = xpnt
        self.ypnt = ypnt

    def set_wavelength(self, w):
        self.w = w
        self.nw = len(self.w)

    def get_frequency(self):
        self.f = natconst.cc / self.w * 1e4

    def set_frequency(self, f):
        self.f = f
        self.nw = len(f)

    def get_wavelength(self):
        self.w = natconst.cc / self.f * 1e4

    def set_dpc(self, dpc):
        self.dpc = dpc

    def set_restfreq(self, restfreq):
        """ set the rest frequency for a line
        """
        self.restfreq = restfreq

    def get_velocity(self):
        """ get the line of sight velocity for lines
        in cm/s
        """
        if hasattr(self, 'f') is False:
            self.get_frequency()

        if hasattr(self, 'restfreq') is False:
            raise ValueError('restfreq must be known to get the velocity')

        self.v = natconst.cc * (self.restfreq - self.f) / self.restfreq

    def write_fits(self, quant, fname):
        """ write to a fits file
        """
        hdu = fits.PrimaryHDU(getattr(self, quant).T)

        # prepare some stuff
        setvals = [
            # keyword, value, comment
            ['NAXIS', 2, ''], 

            ['BTYPE', 'Intensity', ''],
            ['BUNIT', 'Jy/beam ', 'Brightness (pixel) unit'],
            ['BSCALE', 1.0, ''], 
            ['BZERO', 0.0, ''], 

            # write the axis stuff in: naxis, ctype, crval, crpix, cdelt, cunit
            # spatial axis
            ['NAXIS1', len(self.laxis), ''], 
            ['CTYPE1', 'OFFSET ', ''], 
            ['CRVAL1', self.laxis[0] / natconst.au / self.dpc, ''],
            ['CRPIX1', 1, ''], 
            ['CDELT1', (self.laxis[1] - self.laxis[0]) / natconst.au / self.dpc, ''], 
            ['CUNIT1', 'arcsec', ''], 

            # frequency axis
            ['NAXIS2', len(self.f), ''],
            ['CTYPE2', 'FREQ', ''], 
            ['CRVAL2', self.f[0], ''],
            ['CRPIX2', 1, ''],
            ['CDELT2', self.f[1] - self.f[0], ''],
            ['CUNIT2', 'Hz', ''],

            # stokes axis 
#            ['NAXIS3', 1, ''], 
#            ['CTYPE3', 'STOKES', ''],
#            ['CRVAL3', 1, ''], 
#            ['CDELT3', 1, ''], 
#            ['CRPIX3', 1, ''], 
#            ['CUNIT3', ' ', ''], 

            # others
            ['SPECSYS', 'LSRK', ''], 

            # not sure what these are
            ['PC1_1', 1.0, ''], 
            ['PC2_1', 0.0, ''], 
            ['PC3_1', 0.0, ''], 

            ['PC1_2', 0.0, ''], 
            ['PC2_2', 1.0, ''], 
            ['PC3_2', 0.0, ''],
 
            ['PC1_3', 0.0, ''],
            ['PC2_3', 0.0, ''], 
            ['PC3_3', 1.0, ''],
            ]

        if hasattr(self, 'psfarg'):

            # sometimes bmaj depends on the wavelength
            # in which case, we just pick one
            try:
                bmaj = self.psfarg['bmaj'][0] / 3600
                bmin = self.psfarg['bmin'][0] / 3600
                bpa = self.psfarg['bpa'][0]
            except:
                bmaj = self.psfarg['bmaj'] / 3600
                bmin = self.psfarg['bmin'] / 3600
                bpa = self.psfarg['bpa']

            setvals.extend( [
                ['BMAJ', bmaj, ''],
                ['BMIN', bmin, ''],
                ['BPA', bpa, ''],
                ])

        if hasattr(self, 'restfreq'):
            setvals.append(['RESTFRQ', self.restfreq, ''])

        # now fill in the header
        for icombo in setvals:
            hdu.header.set(icombo[0], icombo[1], icombo[2])

        hdul = fits.HDUList([hdu])

        hdul.writeto(fname)

    # ==== plotting ====
    def plot(self, quant='I', iwav=0, ax=None, 
        scale_x=1., scale_y=1., **kwargs):
        """ plot the profile as a function laxis
        """
        prof = getattr(self, quant)

        if ax is None:
            ax = plt.gca()

        ax.plot(self.laxis / scale_x, prof[:,iwav] / scale_y, **kwargs)

    def plot_track(self, axis_unit='au', ax=None, **kwargs):
        """ plot the track of the cut
        """
        if ax is None:
            ax = plt.gca()

        # determine unit
        if axis_unit == 'cm':
            fac = 1. 
        elif axis_unit == 'au':
            fac = 1. / natconst.au
        elif axis_unit == 'arcsec':
            fac = 1. / natconst.au / self.dpc
        elif axis_unit == 'mas':
            fac = 1. / natconst.au / self.dpc * 1e3
        else:
            raise ValueError('axis_unit unknown: %s'%axis_unit)

        ax.plot(self.xpnt*fac, self.ypnt*fac, **kwargs)

# ====
# radial profile
# ====
class radialProfile(object):
    """ the radial profile from an image
    each object can only contain 1 quantity
    the quantity is a function of wavelength 
    """
    def __init__(self):
        pass

    def set_quantity(self, quant):
        self.quantity = quant

    def set_wavelength(self, w):
        self.w = w
        self.nw = len(self.w)

    def set_dpc(self, dpc):
        self.dpc = dpc

# ==== 
# image grid
# ====
class baseGrid(object):
    """ image grid object
    """
    def __init__(self):
        pass

    def set_frequency(self, f):
        self.f = f
#        self.nf = len(self.f) # don't use nf

    def get_frequency(self):
        self.f = natconst.cc * 1e4 / self.w
#        self.nf = len(self.f) # don't use nf

    def set_wavelength(self, w):
        self.w = w
        self.nw = len(self.w)

    def get_wavelength(self):
        self.w = natconst.cc * 1e4 / self.f
        self.nw = len(self.w)

    def set_dpc(self, dpc):
        """ distance in pc, optional
        """
        self.dpc = dpc

    def del_dpc(self):
        del self.dpc

    def set_restfreq(self, restfreq):
        """ set the rest frequency for a line
        """
        self.restfreq = restfreq

    def del_restfreq(self, restfreq):
        del self.restfreq

    def get_velocity(self):
        """ get the line of sight velocity for lines
        in cm/s
        """
        if hasattr(self, 'f') is False:
            self.get_freqeuncy()

        if hasattr(self, 'restfreq') is False:
            raise ValueError('restfreq must be known to get the velocity')

        self.v = natconst.cc * (self.restfreq - self.f) / self.restfreq

    def del_velocity(self):
        del self.v

    def rotate_in_sky(self, phi):
        """ rotate the grid in the sky 
        """
        x1 = self.xx * np.cos(phi) - self.yy * np.sin(phi)
        y1 = self.xx * np.sin(phi) + self.yy * np.cos(phi)
        self.xx, self.yy = x1, y1
        self.pp = np.arctan2(self.yy, self.xx)

class rectangularGrid(baseGrid):
    def __init__(self):
        baseGrid.__init__(self)

    def set_x(self, x):
        self.x = x
        self.nx = len(x)
        self.dx = x[1] - x[0]

    def set_y(self, y):
        self.y = y
        self.ny = len(y)
        self.dy = y[1] - y[0]

    def setup_from_file(self, data):
        """ set the spatial axis info from an radmc3d image
        x, y : 1d ndarray
            the original grid of the image
        xx, yy: 2d ndarray
            the spatial mesh grid in the sky
        """
        self.nx = data['nx']
        self.dx = data['dx']
        self.x = ((np.arange(self.nx, dtype=np.float64) + 0.5) - self.nx / 2) * self.dx

        self.ny = data['ny']
        self.dy = data['dy']
        self.y = ((np.arange(self.ny, dtype=np.float64) + 0.5) - self.ny / 2) * self.dy

        self.set_wavelength(data['wav'])
        self.get_frequency()

        # probably easier to do this from the start
        self.get_mesh()

    def setup_from_fits(self, hdr):
        """ setup the image grid based on the hdr
        since the grid is always in cgs units, use dpc to convert from arcsec
        Based on the definition of stokes, the horizontal axis is North and the vertical axis is East. 
        This means that naxis1 is actually the y-axis and naxis2 is the x-axis
        """
        if not hasattr(self, 'dpc'):
            raise ValueError('need dpc to set up from fits')

        # ==== x axis ====
        # the second axis is the DEC which will be the x-axis
        self.nx = hdr['naxis2']
        self.dx = hdr['cdelt2'] * 3600 * self.dpc * natconst.au
        self.x = (np.arange(self.nx) - (hdr['crpix2']-1) ) * self.dx
        self.xpixel = np.arange(self.nx)
        self.dec = (np.arange(self.nx) - (hdr['crpix2']-1)) * hdr['cdelt2'] +hdr['crval2']

        # ==== y axis ====
        # usually the first axis is the RA which will be our y-axis
        self.ny = hdr['naxis1']
        self.dy = hdr['cdelt1'] * 3600 * self.dpc * natconst.au
        self.y = (np.arange(self.ny) - (hdr['crpix1']-1) ) * self.dy
        self.ypixel = np.arange(self.ny)
        self.ra = (np.arange(self.ny) - (hdr['crpix1']-1)) * hdr['cdelt1'] + hdr['crval1']

        try:
            self.nw = hdr['naxis3']
            self.f = (np.arange(self.nw) - (hdr['crpix3'] - 1) ) * hdr['cdelt3']+ hdr['crval3']
            self.get_wavelength()
        except:
            self.nw = 1

        self.fits_hdr = hdr

    def get_pixel_area(self):
        """
        the pixel area corresponds to a solid angle, thus we need to know a distance
        The physical area of the image cell isn't as useful
        Attributes
        ----------
        pixel_area : float
            the solid angle in arcsec^2
        """
        self.pixel_area = (self.dx / self.dpc / natconst.pc / natconst.rad * 3600) * (self.dy / self.dpc / natconst.pc / natconst.rad * 3600)

    def recenter(self, x0, y0):
        """ reset the center
        x0, y0 : float 
            the center in the current coordinate system 
        """
        self.x -= x0
        self.y -= y0 

    def get_mesh(self):
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.rr = np.sqrt(self.xx**2 + self.yy**2)
        self.pp = np.arctan2(self.yy, self.xx)

class circularGrid(baseGrid):
    """ grid for the circular image
    """
    def __init__(self):
        baseGrid.__init__(self)

    def setup_from_file(self, data):
        self.r = data['rcell']
        self.nr = len(self.r)
        self.ri = data['rwall']

        self.phi = data['pcell']
        self.nphi = len(self.phi)
        self.phii = data['pwall']

        self.set_wavelength(data['wav'])

        self.get_frequency()

        self.get_mesh()

    def get_pixel_area(self):
        """
        Calculates the pixel area

        Returns
        -------
        The pixel area in arcsec^2
        """
        dr2 = np.pi * (self.ri[1:]**2 - self.ri[:-1]**2)
        dphi = self.phii[1:] - self.phii[:-1]

        self.pixel_area = dr2[:,None] * dphi[None,:] / 2. / np.pi * (3600. / self.dpc / natconst.pc / natconst.rad)**2

    def get_mesh(self):
        self.rr, self.pp = np.meshgrid(self.r, self.phi, indexing='ij')
        self.xx = self.rr * np.cos(self.pp)
        self.yy = self.rr * np.sin(self.pp)

# ====
# image classes
# ====
class baseImage(object):
    """
    the parent of all images. This will keep the intensity, tau surface, optical depth information, etc
    all image dimensions are in x by y by f

    tau3d :
        tau=? location in model coordinates
    """

    def __init__(self):
        self.quantity = [] # keep track of the physical quantity names

    def set_quantity(self, quantname, quant):
        """ sets a  3d quantity. the quantity can be anything
        as long as it's in nx by ny by nw
        Parameters
        ----------
        quant : 3d np.ndarray
            the array
        quantname : str
            name of the quantity
        """
        self.quantity.append(quantname)
        setattr(self, quantname, quant)

    def trim(self, par):
        """ take out a section of the image
        arguments in par
            xlim, ylim, flim
        """
        if isinstance(self.grid, rectangularGrid):
            if 'xlim' in par:
                # understand the direction
                if (self.grid.x[1] - self.grid.x[0]) > 0:
                    x_a = np.argmin(abs(min(par['xlim']) - self.grid.x))
                    x_b = np.argmin(abs(max(par['xlim']) - self.grid.x)) + 1
                else:
                    x_a = np.argmin(abs(max(par['xlim']) - self.grid.x))
                    x_b = np.argmin(abs(min(par['xlim']) - self.grid.x)) + 1

                self.grid.x = self.grid.x[x_a:x_b]
                self.grid.nx = len(self.grid.x)
            else:
                x_a = 0
                x_b = self.grid.nx

            if 'ylim' in par:
                # understand the direction
                if self.grid.y[1] - self.grid.y[0] > 0:
                    y_a = np.argmin(abs(min(par['ylim']) - self.grid.y))
                    y_b = np.argmin(abs(max(par['ylim']) - self.grid.y)) + 1
                else:
                    y_a = np.argmin(abs(max(par['ylim']) - self.grid.y))
                    y_b = np.argmin(abs(min(par['ylim']) - self.grid.y)) + 1

                # trim
                self.grid.y = self.grid.y[y_a:y_b]
                self.grid.ny = len(self.grid.y)
            else:
                y_a = 0
                y_b = self.grid.nx

        elif isinstance(self.grid, circularGrid):
            raise ValueError('not done yet')

            rreg = (min(par['rlim']) <= self.grid.r) & (self.grid.r <= max(par['rlim']))
            self.grid.r = self.grid.r[rreg]
            preg = (min(par['plim']) <= self.grid.phi) & (self.grid.phi <= max(par['plim']))
            self.grid.phi = self.grid.phi[preg]
            for q in self.quantity:
                dum = getattr(self, q)[rreg,:,:]
                setattr(self, q, dum[:,preg,:])
        else:
            raise ValueError('grid unknown')

        # trim the wavelength axis
        if 'wlim' in par:
            raise ValueError('not done yet')

        elif 'vlim' in par:
            # understand the direction
            if (self.grid.v[1] - self.grid.v[0]) > 0:
                w_a = np.argmin(abs(min(par['vlim']) - self.grid.v))
                w_b = np.argmin(abs(max(par['vlim']) - self.grid.v)) + 1
            else:
                w_a = np.argmin(abs(max(par['vlim']) - self.grid.v))
                w_b = np.argmin(abs(min(par['vlim']) - self.grid.v)) + 1

            self.grid.w = self.grid.w[w_a:w_b]
            self.grid.v = self.grid.v[w_a:w_b]
            self.grid.nw = len(self.grid.w)

            # recalculate the frequency
            self.grid.get_frequency()
        else:
            w_a = 0
            w_b = self.grid.nw

        # trim the image
        for q in self.quantity:
            setattr(self, q, getattr(self, q)[x_a:x_b,y_a:y_b,w_a:w_b])

    def get_peak_loc(self, quant='I', iwav=0):
        """ get the location of the peak 
        """
        if quant not in self.quantity:
            raise ValueError('quant unknown: %s'%quant)

        im = getattr(self, quant)[:,:,iwav]

        inx = np.unravel_index(np.nanargmax(im), im.shape)

        if isinstance(self.grid, rectangularGrid):
            return self.grid.x[inx[0]], self.grid.y[inx[1]]
        else:
            raise ValueError('not done yet')

    def get_fn_quant_w(self, quant='I', iwav=0, **kwargs):
        """ returns the interp2d object for just one wavelength
        Given the number of pixel required for one image (~200x200), it's typically impossible to use interp2d and use the meshgrid of x and y. 
        We have to stick to interpolation on a regular grid
        """
        if quant not in self.quantity:
            raise ValueError('the property to be interpolated does not exist: %s'%quant)

        if isinstance(self.grid, rectangularGrid):
#            fn = interp2d(self.grid.x, self.grid.y, getattr(self, quant)[...,iwav].T, **kwargs)
             fn = RectBivariateSpline(self.grid.x, self.grid.y, getattr(self, quant)[...,iwav], **kwargs)
        elif isinstance(self.grid, circularGrid):
#            fn = interp2d(self.grid.xx, self.grid.yy, getattr(self, quant)[...,iwav], **kwargs)

            # wrap around phi for both the beginning and end
            phi = np.zeros([len(self.grid.phi)+2])
            phi[1:-1] = self.grid.phi
            phi[0] = self.grid.phi[-1] - 2 * np.pi 
            phi[-1] = self.grid.phi[0] + 2 * np.pi

            dat = np.zeros([len(self.grid.r), len(phi)])
            dat[:,1:-1] = getattr(self, quant)[...,iwav] * 1
            dat[:,0] = dat[:,-2] * 1
            dat[:,-1] = dat[:,1] * 1

            fn = RectBivariateSpline(self.grid.r, phi, dat, **kwargs)

        else:
            raise ValueError('unknown image grid')

        return fn 

    def get_fn_quant(self, quant='I', **kwargs):
        """ prepare interp2d object for each wavelength and keep as attribute
        """
        if quant not in self.quantity:
            raise ValueError('the property to be interpolated does not exist: %s'%quant)
        
        ff = []
        for ii in range(self.grid.nw):
            fn = self.get_fn_quant_w(quant=quant, iwav=ii, **kwargs)
            ff.append(fn)

        setattr(self, 'fn_%s'%quant, ff)

    def cut(self, laxis, quant=['I'], track='linear', 
        trackkw={'x0':0, 'y0':0, 'theta':0},
        mode='fn', 
        ):
        """ 
        interpolate a cut across some location 
        Parameters
        ----------
        quant : list of str
            quantity to be sliced
        laxis : 1d ndarray
            some parameter

        track : str
            linear = a linear cut
            ellipse = an elliptical cut

        trackkw : dict
            keyword arguments for the track of the cut 

        mode : str
            'fn' = use the fn_quant method to iterate each wavelength
            'lin' = search for nearest indices and linear interpolate. easier for interpolating throughout wavelength at once

        Returns
        -------
        imageCut object 
        """
        # ==== calculate path ====
        # always in cm 
        if track == 'input':
            # use arbitrary input
            xpnt = trackkw['x']
            ypnt = trackkw['y']
            # must be the same length as laxis
            if (len(laxis) != len(xpnt)) | (len(laxis) != len(ypnt)):
                raise ValueError("length of x and y must equal laxis when track = 'input'")
        elif track == 'linear':
            # laxis is the length
            xpnt = trackkw['x0'] + laxis * np.cos(trackkw['theta'])
            ypnt = trackkw['y0'] + laxis * np.sin(trackkw['theta'])

        elif track == 'ellipse':
            # laxis is the phi
            xpnt = trackkw['r'] * np.cos(laxis)
            ypnt = trackkw['r'] * np.sin(laxis)

            # rotate by inclination 
            ypnt, zpnt = rot_inc(trackkw['inc'], ypnt, 0)

            # rotate by some angle
            xpnt, ypnt = rot_eon(trackkw['ang'], xpnt, ypnt)

            xpnt += trackkw['x0']
            ypnt += trackkw['y0']

        else:
            raise ValueError('track unknown')

        # the actual points depend on the shape of the grid
        if isinstance(self.grid, rectangularGrid):
            v1 = xpnt
            v2 = ypnt
        elif isinstance(self.grid, circularGrid):
            rpnt = np.sqrt(xpnt**2 + ypnt**2)
            ppnt = np.arctan2(ypnt, xpnt)
            reg = ppnt < 0
            ppnt[reg] += 2 * np.pi 

            v1 = rpnt
            v2 = ppnt
        else:
            raise ValueError('grid shape unknown')

        # prepare the imageCut object
        cut = imageCut()
        cut.set_laxis(laxis)
        cut.set_xy(xpnt, ypnt)
        cut.set_wavelength(self.grid.w)

        # ==== interpolation ====
        if mode == 'fn':
            for iquant in quant:
                # check if there is already an interpolation object
                # setup the interpolation if it doesn't have one
                if not hasattr(self, 'fn_%s'%iquant):
                    self.get_fn_quant(quant=iquant)
 
                fn = getattr(self, 'fn_%s'%iquant)

                # interpolate
                prof = np.zeros([len(laxis), self.grid.nw])
                for i in range(self.grid.nw):
                    prof[:,i] = fn[i](v1, v2, grid=False)

                cut.set_quantity(iquant, prof)

                del fn, prof

        elif mode == 'lin':
            # find nearest index and interpolate
            # it seems like the fn method is pretty fast already
            pass
        else:
            raise ValueError('cut mode unknown: %s'%mode)

        # some additional attributes
        if hasattr(self.grid, 'dpc'):
            cut.set_dpc(self.grid.dpc)

        if hasattr(self.grid, 'restfreq'):
            cut.set_restfreq(self.grid.restfreq)
            cut.get_velocity()

        return cut

    def cut_slit(self, laxis, quant=['I'], width=1, 
        trackkw={'x0':0, 'y0':0, 'theta':0}):
        """ instead of interpolating, we use a slit and average the pixels perpendicular to the cut. 
        The slit can only be linear

        Parameters
        ----------
        laxis : 1d ndarray
            the location in the slit direction in cm
        width : int
            The number of pixels. Default is 1
        """
        # check if it's Rectangular grid
        if isinstance(self.grid, rectangularGrid) is False:
            raise ValueError('grid must be a rectangular grid')

        # the angle in the coordinates of the image
        angle = trackkw['theta']

        # unit direction of the slit
        vec =  (np.cos(angle), np.sin(angle))

        # the direction perpendicular to the slit
        per = (np.cos(angle + np.pi/2), np.sin(angle + np.pi/2))

        # the pixel length in the direction perpendicular to the slit
        if (-1 <= np.tan(angle+np.pi/2)) & (np.tan(angle+np.pi/2) <= 1):
            dp = self.grid.dx / np.cos(angle+np.pi/2)
        else:
            dp = self.grid.dy / np.sin(angle+np.pi/2)
        dp = abs(dp)

        # the different centers
        if np.mod(width, 2) == 1:
            offset = np.arange(-(width//2), width//2 + 1)
        else:
            offset = np.arange(- (width//2), width//2) + 0.5
        xcen = trackkw['x0'] + offset * per[0] * dp
        ycen = trackkw['y0'] + offset * per[1] * dp

        # ==== begin interpolation ====
        profs = []
        for iquant in quant:
            # interpolate along different lines
            prof = np.zeros([len(laxis), len(self.grid.f), width])
            for i in range(width):
                # the coordinates for this line
                xpnt = xcen[i] + laxis * vec[0]
                ypnt = ycen[i] + laxis * vec[1]

                # convert to pixel coordinates
                x = np.interp(xpnt, self.grid.x, np.arange(self.grid.nx))
                y = np.interp(ypnt, self.grid.y, np.arange(self.grid.ny))
                z = np.arange(self.grid.nw, dtype=int)

                # calculate the coordinate grid
#                xi, dum = np.meshgrid(x, z, indexing='ij')
#                yi, dum = np.meshgrid(y, z, indexing='ij')
#                dum, zi = np.meshgrid(np.ones(len(x)), z, indexing='ij')
#                prof[:,:,i] = map_coordinates(getattr(self, iquant), [xi,yi,zi], order=0)

# need to test this
#                xi, yi, zi = np.meshgrid(x, y, z, indexing='ij')
#                prof[:,:,i] = map_coordinates(getattr(self, iquant), [xi,yi,zi], order=0)
            profs.append( np.mean(prof, axis=2) )
 
        # prepare the imageCut object
        cut = imageCut()
        cut.set_laxis(laxis)
        cut.set_xy(trackkw['x0'] + laxis*vec[0], trackkw['y0'] + laxis*vec[1])
        cut.set_frequency(self.grid.f)
        cut.get_wavelength()
        cut.stokes_unit = self.stokes_unit

        for i, iquant in enumerate(quant):
            cut.set_quantity(iquant, profs[i])

        # some additional attributes
        if hasattr(self.grid, 'dpc'):
            cut.set_dpc(self.grid.dpc)

        if hasattr(self.grid, 'restfreq'):
            cut.set_restfreq(self.grid.restfreq)
            cut.get_velocity()

        if hasattr(self, 'rms'):
            cut.rms = self.rms

        return cut

    # ==== plotting ====
    def plot(self, quant='I', iwav=0,
            fillmode='contourf', levels=10, 
            scale_axis=1., scale_map=1., relmax=False, 
            axis_unit='au', 
            ax=None, **kwargs):
        """ plot the 2d image in rectangular image (x,y)
        Parameters
        ----------
        """
        if ax is None:
            ax = plt.gca()

        # determine the physical quantity
        if quant not in self.quantity:
            raise ValueError('quantity unknown: %s'%quant)

        im2d = getattr(self, quant)[...,iwav] / scale_map

        if relmax:
            im2d = im2d / np.nanmax(im2d)

        # coordinate units
        if axis_unit == 'cm':
            unitfac = 1.
        elif axis_unit == 'au':
            unitfac = natconst.au
        elif axis_unit == 'arcsec':
            unitfac = natconst.au * self.grid.dpc
        elif axis_unit == 'mas':
            unitfac = natconst.au * self.grid.dpc * 1e-3
        else:
            raise ValueError('axis_unit unknown')

        if hasattr(self.grid, 'xx') is False:
            self.grid.get_mesh()

        xx = self.grid.xx / unitfac / scale_axis
        yy = self.grid.yy / unitfac / scale_axis

        # if the image is circular, then wrap around the last phi
        if isinstance(self.grid, circularGrid):
            xx = np.concatenate((xx, xx[:,0][:,None]), axis=1)
            yy = np.concatenate((yy, yy[:,0][:,None]), axis=1)
            im2d = np.concatenate((im2d, im2d[:,0][:,None]), axis=1)

        """
        # plot based on grid shape
        if isinstance(self.grid, rectangularGrid):
            pc = ax.pcolormesh(self.grid.x / unitfac / scale_axis, self.grid.y / unitfac / scale_axis, im2d.T, **kwargs)
        elif isinstance(self.grid, circularGrid):
            pc = ax.pcolormesh(self.grid.xx / unitfac / scale_axis, self.grid.yy / unitfac / scale_axis, im2d, **kwargs)
        else:
            raise ValueError('image grid unknown')

        """ 
        if fillmode == 'contour':
            pc = ax.contour(xx, yy, im2d, levels, **kwargs)
        elif fillmode == 'pcolormesh':
            pc = ax.pcolormesh(xx, yy, im2d, **kwargs)
        elif fillmode == 'contourf':
            pc = ax.contourf(xx, yy, im2d, levels, **kwargs)
        else:
            raise ValueError('fillmode unknown: %s'%fillmode)

#        cbar = plt.colorbar(pc, ax=ax)

        return {'pc':pc}

    def plot_rphi(self, quant='I', iwav=0,
        scale_r=1., scale_map=1.,
        ax=None, **kwargs):
        """ plot the 2d image in circular coordinates (r, phi)
        """
        if ax is None:
            ax = plt.gca()

        if quant not in self.quantity:
            raise ValueError('quantity unknown: %s'%quant)

        im2d = getattr(self, quant)[...,iwav] / scale_map

        if isinstance(self.grid, rectangularGrid):
            pc = ax.pcolormesh(
                self.grid.rr / scale_r,
                self.grid.pp /scale_phi,
                im2d, **kwargs)
        elif isinstance(self.grid, circularGrid):
            pc = ax.pcolormesh(
                self.grid.r / scale_r,
                self.grid.phi /scale_phi,
                im2d, **kwargs)
        else:
            raise ValueError('image grid unknown')

        cbar = plt.colorbar(pc, ax=ax)

        return {'pc':pc, 'cbar':cbar}

class intensity(baseImage):
    """ the intensity image with some unit 
    I
    Q
    U
    V
    tb : brighntess temperature

    lpi : ndarray, optional
        linear polarized intensity

    lpol : ndarray, optional 
        linear polarization fraction 

    lpa : ndarray, optional
        linear polarization angle

    cpol : ndarray, optional 
        circular polarization fraction 

    pol : ndarray, optional 
        polarization fraction 
    """
    def __init__(self):
        baseImage.__init__(self)

    def read(self, fname, binary=False, shape='rect'):
        """ read the model image
        shape : str
            'rect' = rectangular
            'circ' = circular
        """
        # use the basic functions to read in data from radmc3d
        if shape == 'rect':
            out = read_rectangular_image(fname, binary=binary)
            self.grid = rectangularGrid()
        elif shape == 'circ':
            out = read_circular_image(fname, binary=binary)
            self.grid = circularGrid()
        else:
            raise ValueError('image shape unknown')

        # setup the grid
        self.grid.setup_from_file(out)

        # assign stokes values based on the image shape 
        if len(out['image'].shape) == 3:
            self.set_stokes(out['image'])
            self.set_stokes_unit('cgs')

        elif len(out['image'].shape) == 4:
            self.set_stokes(
                out['image'][...,0], 
                Q = out['image'][...,1], 
                U = out['image'][...,2], 
                V = out['image'][...,3]
            )
            self.set_stokes_unit('cgs')
        else:
            raise ValueError('image dimension unknown')

    def read_fits(self, fname, dpc, stokes_unit='jypbeam'):
        """ read fits image
        the fits data can come in a variety...
        but some common features are
        - these at least depend on x,y
        - 
        """
        hdul = fits.open(fname)
        hdr = hdul[0].header
        dat = hdul[0].data

        # setup the grid
        self.grid = rectangularGrid()
        self.grid.set_dpc(dpc)
        self.grid.setup_from_fits(hdr)

        # usually, ra is decreasing
        # change the ordering so that it is increasing
        if self.grid.dy < 0:
            dat = dat[...,::-1]
            self.grid.dy *= -1
#            self.grid.y *= -1
            self.grid.y = self.grid.y[::-1]
            self.grid.ra = self.grid.ra[::-1]
            self.grid.ypixel = self.grid.ypixel[::-1]

        dim = dat.shape
        if len(dim) == 2:
            self.set_stokes(dat[:,:,None])
        elif len(dim) == 3:
            self.set_stokes(np.moveaxis(dat, 0, -1))
        elif len(dim) == 4:
            # move the frequency to the last
            dat = np.moveaxis(dat, 1, -1)

            if dim[0] == 1:
                self.set_stokes(
                    dat[0,:,:,:],
                )
            elif dim[0] == 3:
                self.set_stokes(
                    dat[0,:,:,:],
                    Q=dat[1,:,:,:],
                    U=dat[2,:,:,:],
                )
            elif dim[0] == 4:
                self.set_stokes(
                    dat[0,:,:,:],
                    Q=dat[1,:,:,:],
                    U=dat[2,:,:,:],
                    V=dat[3,:,:,:],
                )
            else:
                raise ValueError('the number of dimensions for stokes is unknown')
        else:
            raise ValueError('dimension unknown')

        self.set_stokes_unit(stokes_unit)

        # usually only one set of bmaj, bmin, bpa per fits file
        if stokes_unit == 'jypbeam':
            self.psfarg = {
                'bmaj':hdr['bmaj'] * 3600,
                'bmin':hdr['bmin'] * 3600, 
                'bpa':hdr['bpa'],
                'type':'gaussian', 
                }

        hdul.close()

    def set_stokes_unit(self, stokes_unit):
        """ determine the unit of the stokes value
        """
        if stokes_unit not in ['cgs', 'jyppix', 'jypbeam']:
            raise ValueError('stokes_unit unknown: %s'%stokes_unit)
        self.stokes_unit = stokes_unit

    def set_stokes(self, I, Q=None, U=None, V=None):
        """ setup the stokes values
        """
        self.I = I
        self.quantity.append('I')

        if Q is not None:
            self.Q = Q
            self.quantity.append('Q')
        if U is not None:
            self.U = U
            self.quantity.append('U')
        if V is not None:
            self.V = V
            self.quantity.append('V')

        if (Q is not None) | (U is not None) | (V is not None):
            self.isStokes = True
        else:
            self.isStokes = False

    def set_rms(self, I_rms, Q_rms=None, U_rms=None, V_rms=None):
        """ set the rms value which is useful for observational data
        Parameters
        ----------
        I_rms : float or 1d ndarray 
            rms as a function of wavelength. in the same units as stokes. 
            if it's a float, this will be propagated to the length of wavlength
        """
        if (len(I_rms) != 1 ) | (len(I_rms) != self.grid.nw):
            raise ValueError('length of I_rms should be 1 or the same as wavelength')
        self.I_rms = I_rms

        if Q_rms is not None:
            self.Q_rms = Q_rms
        if U_rms is not None:
            self.U_rms = U_rms
        if V_rms is not None:
            self.V_rms = V_rms

    def del_rms(self):
        del self.I_rms, self.Q_rms, self.U_rms, self.V_rms

    def get_polarization(self, thres=3, debias=False):
        """ calculate polarization related properties
        """
        # check if the image has polarization
        if self.isStokes is False:
            return

        # linear polarized intensity
        self.lpi = np.sqrt(self.Q**2 + self.U**2)
        if debias:
            sigma = np.sqrt(self.Q_rms**2 + self.U_rms**2)
            reg = self.lpi < sigma[None,None,:]
            self.lpi = np.sqrt(self.Q**2 + self.U**2 - sigma[None,None,:]**2)
            self.lpi[reg] = 0

        # total polarized intensity 
        self.pi = np.sqrt(self.Q**2 + self.U**2 + self.V**2)

        # polarization angle
        self.lpa = calc_lpa(self.Q, self.U)

        # flag out regions where we need to divide by small I
        reg = self.I > 0
        if hasattr(self, 'I_rms'):
            reg = self.I >= self.I_rms[None,None,:] * thres

        self.lpa[~reg] = 0.

        # linear polarization fraction
        self.lpol = np.zeros_like(self.I)
        self.lpol[reg] = self.lpi[reg] / self.I[reg]

        # total polarization fraction 
        self.pol = np.zeros_like(self.I)
        self.pol[reg] = self.pi[reg] / self.I[reg]

        # circular polarization
        self.cpol = np.zeros_like(self.I)
        self.cpol[reg] = self.V[reg] / self.I[reg]

        for iname in ['lpi', 'pi', 'lpa', 'lpol', 'pol', 'cpol']:
            self.quantity.append(iname)

    def del_polarization(self):
        """ delete polarization related properties
        """
        for iname in ['lpi', 'pi', 'lpa', 'lpol', 'pol']:
            delattr(self, iname)
            self.quantity.remove(iname)

    def get_tb(self):
        """ calculate brightness temperature. always in Kelvin 
        """
        self.quantity.append('tb')

        self.tb = np.zeros_like(self.I)
        for i in range(len(self.grid.w)):
            if self.stokes_unit == 'cgs':
                self.tb[...,i] = cgs_to_tb(self.I[...,i], self.grid.w[i])

            elif self.stokes_unit == 'jyppix':
                self.tb[...,i] = jyppix_to_tb(
                    self.I[...,i], self.grid.pixel_area, self.grid.w[i]
                )

            elif self.stokes_unit == 'jypbeam':
                try:
                    bmaj = self.psfarg['bmaj'][i]
                    bmin = self.psfarg['bmin'][i]
                except:
                    bmaj = self.psfarg['bmaj']
                    bmin = self.psfarg['bmin']
                self.tb[...,i] = jypbeam_to_tb(
                    self.I[...,i], bmaj, bmin, self.grid.w[i]
                )
            else:
                raise ValueError('Internal error. stokes_unit unknown: %s'%self.stokes_unit)

    def del_tb(self):
        del self.tb
        self.quantity.remove('tb')
                
    def convert_stokes_unit(self, newunit):
        """ convert the unit for the stokes parameters
        it doesn't do anything if the current unit and the new unit are the same
        newunit : str
        """
        # check if the new unit is the same as the current one
        if self.stokes_unit == newunit:
            return

        # ==== check for necessary arguments ====
        # jyppix
        if (self.stokes_unit == 'jyppix') | (newunit == 'jyppix'):
            if hasattr(self.grid, 'dpc') is False:
                raise ValueError('dpc unknown')

            if hasattr(self.grid, 'pixel_area') is False:
                self.grid.get_pixel_area()

        # jyppbeam
        if (self.stokes_unit == 'jypbeam') | (newunit == 'jypbeam'):
            if hasattr(self, 'psfarg') is False:
                raise ValueError('psfarg unknown')

        # ==== convert the stokes units ====
        if self.stokes_unit == 'cgs':
            if newunit == 'jyppix':
                fac = self.grid.pixel_area * (natconst.rad / 3600)**2 / natconst.jy
                if type(fac) == np.ndarray:
                    fac = fac[:,:,None]

                for iquant in ['I','Q','U', 'V', 'pi']:
                    if iquant in self.quantity:
                        setattr(self, iquant, getattr(self, iquant) * fac)

            elif newunit == 'jypbeam':
                # since beam is wavelength dependent
                fac = (self.psfarg['bmaj'] / 3600*natconst.rad) * (self.psfarg['bmin'] / 3600* natconst.rad) * np.pi/4./np.log(2.) / natconst.jy

                for iquant in ['I','Q','U', 'V', 'pi']:
                    if iquant in self.quantity:
                        setattr(self, iquant, getattr(self, iquant) * fac)
            else:
                raise ValueError('newunit unknown: %s'%newunit)

        elif self.stokes_unit == 'jyppix':
            if newunit == 'cgs':
                fac = self.grid.pixel_area * (natconst.rad / 3600)**2 / natconst.jy
                if type(fac) == np.ndarray:
                    fac = fac[:,:,None]

                for iquant in ['I','Q','U', 'V', 'pi']:
                    if iquant in self.quantity:
                        setattr(self, iquant, getattr(self, iquant) / fac)
            elif newunit == 'jypbeam':
                raise ValueError('not done yet')
            else:
                raise ValueError('newunit unknown: %s'%newunit)

        elif self.stokes_unit == 'jypbeam':
            if newunit == 'cgs':
                # since beam is wavelength dependent
                fac = (self.psfarg['bmaj'] / 3600*natconst.rad) * (self.psfarg['bmin'] / 3600*natconst.rad) * np.pi/4./np.log(2.) / natconst.jy
                for iquant in ['I','Q','U', 'V', 'pi']:
                    if iquant in self.quantity:
                        setattr(self, iquant, getattr(self, iquant) / fac[None,None,:] )
            elif newunit ==  'jyppix':
                raise ValueError('not done yet')
            else:
                raise ValueError('newunit unknown: %s'%newunit)
        else:
            raise ValueError('inner inconsistency. stokes_unit unknown: %s'%self.stokes_unit)

        self.set_stokes_unit(newunit)

    def get_flux_density(self):
        """ 
        calculate the flux density as a function of wavelength 
        recall that it's a flux since we integrate over the solid angle and the "density" refers to the spectral density
        This requires us to place the object at a certain distance
        express the flux density in Jy
        """
        if hasattr(self.grid, 'pixel_area') is False:
            self.grid.get_pixel_area()

        if self.stokes_unit == 'cgs':
            self.flux_density = np.sum(self.I * self.grid.pixel_area, axis=(0,1)) * (natconst.rad / 3600.)**2 / natconst.jy
        elif self.stokes_unit == 'jyppix':
            self.flux_density = np.sum(self.I, axis=(0,1))
        elif self.stokes_unit == 'jypbeam':
            beam_area = np.pi * self.psfarg['bmaj'] * self.psfarg['bmin'] / 4. / np.log(2)
            self.flux_density = np.sum(self.I * self.grid.pixel_area, axis=(0,1)) / beam_area
        else:
            raise ValueError('stokes unit unknown')

    def set_psf(self, psf, psfarg):
        """ save the psf and keep a record of some basic arguments
        psf : ndarray. x by y by w
        psfarg : dict
            type : str
                'gaussian' : see convolve_gaussian()
        """
        self.psf = psf
        self.psfarg = psfarg
        self.quantity.append('psf')

    def del_psf(self):
        delattr(self, 'psf')
        delattr(self, 'psfarg')
        self.quantity.remove('psf')

    def convolve_gaussian(self, bmaj, bmin, bpa):
        """ convolve the image with a gaussian beam 
        Parameters
        ----------
        bmaj : float or list
            beam major axis in arcsec. the length of list should correspond to the number of wavelengths. If only a float, then bmaj will be the same for all the frequencies
        bmin : float or list 
        bpa : float or list
            in degrees

        Returns
        -------
        a copy of itself, but the stokes I (Q,U,V, if applicable) are replaced.
        The polarization properties and brightness temperature do not exist

        Additional Attributes
        ---------------------
        psf : 3d ndarray
        psfarg : dict
            bmaj : 1d ndarray
                beam major axis in arcsec
            bmin : 1d ndarray
                beam minor axis in arcsec
            bpa : 1d ndarray
                beam position angle in degrees 
        """
        # if bmaj, bmin, or bpa are lists, check if it's the same size as wavelength
        for ival in [bmaj, bmin, bpa]:
            if (type(ival) is list) | (type(ival) is np.ndarray):
                if len(ival) != self.grid.nw:
                    raise ValueError('the number of elements should be 1 or match the number of wavelengths')

        if isinstance(self.grid, rectangularGrid) is False:
            raise ValueError('currently only rectangular grid works')

        # check if dpc is set
        if hasattr(self.grid, 'dpc') is False:
            raise ValueError('dpc must be set')

        # ==== start calculation ====
        # create a new image
        out = intensity()

        # keep the grid, dpc
        out.grid = copy.deepcopy(self.grid)
        out.set_stokes_unit(self.stokes_unit)

        # pixel size in arcseconds
        dx = self.grid.dx / natconst.au / self.grid.dpc
        dy = self.grid.dy / natconst.au / self.grid.dpc

        cI = np.zeros_like(self.I)
        if self.isStokes:
            cQ = np.zeros_like(cI)
            cU = np.zeros_like(cI)
            cV = np.zeros_like(cI)

        psf = np.zeros_like(self.I)
        psfarg = {'type':'gaussian'}
        psfarg['bmaj'] = np.zeros(self.grid.nw)
        psfarg['bmin'] = np.zeros_like(psfarg['bmaj'])
        psfarg['bpa'] = np.zeros_like(psfarg['bmaj'])
        for i in range(self.grid.nw):
            if (type(bmaj) is list) | (type(bmaj) is np.ndarray):
                bmaj_in = bmaj[i]
            else:
                bmaj_in = bmaj

            if (type(bmaj) is list) | (type(bmaj) is np.ndarray):
                bmin_in = bmin[i]
            else:
                bmin_in = bmin

            if (type(bpa) is list) | (type(bpa) is np.ndarray):
                bpa_in = bpa[i]
            else:
                bpa_in = bpa

            psf[:,:,i] = get_gaussian_psf(self.grid.nx, self.grid.ny, 
                bmaj_in, bmin_in, bpa_in, pscale=[dx,dy])
            psfarg['bmaj'][i] = bmaj_in
            psfarg['bmin'][i] = bmin_in
            psfarg['bpa'][i] = bpa_in

            cI[:,:,i] = convolve_image_psf(self.I[:,:,i], psf[:,:,i])
            if self.isStokes:
                for istoke, iname in zip([cQ, cU, cV], 
                                         ['Q','U','V']):
                    istoke[:,:,i] = convolve_image_psf( 
                        getattr(self, iname)[:,:,i], psf[:,:,i])

        if self.isStokes:
            out.set_stokes(cI, Q=cQ, U=cU, V=cV)
        else:
            out.set_stokes(cI)

        out.set_psf(psf, psfarg)

        return out

    def rotate_in_sky(self, phi):
        """ change the properties that change depending on the 
        orientation in the sky. this completely replaces the previous values
        phi : float
            the angle in radians. counter clockwise from the x-axis
        """
        # change the grid
        self.grid.rotate_in_sky(phi)

        # change stokes Q, U, lpa
        if self.isStokes:
            eta = phi
            q = np.cos(2*eta) * self.Q - np.sin(2*eta) * self.U
            u = np.sin(2*eta) * self.Q + np.cos(2*eta) * self.U
            self.Q = q
            self.U = u
            self.lpa = calc_lpa(self.Q, self.U)

        # change the beam pa
        if hasattr(self, 'psfarg'):
            if self.psfarg['type'] == 'gaussian':
                # the beam is east-of-north
                self.psfarg['bpa'] = self.psfarg['bpa'] - phi / natconst.rad

    def fold_symmetry(self, diskrot, fold='xy', cellsize=None, xmax=None, ymax=None):
        """ sometimes, an image can have symmetries
        averge between left-right and top-bottom 
        Produces a new intensity object since the grid is completely new
        The image should be in the original frame and not in the rotated frame, since it's easier for interpolation 
        Parameters
        ----------
        fold : str
            x = fold across x meaning x is held fixed and fold the paper over
            y = fold across y
            xy = fold across both x and y
        """
        if isinstance(self.grid, circularGrid):
            raise ValueError('circular grid not allowed for the moment')

        # determine the cellsize
        if cellsize is None:
            cellsize = abs(self.grid.x[1] - self.grid.x[0])

        # determine the maximum in x
        if xmax is None:
            xmax = self.grid.x.max()
        if ymax is None:
            ymax = self.grid.y.max()

        # determine the stokes for averaging
        # we need the stokes in the rotated frame
        I = self.I * 1.
        image_set = [I]
        if self.isStokes:
            ang = diskrot
            Q = np.cos(2*ang) * self.Q + np.sin(2*ang) * self.U
            U = - np.sin(2*ang) * self.Q + np.cos(2*ang) * self.U
            V = self.V * 1.
            image_set = [I, Q, U, V]

        # setup the grid for the eventual folded image
#        x1 = np.arange(0.5*cellsize, xmax, cellsize)
#        y1 = np.arange(0.5*cellsize, ymax, cellsize)
        x1 = np.arange(0, xmax, cellsize)
        y1 = np.arange(0, ymax, cellsize)

        x = np.concatenate((-x1[::-1], x1[1:]))
        y = np.concatenate((-y1[::-1], y1[1:]))
        xx, yy = np.meshgrid(x, y, indexing='ij')

        fI = np.zeros([len(x), len(y)])
        if self.isStokes:
            fQ = np.zeros_like(fI)
            fU = np.zeros_like(fI)
            fV = np.zeros_like(fI)

        # calculate the coordinates in the modified sky coordinates
        # which means the x-axis is in decreasing RA
        xs = np.cos(diskrot) * xx - np.sin(diskrot) * yy
        ys = np.sin(diskrot) * xx + np.cos(diskrot) * yy

        # interpolate the stokes parameters
        fn = []
        for im2d in image_set:
#            fn.append(interp2d(self.x, self.y, im2d.T, kind='cubic'))
             fn.append(RectBivariateSpline(self.grid.x, self.grid.y, im2d))

        for i in range(len(x)):
            fI[i,:] = fn[0](xs[i,:], ys[i,:], grid=False)
            if self.isStokes:
                fQ[i,:] = fn[1](xs[i,:], ys[i,:], grid=False)
                fU[i,:] = fn[2](xs[i,:], ys[i,:], grid=False)
                fV[i,:] = fn[3](xs[i,:], ys[i,:], grid=False)

#            for j in range(len(y)):
#                fI[i,j] = fn[0](xs[i,j], ys[i,j])
#                fQ[i,j] = fn[1](xs[i,j], ys[i,j])
#                fU[i,j] = fn[2](xs[i,j], ys[i,j])
#                fV[i,j] = fn[3](xs[i,j], ys[i,j])

        # keep the original one without folding first
        I0 = fI * 1.
        if self.isStokes:
            Q0 = fQ * 1.
            U0 = fU * 1.
            V0 = fV * 1.

        snr_improve = 1.
        # fold left-right (fold across y)
        if 'y' in fold:
            fI = 0.5 * (fI + fI[::-1,:])
            if self.isStokes:
                fQ = 0.5 * (fQ + fQ[::-1,:])
                fU = 0.5 * (fU - fU[::-1,:])
                fV = 0.5 * (fV - fV[::-1,:])
            snr_improve *= np.sqrt(2.)

        # fold top-bottom
        if 'x' in fold:
            fI = 0.5 * (fI + fI[:,::-1])
            if self.isStokes:
                fQ = 0.5 * (fQ + fQ[:,::-1])
                fU = 0.5 * (fU - fU[:,::-1])
                fV = 0.5 * (fV - fV[:,::-1])
            snr_improve *= np.sqrt(2.)

        # prepare the intensity object
        # this is already rotated
        out = intensity()
        out.grid = rectangularGrid()
        out.grid.x = x
        out.grid.nx = len(out.grid.x)
        out.grid.y = y
        out.grid.ny = len(out.grid.y)
        out.grid.xx = xx
        out.grid.yy = yy
        out.grid.dpc = self.grid.dpc
        out.grid.w = self.grid.w

        out.I = fI
        out.quantity.append('I')
        self.isStokes = False
        if self.isStokes:
            out.Q = fQ
            out.U = fU
            out.V = fV
            out.quantity.extend(['Q', 'U', 'V'])
            out.isStokes = True

#        out.I0 = I0
#        out.Q0 = Q0
#        out.U0 = U0
#        out.V0 = V0
#        out.quantity.extend(['I0', 'Q0', 'U0', 'V0'])

        # determine the beam
#        out.psfarg = self.psfarg.copy()
#        out.psfarg['bpa'] = self.psfarg['bpa'] - diskrot / rad
        bfwhm = np.sqrt(self.psfarg['bmaj'] * self.psfarg['bmaj'])
        out.psfarg = {'bmaj':bfwhm, 'bmin':bfwhm, 'bpa':0, 'type':'gaussian'}

        # determine the noise
        for ikey in ['I_rms', 'Q_rms', 'U_rms', 'V_rms']:
            if hasattr(self, ikey):
                setattr(out, ikey, getattr(self, ikey) / snr_improve)

        return out

    def average_channel(self, navg):
        """ average the images across frequency
        navg : int
            the desired number of channels to average
        """
        if navg <= 1:
            return

        nfull = self.grid.nw // navg
        nres = self.grid.nw -  nfull * navg

        if nres == 0:
            nw = nfull
        else:
            nw = nfull + 1

        # determine the quantities 
        quant = ['I']
        if hasattr(self, 'Q'):
            quant.append('Q')
        if hasattr(self, 'U'):
            quant.append('U')
        if hasattr(self, 'V'): 
            quant.append('V')

        for iquant in quant:
            newim = np.zeros([self.grid.nx, self.grid.ny, nw])
            newf = np.zeros([nw])

            # Begin iteration 
            for i in range(nfull):
                # find the indices
                inx = np.arange(navg, dtype=int) + i * navg

                newim[:,:,i] = np.mean(getattr(self, iquant)[:,:,inx], axis=2)
                newf[i] = np.mean(self.grid.f[inx])

            # Take care of the residual images
            if nres != 0:
                newim[:,:,-1] = np.mean(getattr(self, iquant)[:,:,nfull*navg:], axis=2)
                newf[-1] = np.mean(self.grid.f[nfull*navg:])

            # record the changes
            setattr(self, iquant, newim)

        # change the grid too
        self.grid.set_frequency(newf)
        self.grid.get_wavelength()
        if hasattr(self.grid, 'restfreq'):
            self.grid.get_velocity()

    # ==== plotting ====
    def plot_poldir(self, ax=None, iwav=0, 
        sample='int', sample_kw={'nv1':20,'nv2':20}, 
        axis_unit='au', north='right', 
        unitlpol=0.01, unitlen=10, 
        lpol_lim=[0, 1], 
        unitloc=None, 
        textcolor='w',
        quiv_kwargs={'width':0.005, 'facecolor':'w'}, 
        ):
        """ plot polarization direction. only for rectangular display 
        Parameters
        ----------
        iwav : int
            index of the wavelength
        sample : str
            the differnt modes of sampling the vectors
            'int' = simply sample at distrete intervals
            'interp' = do interpolation
        sample_kw : dict
            The input parameters depending on the sampling mode
            'int' 
                nv1, nv2 : int
                    number of vectors along the first axis and second axis
            'interp'
        axis_unit : str
            length scale of the axis. 'cm', 'au', 'arcsec'
        north : str
            the direction of north for the image. 'right' or 'up'
        unitlpol : float
            unit length of the vectors in polarization fraction 
        unitlen : str or float
            the length of the vector in units of the axis_unit of each unit of polarization
        quiv_kwargs : dict
            arguments for styles of the quiver vectors. useful arguments include: width, facecolro, width, edgecolor, linewidth
        """
        if ax is None:
            ax = plt.gca()

        if isinstance(self.grid, rectangularGrid):
            axis1 = self.grid.x
            axis2 = self.grid.y
        elif isinstance(self.grid, circularGrid):
            axis1 = self.grid.r
            axis2 = self.grid.phi
        else:
            raise ValueError('image grid shape unknown')

        # coordinate units
        if axis_unit == 'cm':
            unitfac = 1.
        elif axis_unit == 'au':
            unitfac = natconst.au
        elif axis_unit == 'arcsec':
            unitfac = natconst.au * self.grid.dpc
        else:
            raise ValueError('axis_unit unknown')

        # ==== sample the points ====
        if sample == 'int':
            nv1 = sample_kw['nv1']
            nv2 = sample_kw['nv2']
            ii = np.linspace(0, len(axis1)-1, nv1, dtype=int)
            jj = np.linspace(0, len(axis2)-1, nv2, dtype=int)

            if isinstance(self.grid, rectangularGrid):
#                x = self.grid.x[ii] / unitfac
#                y = self.grid.y[jj] / unitfac
#                xx, yy = np.meshgrid(x, y, indexing='ij')
                xx = self.grid.xx[np.ix_(ii,jj)]
                yy = self.grid.yy[np.ix_(ii,jj)] 

            elif isinstance(self.grid, circularGrid):
                r = self.grid.r[ii]
                phi = self.grid.phi[jj]
 
                rr, pp = np.meshgrid(r, phi, indexing='ij')
                xx = rr * np.cos(pp)
                yy = rr * np.sin(pp)
            else:
                raise ValueError('grid instance unknown')

            # determine the values at that point
            lpa = self.lpa[:,:,iwav][np.ix_(ii,jj)]
            lpol = self.lpol[:,:,iwav][np.ix_(ii,jj)]

        else:
            # actually interpolate the values
            try:
                fn_Q = self.fn_Q[iwav]
            except AttributeError:
                fn_Q = self.get_fn_quant_w(quant='Q', iwav=iwav)

            try:
                fn_U = self.fn_U[iwav]
            except AttributeError:
                fn_U = self.get_fn_quant_w(quant='U', iwav=iwav)

            try:
                fn_lpol = self.fn_lpol[iwav]
            except AttributeError:
                fn_lpol = self.get_fn_quant_w(quant='lpol', iwav=iwav)

            # determine the sampling geometry
            if sampl_kw['shape'] == 'input':
                # the user directly gives the 2d x and y sampling points
                xx, yy = sample_kw['xx'], sample_kw['yy']

            elif sample_kw['shape'] == 'rect':
                # the sampling points are on a rectangular grid
                xx, yy = np.meshgrid(sample_kw['x'], sample_kw['y'], indexing='ij')

            elif sample_kw['shape'] == 'ell':
                # the sampling points are an ellipse
                """ 
                r : radius grid
                p : phi grid
                inc : inclination
                rot : the angle of the major axis (counterclock-wise from x-axis)
                """
                rr, pp = np.meshgrid(sample_kw['r'], sample_kw['p'], indexing='ij')
                dumx = rr * np.cos(pp)
                dumy = rr * np.sin(pp) * np.cos(sample_kw['inc'])
                ang = sample_kw['rot']
                xx = np.cos(ang) * dumx - np.sin(ang) * dumy
                yy = np.sin(ang) * dumx + np.cos(ang) * dumy

            else:
                raise ValueError('sample_kw shape unknown: %s'%sample_kw['shape'])

            # interpolate the Q and U for the angle
            # the interpolation depends on the shape of the grid
            lpa = np.zeros_like(xx)
            plt_Q = np.zeros_like(lpa)
            plt_U = np.zeros_like(lpa)
            lpol = np.zeros_like(lpa)
            if isinstance(self.grid, rectangularGrid):
                for i in range(xx.shape[1]):
                    plt_Q[:,i] = fn_Q(xx[:,i], yy[:,i], grid=False)
                    plt_U[:,i] = fn_U(xx[:,i], yy[:,i], grid=False)
                    lpol[:,i] = fn_lpol(xx[:,i], yy[:,i], grid=False)

            elif isinstance(self.grid, circularGrid):
                rr = np.sqrt(xx**2 + yy**2)
                pp = np.arctan2(yy, xx)
                for i in range(xx.shape[1]):
                    plt_Q[:,i] = fn_Q(rr[:,i], pp[:,i], grid=False)
                    plt_U[:,i] = fn_U(rr[:,i], pp[:,i], grid=False)
                    lpol[:,i] = fn_lpol(rr[:,i], pp[:,i], grid=False)
            else:
                raise ValueError('grid instance unknown')
            lpa = calc_lpa(plt_Q, plt_U)

            del plt_Q, plt_U

        if north == 'right':
            hori = xx / unitfac 
            vert = yy / unitfac
        elif north == 'up':
            hori = yy / unitfac
            vert = xx / unitfac
        else:
            raise ValueError('north unknown')

        # ==== determine the length of the vector ====
        if unitlen == 'no_scale':
            # not to scale 
            vlen = 1
            plt_len = None
        else:
            vlen = lpol / unitlpol * unitlen
            plt_len = str(unitlpol * 100) + '%'

        # ==== determine the angles ====
        if isinstance(self.grid, rectangularGrid):
            if north == 'right':
                v_frame_hori = vlen * np.cos(lpa)
                v_frame_vert = vlen * np.sin(lpa)
            elif north == 'up':
                v_frame_hori = vlen * np.sin(lpa)
                v_frame_vert = vlen * np.cos(lpa)
            else:
                raise ValueError('north unknown')
        else:
            v_frame_hori = vlen * np.cos(lpa)
            v_frame_vert = vlen * np.sin(lpa)

        # take out any vectors with polarization too small or too large
        if lpol_lim is not None:
            reg = (lpol < lpol_lim[0]) | (lpol > lpol_lim[1])
            v_frame_hori[reg] = np.nan
            v_frame_vert[reg] = np.nan

        ax.quiver(hori, vert, v_frame_hori, v_frame_vert,  
            angles='xy', 
            pivot='mid', 
            scale=1,
            scale_units='xy', 
            headwidth=1e-10, headlength=1e-10, headaxislength=1e-10, 
            **quiv_kwargs)

        # set up the text
        if plt_len != None:
            if unitloc is None:
                textx = 0.75 * xx.max() / unitfac
                texty = 0.75 * yy.min() / unitfac
            else:
                textx, texty = unitloc
            ax.quiver(textx, texty, unitlen * np.cos(np.pi), unitlen * np.sin(np.pi), 
                angles='xy', 
                pivot='mid', 
                scale=1,
                scale_units='xy',
                headwidth=1e-10, headlength=1e-10, headaxislength=1e-10, 
                **quiv_kwargs)

            ax.text(textx, texty, plt_len, color=textcolor, 
                va='bottom', ha='center')

    def plot_beam(self, iwav=0, beamxy=None, axis_unit='au', north='right', 
            facecolor='w', ax=None):
        """ plot the gaussian beam size
        Parameters
        ----------
        implot : ax map? 
        beamxy : tuple
            location of the center of the beam 
        axis_unit : string
            unit the beam should be and also for beamxy
        north : str
            direction of north. 'right' or 'up'
        """
        if self.psfarg['type'] != 'gaussian':
            raise ValueError('psf type not available')

        if ax is None:
            ax = plt.gca()

        # determine size of ellipse to overplot
        # conversion factor from arcsec 
        if axis_unit == 'cm':
            fac = self.grid.dpc * natconst.au
        elif axis_unit == 'au':
            fac = self.grid.dpc
        elif axis_unit == 'arcsec':
            fac = 1. 
        else:
            raise ValueError('axis_unit unknown: %s'%axis_unit)

        # fetch the bmaj, bmin and bpa
        try:
            bmaj = self.psfarg['bmaj'][iwav]
            bmin = self.psfarg['bmin'][iwav]
            bpa = self.psfarg['bpa'][iwav]
        except TypeError:
            bmaj = self.psfarg['bmaj']
            bmin = self.psfarg['bmin']
            bpa = self.psfarg['bpa']

        # determine the height and width of the ellipse
        if north == 'right':
            ewidth = bmaj
            eheight = bmin
            ang = bpa
        elif north == 'up':
            ewidth = bmin
            eheight = bmaj
            ang = - bpa
        else:
            raise ValueError('north unknown')

        # center of the beam
        if beamxy is None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ex = xmin + 0.8 * (xmax - xmin)
            ey = ymin + 0.2 * (ymax - ymin)
        else:
            ex = beamxy[0]
            ey = beamxy[1]

        # set up the ellipse
        ells = Ellipse( xy=(ex,ey), 
            width=ewidth*fac, height=eheight*fac, angle=ang)

        ells.set_facecolor(facecolor)
        ells.set_fill(True)
        ax.add_patch(ells)

class opticalDepth(baseImage):
    """ optical depth 
    Attributes
    ----------
    image 
    """
    def __init__(self):
        baseImage.__init__(self)

    def read(self, fname, binary=False, shape='rect'):
        """ read the optical depth image
        shape : str
            'rect' = rectangular
            'circ' = circular
        """
        if shape == 'rect':
            out = read_rectangular_image(fname, binary=binary)
            self.grid = rectangularGrid()
        elif shape == 'circ':
            out = read_circular_image(fname, binary=binary)
            self.grid = circularGrid()
        else:
            raise ValueError('image shape unknown')

        self.grid.setup_from_file(out)

        if len(out['image'].shape) == 3:
            self.isStokes = False
        elif len(out['image'].shape) == 4:
            self.isStokes = True
            raise ValueError('stokes is not allowed for optical depth')
        else:
            raise ValueError('image dimension unknown')

        self.image = out['image']
        self.quantity.append('image')

    def rotate_in_sky(self, phi):
        """ change the properties that change depending on the 
        orientation in the sky. this completely replaces the previous values
        phi : float
            the angle in radians. counter clockwise from the x-axis
        """
        # change the grid
        self.grid.rotate_in_sky(phi)

class tauSky(baseImage):
    """ tau location from the plane of sky
    Attributes
    ----------
    image
    """
    def __init__(self):
        baseImage.__init__(self)

    def read(self, fname, binary=False, shape='rect'):
        """ read the tau=? location image
        shape : str
            'rect' = rectangular
            'circ' = circular
        """
        if shape == 'rect':
            out = read_rectangular_image(fname, binary=binary)
            self.grid = rectangularGrid()
        elif shape == 'circ':
            out = read_circular_image(fname, binary=binary)
            self.grid = circularGrid()
        else:
            raise ValueError('image shape unknown')

        self.grid.setup_from_file(out)

        if len(out['image'].shape) == 3:
            self.isStokes = False
        elif len(out['image'].shape) == 4:
            self.isStokes = True
            raise ValueError('stokes is not allowed for optical depth')
        else:
            raise ValueError('image dimension unknown')

        self.image = out['image']
        self.quantity.append('image')

class tau3d(baseImage):
    """ tau location in model coordinates
    Attributes
    ----------
    tx, ty, tz
    """
    def __init__(self):
        baseImage.__init__(self)

    def read(self, fname, binary=False, shape='rect'):
        """ read image
        """
        if shape == 'rect':
            out = read_rectangular_tau3d(fname, binary=binary)
            self.grid = rectangularGrid()
        elif shape == 'circ':
            raise ValueError('not done yet')
            out = read_circular_image(fname, binary=binary)
            self.grid = circularGrid()
        else:
            raise ValueError('image shape unknown')

        # there aren't any grid info, so do this for now
        self.grid.set_wavelength(out['wav'])

        self.tx = out['x']
        self.ty = out['y']
        self.tz = out['z']

        for iname in ['tx', 'ty', 'tz']:
            self.quantity.append(iname)

class spectralIndex(baseImage):
    """ class to calculate spectral index
    Attributes
    ----------
    im1, im2
    image
    """
    def __init__(self):
        baseImage.__init__(self)

    def set_image1(self, intensity, iwav=0):
        """ set the first intensity image
        iwav : int
            the selected wavelength from the intensity object
        """
        self.im1 = intensity
        self.iwav1 = iwav
        self.w1 = self.im1.grid.w[iwav]

        self.grid = copy.deepcopy(self.im1.grid)

    def set_image2(self, intensity, iwav=0):
        self.im2 = intensity
        self.iwav2 = iwav
        self.w2 = self.im2.grid.w[iwav]

    def get_spec(self, thres=3):
        """ calculate the spectral index
        use the first image as the base grid
        """
        # check if the stokes units are the same
        if self.im1.stokes_unit != self.im2.stokes_unit:
            raise ValueError('stokes_unit are not the same')

        # obtain the interpolation object of the second object
        fn = self.im2.get_fn_quant_w(quant='I', iwav=self.iwav2)

        # interpolate 
        if isinstance(self.grid, rectangularGrid):
            im2_interp = fn(self.grid.x, self.grid.y).T
        elif isinstance(self.grid, circularGrid):
            if hasattr(self.grid, 'xx') is False:
                self.grid.get_mesh()
            raise ValueError('not done yet')
        else:
            raise ValueError('grid type unknown')

        # calculate spectral index 
        dlognu = - np.log(self.w2 / self.w1)
        dlogI = np.log(im2_interp) - np.log(self.im1.I[:,:,self.iwav1])

        self.spec = dlogI[:,:,None] / dlognu
        self.quantity.append('spec')

        # filter out regions below some threshold
        if hasattr(self.im1, 'I_rms') & hasattr(self.im2, 'I_rms'):
            reg = (self.im1.I[:,:,self.iwav1] <= thres*self.im1.I_rms[self.iwav1]) | (im2_interp <= thres * self.im2.I_rms[self.iwav2])
            self.spec[reg] = 0. 

# ====================
# reading / writing data
# ====================
def read_rectangular_image(fname, binary=False):
    """Reads a rectangular image calculated by RADMC-3D 

    Parameters
    ----------

    fname   : str

    binary  : bool, optional
             False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)

    """
    out = {}
    if binary:
        with open(fname, 'rb') as rfile:
            # hdr
            iformat, nx, ny, nw = np.fromfile(rfile, count=4, dtype=int)

            # Pixel sizes
            dx, dy = np.fromfile(rfile, count=2, dtype=np.float64)

            # Wavelength of the image
            wav = np.fromfile(rfile, count=nw, dtype=np.float64)

            # intensity information 
            if iformat == 1:
                # just stokes I
                dim = (nx, ny, nw)
            elif iformat == 3:
                # full stokes
                dim = (4, nx, ny, nw)
            flat_im = np.fromfile(rfile, count=np.prod(dim), dtype=np.float64)
    else:
        with open(fname, 'r') as rfile:
            # hdr
            iformat, nx, ny, nw = np.fromfile(rfile, count=4, sep=" ", dtype=int)
            
            # Pixel sizes
            dx, dy = np.fromfile(rfile, count=2, sep=" ")

            # Wavelength of the image
            wav = np.fromfile(rfile, count=nw, sep=" ")

            # intensity information 
            if iformat == 1:
                # just stokes I
                dim = (nx, ny, nw)
            elif iformat == 3:
                # full stokes
                dim = (4, nx, ny, nw)
            flat_im = np.fromfile(rfile, count=np.prod(dim), sep=" ")

    im = np.reshape(flat_im, dim, order='F')
    if iformat == 3:
        im = np.moveaxis(im, 0, -1)

    out['iformat'] = iformat
    out['nx'] = nx
    out['ny'] = ny
    out['nw'] = nw
    out['dx'] = dx
    out['dy'] = dy
    out['wav'] = wav
    out['image'] = im

    return out

def read_rectangular_tau3d(fname, binary=False):
    """ read rectangular tau3d
    """
    if binary:
        raise ValueError('not done yet')

    with open(fname, 'r') as rfile:

        dum = ''
        # Format number
        iformat = int(rfile.readline())

        # Nr of pixels
        dum = rfile.readline()
        dum = dum.split()
        nx = int(dum[0])
        ny = int(dum[1])

        # Nr of frequencies
        nwav = int(rfile.readline())

        # Wavelength of the image
        wav = np.zeros(nwav, dtype=np.float64)
        for iwav in range(nwav):
            wav[iwav] = float(rfile.readline())

        # blank space
        dum = rfile.readline()

        data = np.fromfile(rfile, sep=' ', count=-1)
        data = np.reshape(data, (3,nx,ny,nwav), order='F')

        x = data[0,...]
        y = data[1,...]
        z = data[2,...]

    return {'nx':nx, 'ny':ny, 'nwav':nwav, 'wav':wav,
            'x':x, 'y':y,'z':z}

def read_circular_image(fname, binary=False):
    """Reads a circular image calculated by RADMC-3D 

    Parameters
    ----------

    fname   : str

    binary  : bool, optional
             False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)

    """
    out = {}
    if binary:
        with open(fname, 'rb') as rfile:
            # hdr
            iformat, nr, nphi, nw = np.fromfile(rfile, count=4, dtype=int)

            # radius info
            rwall = np.fromfile(rfile, count=nr+1, dtype=np.float64)
            rcell = np.fromfile(rfile, count=nr, dtype=np.float64)

            # phi information
            pwall = np.fromfile(rfile, count=nphi+1, dtype=np.float64)
            pcell = np.fromfile(rfile, count=nphi, dtype=np.float64)

            # Wavelength of the image
            wav = np.fromfile(rfile, count=nw, dtype=np.float64)

            # intensity information 
            if iformat == 1:
                # just stokes I
                dim = (nr+1, nphi, nw)
            elif iformat == 3:
                # full stokes
                dim = (4, nr+1, nphi, nw)

            flat_im = np.fromfile(rfile, count=np.prod(dim), dtype=np.float64)
    else:
        with open(fname, 'r') as rfile:
            # hdr
            iformat, nr, nphi, nw = np.fromfile(rfile, count=4, sep=" ", dtype=int)

            # radius info
#            ! need to check the extra 0!
            rwall = np.fromfile(rfile, count=nr+2, sep=" ")
            rcell = np.fromfile(rfile, count=nr+1, sep=" ")

            # phi information
            pwall = np.fromfile(rfile, count=nphi+1, sep=" ")
            pcell = np.fromfile(rfile, count=nphi, sep=" ")

            # Wavelength of the image
            wav = np.fromfile(rfile, count=nw, sep=" ")

            # intensity information 
            if iformat == 1:
                # just stokes I
                dim = (nr+1, nphi, nw)
            elif iformat == 3:
                # full stokes
                dim = (4, nr+1, nphi, nw)
            else:
                raise ValueError('iformat unkonwn')

            flat_im = np.fromfile(rfile, count=np.prod(dim), sep=" ")

    im = np.reshape(flat_im, dim, order='F')
    if iformat == 3:
        im = np.moveaxis(im, 0, -1)

    if iformat == 1:
        isStokes = False
    elif iformat == 3:
        isStokes = True
    else:
        raise ValueError('iformat unkonwn')

    out['iformat'] = iformat
    out['nr'] = nr
    out['nphi'] = nphi
    out['nw'] = nw
    out['rwall'] = rwall
    out['rcell'] = rcell
    out['pwall'] = pwall
    out['pcell'] = pcell
    out['wav'] = wav
    out['image'] = im
    out['isStokes'] = isStokes

    return out

def write_camera_wavelength(wav, fname='camera_wavelength_micron.inp', fdir=''):
    """ write camera_wavelength_micron.inp for this list of wavelengths
    """
    if fdir != '':
        fname = os.path.join(fdir, fname)

    with open(fname, 'w') as f:
        f.write('%d\n'% len(wav))
        for ii in range(len(wav)):
            f.write('%.7e\n'%wav[ii])
    
def read_camera_wavelength(fname='camera_wavelength_micron.inp', fdir=''):
    """ read camera_wavelength_micron.inp
    """
    if fdir != '':
        fname = os.path.join(fdir, fname)

    data = np.fromfile(fname, count=-1, sep=' ', dtype=np.float64)
    wav = data[1:]
    return wav

# ====================
# elementary calculations
# ====================
def calc_beam_solid_angle(bmaj, bmin):
    """ bmaj, bmin in arcsec
    """
    return (bmaj / 3600. * natconst.rad) * (bmin / 3600. * natconst.rad) * np.pi/4. / np.log(2.)


def calc_lpa(Q, U):
    """ function to calculate polarization angle
    Parameters 
    ----------
    Q : ndarray
    U : ndarray

    Returns
    -------
    ang : ndarray
        the polarization angle in radians and in between 0, 2pi
    """
    ang = np.arctan2(U, Q) / 2.
    reg = ang < 0.
    ang[reg] = ang[reg] + 2*np.pi
    reg = ang > np.pi
    ang[reg] = ang[reg] - np.pi
    return ang

def rot_inc(inc, y, z):
    """
    fix the object, and rotate the observer frame around the x-axis
    and return the coordinates of the object in the new frame
    """
    return grid.rot_x(inc, y, z)

def rot_eon(ang_eon, x, y):
    """
    east is the x-axis, north is the y-axis
    rotate by some angle east-of-north, meaning left-hand rotate around the z-axis, relative to the y-axis
    return the coordinates of the object in the new frame
    """
    ang = ang_eon
    x1, y1 = grid.rot_z(ang, x, y)
    return x1, y1

def get_gaussian_psf(nx, ny, bmaj, bmin, bpa, pscale=None):
    """ calculate 2d gaussian psf
    nx, ny : int
        number of grid cells in x, y
    bmaj, bmin: float
        beam major and minor axis, in units corresponding to pscale
    bpa : float
        direction of the beam major axis East-of-North, ie., relative to the y-axis, following the left-hand rule. degrees
    """
    # determine pixel scale
    if pscale is None:
        dx, dy = 1, 1
    else:
        dx, dy = pscale

    # calculate x,y axis
    x = (np.arange(nx, dtype=np.float64) - nx / 2.) * dx
    y = (np.arange(ny, dtype=np.float64) - ny / 2.) * dy

    # Calculate the standard deviation of the Gaussians
    sigmax = bmaj / (2.0 * np.sqrt(2.0 * np.log(2.)))
    sigmay = bmin / (2.0 * np.sqrt(2.0 * np.log(2.)))
    norm = (2. * np.pi * sigmax * sigmay) / dx / dy

    sin_bpa = np.sin(bpa / 180. * np.pi - np.pi / 2.)
    cos_bpa = np.cos(bpa / 180. * np.pi - np.pi / 2.)

    # Define the psf
    xx, yy = np.meshgrid(x, y, indexing='ij')
    bxx = cos_bpa * xx - sin_bpa * yy
    byy = sin_bpa * xx + cos_bpa * yy
    psf = np.exp(-0.5 * (bxx / sigmax)**2 - 0.5 * (byy / sigmay)**2)

    psf /= norm 

    return psf 

def convolve_image_psf(image, psf):
    """ convolves two images
    """
    conv = signal.fftconvolve(image, psf, mode='same')
    return conv

# ====================
# image unit conversions
# ====================
def jyppix_to_cgs(jyppix, dxy):
    """
    convert jy/pixel image to inu in cgs
    Parameters
    ----------
    jyppix       : ndarray
        the image in jy/pixel units
    dxy     : float
        the solid angle size the pixel in arcsec**2
    """
    solid_angle = dxy / 3600.**2 * natconst.rad**2
    return jyppix * natconst.jy / solid_angle

def cgs_to_jyppix(cgsim, dxy):
    solid_angle = dxy / 3600.**2 * natconst.rad**2
    return cgsim * solid_angle / natconst.jy

def jypbeam_to_cgs(jypbeam, bmaj, bmin):
    """ convert image of jy/beam to cgs units
    """
    beam = calc_beam_solid_angle(bmaj, bmin)
    return jypbeam * natconst.jy / beam

def cgs_to_jypbeam(cgsim, bmaj, bmin):
    beam = calc_beam_solid_angle(bmaj, bmin)
    return cgsim * beam / natconst.jy

def cgs_to_tb(cgsim, wav_micron):
    """ convert image in cgs units to brightness temperature
    """
    freq = natconst.cc * 1e4 / wav_micron
    ld2 = (wav_micron*1e-4)**2
    hnu = natconst.hh * freq
    hnu3_c2 = natconst.hh * freq**3 / natconst.cc**2
    tb = hnu / natconst.kk / np.log(2. * hnu3_c2 / abs(cgsim) + 1.) * np.sign(cgsim)
    return tb

def tb_to_cgs(tb, wav_micron):
    """ convert image in brightness temperature to cgs units
    basically the planck function
    """
    freq = natconst.cc * 1e4 / wav_micron
    ld2 = (wav_micron * 1e-4)**2
    hnu = natconst.hh * freq
    hnu3_c2 = natconst.hh * freq**3 / natconst.cc**2
    cgs = 2 * hnu3_c2 / (np.exp(hnu / natconst.kk / tb) - 1.)
    return cgs

def jyppix_to_tb(jyppix, dxy, wav_micron):
    """
    converts jy/pixel image to brightness temperature 
    """
    cgsim = jyppix_to_cgs(jyppix, dxy)
    tb = cgs_to_tb(cgsim, wav_micron)
    return tb

def jypbeam_to_tb(jypbeam, bmaj, bmin, wav_micron):
    """
    converts jy/beam image to brightness temperature 
    """
    cgsim = jypbeam_to_cgs(jypbeam, bmaj, bmin)
    tb = cgs_to_tb(cgsim, wav_micron)
    return tb

def tb_to_jypbeam(tb, bmaj, bmin, wav_micron):
    """ convert brightness temperature to jy/beam
    """
    cgsim = tb_to_cgs(tb, wav_micron)
    jypbeam = cgs_to_jypbeam(cgsim, bmaj, bmin)
    return jypbeam

# ====================
# comman interface to radmc3d
# ====================
class commandImage(object):

    def __init__(self):
        # image mode 
        self.mode = 'image'

        # image coordinates 
        self.npix = 100
        self.sizeau = 100

        self.circ = False

        # wavelength 
        self.lambda_arg = 'lambda 10'

        # camera location 
        self.incl = 60.
        self.phi = None
        self.posang = None
        self.pointau = None
        self.zoomau = None

        # binary options
        self.fluxcons = True
        self.stokes = False
        self.doppcatch = False
        self.binary = False
        self.nostar = False
        self.noscat = False
        self.secondorder = False

        # scattering keywords 
        self.maxnrscat = None
        self.nphot_scat = None

    def set_mode(self, inp):
        """ the imaging mode
        """
        if inp == 'image':
            mode = 'image'
        elif inp == 'surf':
            mode = 'tausurf'
        elif inp == 'tracetau':
            mode = 'image tracetau'
        elif inp == 'tracecolumn':
            mode = 'image tracecolumn'
        else:
            raise ValueError('imaging mode input unknown: %s'%inp)

        self.mode = mode

    def set_wavelength(self, wav=None,
        lambdarange=None, nlam=2,
        iline=None, imolspec=1, vkms=0., widthkms=10., linenlam=10,
        loadlambda=False):
        """ determine the wavelength part of the command
        """
        if loadlambda:
            # loading the camera wavelength input
            c = 'loadlambda'

        elif wav is not None:
            # directly denote the wavelength 
            c = 'lambda %f'%wav

        elif lambdarange is not None:
            # wavelength range 
            c = 'lambdarange {:d} {:d} nlam {:d}'.format(lambdarange[0], lambdarange[1], nlam)
#            c = 'lambdarange %d %d nlam %d'(lambdarange[0], lambdarange[1], nlam)
        elif iline is not None:
            # set velocity grid around a line 
            c = 'iline %d imolspec %d widthkms %f vkms %f linenlam %d'%(
                iline, imolspec, widthkms, vkms, linenlam)
        else:
            raise ValueError('wavelength part incorrect')

        self.lambda_arg = c

    def set_pointau(self, pointau):
        """ pointau : 3 element list
        """
        if len(pointau) != 3:
            raise ValueError('pointau should be a list of 3 elements corresponding to the  cartesian coordinates of the image center')

        self.pointau = pointau

    def get_command(self):
        """ create the command
        """
        com = 'radmc3d'
        com += ' %s'%self.mode
        if self.circ:
            com += ' circ'
        com += ' npix %d'%self.npix
        com += ' sizeau %f'%self.sizeau
        com += ' ' + self.lambda_arg

        # camera location 
        com += ' incl %f'%self.incl
        if self.phi is not None:
            com += ' phi %f'%self.phi
        if self.posang is not None:
            com += ' posang %f'%self.posang
        if self.pointau is not None:
            com += ' pointau'
            for ii in range(len(self.pointau)):
                com += ' %f'%self.pointau[ii]
        if self.zoomau is not None:
            com+= ' zoomau %s'%self.zoomau

        # binary options
        if self.fluxcons:
            com += ' fluxcons'
        if self.stokes:
            com += ' stokes'
        if self.doppcatch:
            com += ' doppcatch'
        if self.binary:
            com += ' imageunform'
        if self.nostar:
            com += ' nostar'
        if self.noscat:
            com += ' noscat'
        if self.secondorder:
            com += ' secondorder'

        # scattering  
        if self.maxnrscat is not None:
            com += ' maxnrscat %d'%self.maxnrscat
        if self.nphot_scat is not None:
            com+= ' nphot_scat %d'%self.nphot_scat

        self.command = com

    def print_command(self):
        """ print the command
        """
        print(self.command)

    def make(self, fname=None):
        """ create the command and run radmc3d
        """
        if hasattr(self, 'command') is False:
            raise ValueError('command not found')

        dum = subprocess.Popen([self.command], shell=True).wait()

        possible_imname = ['image.out', 'circimage.out']
        detect_image = False
        for iname in possible_imname:
            detect_image = detect_image | os.path.isfile(iname)

        if detect_image is False:
            msg = 'Did not succeed in making image. \n'
            msg = msg + 'Failed command: '+self.command
            raise ValueError(msg)

        # rename the file
        if type(fname) is str:
            for iname in possible_imname:
                if os.path.isfile(iname):
                    os.system('mv %s %s'%(iname, fname))

        print('Ran command: %s'%self.command)


