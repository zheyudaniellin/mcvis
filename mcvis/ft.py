"""
ft.py
"""
import numpy as np
import pdb
import scipy.interpolate
from . interferometry.libinterferometry import Visibilities
from . import interferometry, image
from . natconst import au, arcsec
import galario
import matplotlib.pyplot as plt

def interpolate_model(u, v, freq, x, y, im2d, dRA=0, dDec=0, PA=0):
    """
    similar to the interpolate_model function from interpolate_model.py from pdspy.interfereometry
    But, simply calculate the model visibility for one frequency at a time

    Arguments
    ---------
    u, v : list of 1d ndarray
        the u, v for the visibilities from the observations
    freq : 1d ndarray
        the imaging frequency in GHz?
    x, y : 1d ndarray
        grid of the image in arcseconds
    im2d : 2d ndarray
        The model intensity in x by y from radmc3d. By definition, the x-axis(first axis) is along increasing Dec and the y-axis (second axis)is along increasing RA.
        Note that in galario, RA is denoted as x and Dec is denoted as y.
        However, their first index (i) is along Dec and the second index (j) is along RA. In addition, RA should decrease with increasing j, so we need to reverse that order.
        If Dec increases with increasing i, then we need to add a origin='lower' in the sampleImage function. 
    dRA : float
        R.A. offset w.r.t. the phase center by which the image is translated. If dRA > 0 translate the image towards East. Default is 0. units: arcsec
    dDec : float
        Dec. offset w.r.t. the phase center by which the image is translated. If dDec > 0 translate the image towards North. Default is 0. units: arcsec
    PA : float
        Position Angle, defined East of North. Default is 0. units: degrees
    """
    dxy = (x[1] - x[0]) * arcsec
    if dxy <= 0:
        raise ValueError('The pixel size should be greater than 0')

    # let RA decrease with increasing j-index
    inp = im2d[:,::-1]

    vis = galario.double.sampleImage(
        inp.copy(order='C'), 
        dxy, u, v,
        dRA=dRA*arcsec,
        dDec=dDec*arcsec,
        PA = PA * np.pi/180, 
        origin='lower' # this is super crucial. wasn't there for pdspy
        )

    real = vis.real.reshape((u.size, 1))

    # why do we need a negative sign??? (from pdspy)
#    imag = - vis.imag.reshape((u.size, 1))
    # i think it might've been an issue with origin
    imag = vis.imag.reshape((u.size, 1))

    return Visibilities(u, v, freq, real, imag, np.ones(real.shape))

def ft_visibility_to_image(v_real, v_imag):
    """
    conduct fourier transform from visibility to image. 
    the input amp and phase should be organized such that zero frequency is at the center of the image

    Parameters
    ----------
    v_real : 2d ndarray
    v_imag : 2d ndarray
    """
    # create the complex visibility
    vis = v_real + 1j * v_imag

    # shift the zero frequency location to the corner as np.fft requires
    inp = np.fft.ifftshift(vis)

    # inverse fft
    img = np.fft.ifft2(inp).real

    # for whatever reason, I need to shift the quadrants again
    # this has to be wrong, but i don't know how to fix it
    img = np.fft.fftshift(img)

    return img

def ft_image_to_visibility(img):
    """
    fourier transform from the image to visibility
    """
    # ft from image to visibility
    raw = np.fft.fft2(img)

    # reorganize the matrix
    vis = np.fft.fftshift(raw)

    return vis.real, vis.imag

# ==== classes ====
class manager(object):
    def __init__(self, config):
        self.config = config
        self.vis = [] # model visibilities

    def match(self, obs, mod, dRA, dDec, PA):
        """
        match the uv of the observations to get the model visibilities

        The model image from radmc3d has an x-axis that is along North and y-axis that is along RA

        Arguments
        ---------
        obs : observation.manager()
        mod : modeling.manager()
            This should have the attribute images which is a list of tlx.image.intensity
        """
        ndata = len(mod.images)

        # check if this corresponds to the number of observations
        if ndata != len(obs.vis):
            raise ValueError('number of observations do not match the number of model images')

        # iterate
        for i in range(ndata):
            im = mod.images[i]
            v = interpolate_model(
                obs.vis[i].u, obs.vis[i].v, 
                im.grid.f, 
                im.grid.x / au / im.grid.dpc, 
                im.grid.y / au / im.grid.dpc, 
                im.I[...,0], 
                dRA=dRA[i], dDec=dDec[i], PA=PA)

            self.vis.append(v)

        self.matchargs = {'dRA':dRA, 'dDec':dDec, 'PA':PA}
        self.dpc = mod.images[0].grid.dpc

    def get_vis1d(self):
        """
        calculate the 1D visibility for the model
        This should have a much better resolution than what's done for the observations
        """
        self.vis1d = []
        for i in range(len(self.vis)):
            v = interferometry.average(self.vis[i], gridsize=1000, binsize=3500, radial=True)

            self.vis1d.append(v)

    def get_vis2d(self, gridsize=256):
        """
        calculate the 2D visibility for the model
        """
        self.vis2d = []
        self.vis2d_binsize = []
        for i, ivis in enumerate(self.vis):
            # calculate the binsize
            max_uvdist = ivis.uvdist.max() * 1.05
            binsize = 2 * max_uvdist / gridsize

            # grid the data
            g = interferometry.grid(self.vis[i], gridsize=gridsize, binsize=binsize)

            # reshape the quanities, since we know it's 2D anyways
            for ikey in ['u', 'v', 'real', 'imag', 'weights', 'uvdist', 'amp', 'phase']:
                setattr(g, ikey, getattr(g, ikey).reshape(gridsize, gridsize))

            self.vis2d_binsize.append(binsize)
            self.vis2d.append(g)

            # note that the first index is for v and the second index is for u
            # u and v for the visibilities are meshgrids

    def get_img_from_vis2d(self):
        """
        calculate the image from the sampled visibility. This can serve as an approximation
        The image will be an image.intensity object
        """
        self.img_from_vis2d = []
        for i, ivis in enumerate(self.vis2d):
            x, y, f, I = interferometry.invert.vis_to_img(ivis)

            # create the intensity object
            img = image.intensity()
            img.grid = image.rectangularGrid()
            img.grid.set_x(x*self.dpc*au)
            img.grid.set_y(y*self.dpc*au)
            img.grid.set_frequency(f)
            img.grid.set_dpc(self.dpc)
            img.set_quantity('I', I)
            img.set_stokes_unit('jyppix')

            self.img_from_vis2d.append(img)

