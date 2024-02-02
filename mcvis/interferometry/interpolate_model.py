import numpy
import scipy.interpolate
from .libinterferometry import Visibilities
import galario
arcsec = 4.84813681e-6      # radians

def interpolate_model(u, v, freq, x, y, im3d, nthreads=1, dRA=0., dDec=0., \
        code="galario", nxy=1024, dxy=0.01):
    """
    Fourier transform the image
    Arguments
    ---------
    u, v : list of 1d ndarray
        the u, v for the visibilities from the observations
    freq : 1d ndarray
        the imaging frequency
    x, y : 1d ndarray
        grid of the image in arcseconds
    im3d : 3d ndarray
        The model intensity in x by y by frequency. The 
    """
    if code == "galario":
        real = []
        imag = []

        galario.double.threads(nthreads)

        # pixel size in radians
        dxy = (x[1] - x[0]) * arcsec

        for i in range(len(model.freq)):
            vis = galario.double.sampleImage(model.image[::-1,:,i,0].copy(order='C'), \
                    dxy, u, v, dRA=dRA*arcsec, dDec=dDec*arcsec)

            real.append(vis.real.reshape((u.size,1)))
            imag.append(-vis.imag.reshape((u.size,1)))

        real = numpy.concatenate(real, axis=1)
        imag = numpy.concatenate(imag, axis=1)

    else:
        raise ValueError('code for fft unknown')

    return Visibilities(u, v, freq, real, imag, numpy.ones(real.shape))

