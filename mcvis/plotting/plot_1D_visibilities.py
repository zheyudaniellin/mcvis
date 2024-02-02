import pdb
import matplotlib.pyplot as plt
import numpy as np

def plot_1D_visibilities(ax, o1d, m1d, color="k", markersize=8, linewidth=1,
    line_color="g", fontsize="medium"):
    """
    plot the data and model 
    Arguments
    ---------
    o1d : interferometry.Visibility
        the 1D visibility of the observations
    m1d : interferometry.Visibility
        the 1D visibility of the model
    """
    
    # Calculate the error properly.
    real_samples = o1d.real + np.random.normal(0, 1, (o1d.real.size, 1000)) * 1./o1d.weights**0.5
    imag_samples = o1d.imag + np.random.normal(0, 1, (o1d.real.size, 1000)) * 1./o1d.weights**0.5

    amp_samples = (real_samples**2 + imag_samples**2)**0.5

    amp_unc = (np.percentile(amp_samples, [50], axis=1) - \
            np.percentile(amp_samples, [16, 84], axis=1)) * \
            np.array([1,-1])[:,None]

    # Plot the visibilities.
    ax.errorbar(o1d.uvdist/1e3, o1d.amp[:,0]*1e3, yerr=amp_unc*1e3, 
        fmt='o', markersize=markersize, markeredgecolor=color,
        markerfacecolor=color, ecolor=color)

    # plot the model
    ax.plot(m1d.uvdist/1e3, m1d.amp*1e3, "-", color=line_color, \
            linewidth=linewidth)

    # Adjust the plot and add axes labels.
    ax.axis([1, o1d.uvdist.max()/1e3*3, 0, o1d.amp.max()*1.1*1e3])
    ax.set_xscale('log', nonpositive='clip')

    ax.set_xlabel("Baseline [k$\lambda$]", fontsize=fontsize)
    ax.set_ylabel("Amplitude [mJy]", fontsize=fontsize)


