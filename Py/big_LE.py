import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os 
import re
from matplotlib.colors import LogNorm
from matplotlib import colors
from matplotlib.transforms import ScaledTranslation





def get_label(tick):
    multiple = tick / np.pi
    if abs(multiple - round(multiple)) < 1e-12:  # Check if integer multiple
        multiple_int = int(round(multiple))
        if multiple_int == 0:
            return r'$0$'
        elif abs(multiple_int) == 1:
            return r'$\pi$' if multiple_int == 1 else r'$-\pi$'
        else:
            return fr'${multiple_int}\pi$'
    else:  # Check for half-integer multiples
        numerator = round(multiple * 2)
        if abs(numerator) == 1:
            return r'$\pi/2$' if numerator == 1 else r'$-\pi/2$'
        else:
            return fr'${numerator}\pi/2$'





file1 = np.loadtxt('../LE/A2_0.1000_tau_0.1000_beta_1.0000.csv')
data_1 = np.clip(file1, 0, None)

file2 = np.loadtxt('../LE/A2_0.1000_tau_0.1000_beta_3.0000.csv')
data_2 = np.clip(file2, 0, None)

file3 = np.loadtxt('../LE/A2_0.1000_tau_0.1000_beta_5.0000.csv')
data_3 = np.clip(file3, 0, None)

file4 = np.loadtxt('../LE/A2_0.1000_tau_0.2000_beta_5.0000.csv')
data_4 = np.clip(file4, 0, None)

file5 = np.loadtxt('../LE/A2_0.1000_tau_0.4000_beta_5.0000.csv')
data_5 = np.clip(file5, 0, None)

file6 = np.loadtxt('../LE/A2_0.1000_tau_0.6000_beta_5.0000.csv')
data_6 = np.clip(file6, 0, None)

x_ticks = np.arange(0, np.pi + np.pi/4, np.pi/4)
y_ticks = np.arange(0, 2*np.pi + np.pi/4, np.pi/2)

datas = [data_1,data_2,data_3,data_4,data_5,data_6]

x_labels = [get_label(tick) for tick in x_ticks]
y_labels = [get_label(tick) for tick in y_ticks]


name = [["a)","b)"],["c)","d)"],["e)","f)"]]


fig, axs = plt.subplot_mosaic(name,layout='constrained')


fig.set_size_inches(9, 12)  
for i,j in enumerate(axs):
    ax = axs[j]
    im = ax.imshow(datas[i], origin='lower', extent=(0, np.pi, 0, 2*np.pi), aspect='auto', cmap=cmap)
    

for label, ax in axs.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        fontsize='large', va='bottom', fontfamily='serif')
    
plt.savefig("../plot/bigplot.png",bbox_inches='tight',dpi=600)
