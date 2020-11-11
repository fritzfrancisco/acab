import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from acab import juteUtils as jut

def get_ABC(N, Nskip=None):
    Nskip = jut.setDefault(Nskip, 0)
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return abc[Nskip:Nskip+N]

def get_plotCoordinate(x, y, coord):
    '''
    INPUT:
        x.shape (N)
            x-data used for plotting
        y.shape (M)
            y-data used for plotting
        coord.shape (2)
            coordinates relative to borders of plot
    '''
    x = np.array(x)
    y = np.array(y)
    xlen = x.max() - x.min()
    ylen = y.max() - y.min()
    plotCoord = np.array(coord) * np.array([xlen, ylen])
    plotCoord += [x.min(), y.min()]
    return plotCoord

def abc_plotLabels(coord, axs, fontsize=None, Nskip=None, abc=None, **kwgs):
    '''
    INPUT:
        coord.shape (2)
            coordinates relative to borders of plot
        axs [matplotlib.subplotobject, ....]
            list of subplots which needs label
    '''
    abc = jut.setDefault(abc, get_ABC(len(axs), Nskip=Nskip))
    fontsize = jut.setDefault(fontsize, 22)
    Nskip = jut.setDefault(Nskip, 0)
    for i, ax in enumerate(axs):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        coo = coord.copy()
        if ax.yaxis_inverted():
            coo[1] = 1 - coo[1]
        coords = get_plotCoordinate(xlim, ylim, coo)
        ax.text(coords[0], coords[1], abc[i], fontsize=fontsize, **kwgs)


def pcolor_zerowhite(f, axs, mat, xvals=None, yvals=None,
                     cmap=None, cLabel=None,
                     cbar=None, Nticks=None, maxVal=None,
                     cbarHorizontal=None, **kwgs):
    '''
    creates a pcolormesh plot with bwr-colormap
    where white is exaclty at zero
    OR
    with Reds-colormap if only positive or Blues-colormap if only negative data
    INPUT:
        maxVal double
            value at which the results are cut (better name: cutValue)
        cmap matplotlib.cm object
            colormap in use, default is cm.bwr but any other diverging colormap is fine
            e.g. 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'seismic'
        kwgs dict
            keywords for colorbar creation

    '''
    # DEFAULTS
    cbarHorizontal = jut.setDefault(cbarHorizontal, False)
    cbar = jut.setDefault(cbar, True)
    cmap = jut.setDefault(cmap, cm.bwr)
    Nticks = jut.setDefault(Nticks, 4)
    mini = np.nanmin(mat)
    maxi = np.nanmax(mat)
    maxse = np.max([np.abs(mini), maxi])
    maxVal = jut.setDefault(maxVal, maxse)
    if mini*maxi < 0:
        c = axs.pcolormesh(mat, cmap=cmap, vmin=-maxVal, vmax=maxVal)
    elif mini >= 0:
        cmap = truncate_colormap(cmap, minval=0.5, maxval=1.0, n=500)
        c = axs.pcolormesh(mat, cmap=cmap, vmin=mini, vmax=maxVal)
        # c = axs.pcolormesh(mat, cmap=cm.Reds) # old version
    elif maxi <= 0:
        cmap = truncate_colormap(cmap, minval=0, maxval=0.5, n=500)
        c = axs.pcolormesh(mat, cmap=cmap, vmin=-maxVal, vmax=maxi)
        # c = axs.pcolormesh(mat, cmap=cm.Blues_r) # old version
    else:
        c = axs.pcolormesh(mat, cmap=cm.Greys)
    if xvals is not None:
        plot_set_xticks(axs, Nticks, xvals)
    if yvals is not None:
        plot_set_yticks(axs, Nticks, yvals)
    if cbar:
        if cbarHorizontal:
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            ax2_divider = make_axes_locatable(axs)
            cax2 = ax2_divider.append_axes("top", size="7%", pad="17%")
            col = f.colorbar(c, cax=cax2, orientation="horizontal", **kwgs)
            # change tick position to top. Tick position defaults to bottom and overlaps
            # the image.
            # cax2.xaxis.set_ticks_position("top")
            # cax2.xaxis.set_label_position("top")
        else:
            col = f.colorbar(c, ax=axs, **kwgs)
        if cLabel is not None:
            col.set_label(cLabel)
        ticks = col.get_ticks()
        if len(ticks) >=5:
            col.set_ticks(ticks[::2])
        return col


def truncate_colormap(cmap, minval=None, maxval=None, n=None):
    '''
    to truncate an existing colormap
    source:
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    '''
    minval = jut.setDefault(minval, 0.0)
    maxval = jut.setDefault(maxval, 1.0)
    n = jut.setDefault(n, 100)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def hlinesFull(y, axs, low=None, up=None, **kwargs):
    '''
    as plt.hlines but you only need to specify y-position
    the line will span the whole x-axis
    INPUT:
        y float
        axs matplotlib subplot object
        low float
            if lower bound exists for line
        up float
            if upper bound exists for line
    '''
    xlim = axs.get_xlim()
    if low is None:
        low = xlim[0]
    if up is None:
        up = xlim[1]
    axs.hlines(y, low, up, **kwargs)
    axs.set_xlim(xlim)


def vlinesFull(x, axs, low=None, up=None, **kwargs):
    '''
    as plt.vlines but you only need to specify x-position
    the line will span the whole y-axis
    INPUT:
        x float
        axs matplotlib subplot object
        low float
            if lower bound exists for line
        up float
            if upper bound exists for line
    '''
    ylim = axs.get_ylim()
    if low is None:
        low = ylim[0]
    if up is None:
        up = ylim[1]
    axs.vlines(x, low, up, **kwargs)
    axs.set_ylim(ylim)


def histProb(dat, axs=None, **kwgs):
    '''
    as plt.hist but the bins sum to 1
    which is different to plt.hist(dat, density=True)
    where the integral (area) sums to 1
    '''
    if axs is None:
        f, axs = plt.subplots(1)
    weights = np.ones(len(dat), dtype=float)/len(dat)
    out = axs.hist(dat, weights=weights, **kwgs)
    return out


