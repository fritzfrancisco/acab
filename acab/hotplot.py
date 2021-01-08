import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import ConnectionPatch
import numpy as np
from scipy import stats
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


def shareXlimits(axs):
    '''
    chooses automatically the maximal
    INPUT:
        axs list or 1d-np.ndarray
    '''
    xlim = list(axs[0].get_xlim())
    for ax in axs[1:]:
        xlim += list(ax.get_xlim())
    minne  = np.min(xlim)
    maxxe  = np.max(xlim)
    for ax in axs:
        ax.set_xlim([minne, maxxe])


def shareYlimits(axs):
    '''
    INPUT:
        axs list or 1d-np.ndarray
    '''
    ylim = list(axs[0].get_ylim())
    for ax in axs[1:]:
        ylim += list(ax.get_ylim())
    minne  = np.min(ylim)
    maxxe  = np.max(ylim)
    for ax in axs:
        ax.set_ylim([minne, maxxe])


def connectionPatch(axA, axB, posA, posB, **kwgs):
    '''
    Connects points in 2 axes with a line (ACROSS axis boundaries)
        - just a simple wrapper to remember
    Note that axA should always be the last axes, e.g.
        f, [ax0, ax1] = plt.subplots(2)
        axA = ax1
        axB = ax2
    otherwise the connection is not visible in the last axis 
    (because it is overdrawn)
    '''
    con = ConnectionPatch(axesA=axA, axesB=axB, xyA=posA, xyB=posB,
                          coordsA="data", coordsB="data", **kwgs)
    axA.add_artist(con)


def yAxisRight(axs):
    axs.yaxis.tick_right()
    axs.yaxis.set_label_position('right')
    axs.spines['right'].set_visible(True)
    axs.spines['left'].set_visible(False) # no spine 


def yAxisOff(axs):
    axs.get_yaxis().set_visible(False) # no ticks
    axs.spines['left'].set_visible(False) # no spine 


def regressionPlot(x, y, bins=None, alpha=None, c=None,
                   xlab=None, ylab=None, s=None, polyfit=None,
                   axs=None, minDat=None):
    '''
    plot collection to estimate the relation between x and y
    . PLOT0            PLOT1           PLOT2
    . raw data         2d-Histogram    equally weighted bin data
    .             +equally weighted    +regression
    .                bin data
    .             + regression
    '''
    polyfit = jut.setDefault(polyfit, 1)
    bins = jut.setDefault(bins, 40)
    alpha = jut.setDefault(alpha, 0.4)
    c = jut.setDefault(c, 'k')
    xlab = jut.setDefault(xlab, 'x')
    ylab = jut.setDefault(ylab, 'y')
    s = jut.setDefault(s, 6)
    minDat = jut.setDefault(minDat, 20)
    axsWasNone = False
    if axs is None:
        axsWasNone = True
        f, axs = plt.subplots(1, 4, figsize=0.8*plt.figaspect(1/4.6), sharex=True)
    # plot: raw
    axs[0].scatter(x, y, c=c, alpha=alpha/4, s=s/3)
    # plot: histogram
    histOut = axs[1].hist2d(x, y, bins=bins, cmap=cm.Reds)
    # plot equally weighted bins
    [xEqualPart, yEqualPart,
     weightsEP] = jut.bin2d(x, y, bins=bins, func=np.mean, equalWeight=True)
    axs[2].scatter(xEqualPart, yEqualPart, c=c, s=s, alpha=alpha)
    if len(axs) > 3:
        # plot equidistant bins
        [xEquiDistBins, yxEquiDistBins,
         weightsEDB] = jut.bin2d(x, y, bins=bins, func=np.mean, equalWeight=False)
        there = np.where(weightsEDB > minDat)[0]
        ses = weightsEDB[there]
        ses -= ses.min()
        ses = ses * 5*s/ses.max() + s
        axs[3].scatter(xEquiDistBins[there], yxEquiDistBins[there], c=c, s=ses,
                       alpha=alpha, marker='^')
    # plot: regression or polyfit
    regr = None
    if polyfit != False:
        if polyfit > 1:
            regr = np.polyfit(xEqualPart, yEqualPart, polyfit)
            fit = np.poly1d(regr)
        else:
            regr = stats.linregress(x, y)
            fit = lambda a : a*regr[0] + regr[1]
        xes = [xEqualPart[0], xEqualPart[-1]]
        xfit = np.arange(xes[0], xes[1], step=np.diff(xes)/100)
        axs[2].plot(xfit, fit(xfit))
    # # fancy connections:
    # for xe in xes:
    #     pos = [xe, fit(xe)]
    #     connectionPatch(axs[2], axs[1], pos, pos, linestyle='--', color='C0')
    # labels, ticks and limits
    shareYlimits(axs[:2])
    axs[1].set_yticklabels([])
    [yAxisRight(ax) for ax in axs[2:]]
    shareYlimits(axs[2:])
    axs[-2].set_yticklabels([])
    axs[1].set_xlabel(xlab)
    _ = [ax.set_ylabel(ylab) for ax in [axs[0], axs[-1]]]
    if axsWasNone:
        f.tight_layout()
        return regr, f, axs
    return regr
