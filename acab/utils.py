import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path

def var_norm_vecTS(x):
    '''
    create variance of the norm of a vector timeseries
    INPUT:
        x.shape(time, dim)
    '''
    assert len(x.shape) > 1, 'len(x.shape) < 1'
    # return np.dot(x, x.T).diagonal().mean() //
    return (x * x).sum(axis=1).mean()


def shuffle_in_time(x):
    '''
    shuffles in time but keeping values of same times together
    INPUT:
        x.shape(time, dim)
    '''
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return 1 * x[indices]


def minusMean_vecTS(x):
    assert len(x.shape) > 1, 'len(x.shape) < 1'
    return x - x.mean(axis=0)[None, :]


def corrCoef1d(x, y):
    return np.corrcoef(x, y=y)[0, 1]


def corrCoef2d(x, y):
    '''
    create variance of the norm of a vector timeseries
    motivated by [Spatio-temporal correlations in models...., Cavagna et. al., Physical Biology, 2016]
    INPUT:
        x.shape(time, dim)
        x.shape(time, dim)
    '''
    cov = (x * y).sum(axis=1).mean()
    norm = np.sqrt(var_norm_vecTS(x) * var_norm_vecTS(y))
    return cov/norm


def lag_corr(x, y, maxlag, quantile=None, sam=None, plot=True,
             saveplot=False, bootstrap=True, axs=None):
    '''
    computes the lagged correlation C(t) between x,y and y,x
    x and y are assumed to be either 1dimensional or 2 dimensional (directions, velocity....)
    AND bootstrapp the correlation between them with "samples" samples 
    AND plots the results if wanted
    INPUT:
        x.shape(time) OR x.shape(time, 2)
        y.shape(time) OR y.shape(time, 2)
        maxlag int
            defines the maximum lag
        quantile float
            defines the quantile computed from the bootstrapping
        sam int
            number of sample used for bootstrapping
        plot bool
            defines if the results are supposed to be plotted
        saveplot bool
    OUTPUT:
        out = [C, lower, upper]
            C.shape(2, maxlags)
                Correlation between x(t), y(t+lag) and x(t+lag), y(t)
                for different lags
            lower float
            upper float
                lower and upper bounds from bootstrapping

    '''
    if quantile == None:
        quantile = 0.025
    if sam == None:
        sam = 1001
    assert x.shape == y.shape, 'x.shape != y.shape'
    if len(x.shape) == 1:
        minusMean = lambda x: x - x.mean()
        corrCoef = corrCoef1d
    elif len(x.shape) == 2:
        minusMean = minusMean_vecTS
        corrCoef = corrCoef2d
    else:
        assert 1 == 2, 'lag_corr only for 1 or 2d arrays'
    maxlag = int(maxlag) + 1
    sam = int(sam)
    time = len(x)
    out = []
    # standardize the timeseries:
    xx = minusMean(x)
    yy = minusMean(y)

    # do correlation for different lags:
    C = np.empty((2, maxlag))
    for i in range(maxlag):
        xl = xx[:time-i]
        yl = yy[i:]
        C[0, i] = corrCoef(xl, yl)
        xl = xx[i:]
        yl = yy[:time-i]
        C[1, i] = corrCoef(xl, yl)
    out.append(C)
    if bootstrap:
        lower, upper = bootstrapp_corr(x, y, quantile, sam)
        out.append(lower)
        out.append(upper)
    if plot:
        f, axs = plt.subplots(1, figsize=plt.figaspect(1/2))
        axs.scatter(range(maxlag), C[0], color='r', label=r'$C(x(t), y(t+\tau))$')
        axs.scatter(range(maxlag), C[1], color='b', label=r'$C(y(t), x(t+\tau))$')
        axs.set_xlabel(r'$\tau$ (lag)', fontsize=20)
        axs.set_ylabel(r'$C(\tau)$', fontsize=20)
        if bootstrap:
            axs.hlines([lower, upper], 0, maxlag)
        axs.yaxis.set_tick_params(labelsize=20)
        axs.xaxis.set_tick_params(labelsize=20)
        axs.legend(fontsize=20)
        if saveplot:
            f_name = Path.cwd() / 'LaggedCorrelation.png'
            f.savefig(str(f_name), dpi=200)
    return out


def bootstrapp_corr(x, y, quantile, sam):
    '''
    bootstrapp the correlation between 2 TS based on 
    "samples" samples and returns the 2 quantiles
    '''
    samples = int(sam)
    assert samples > 1000, 'larger sample size is recommended (>=1000)'
    assert quantile < 1, 'quantile must be lower 1'
    assert quantile > 0, 'quantile must be larger 0'
    assert x.shape == y.shape, 'x.shape != y.shape'
    if len(x.shape) == 1:
        corrCoef = corrCoef1d
        minusMean = lambda x: x - x.mean()
    elif len(x.shape) == 2:
        corrCoef = corrCoef2d
        minusMean = minusMean_vecTS
    res = np.zeros(samples, dtype='float')
    x = minusMean(x)
    y = minusMean(y)
    for i in range(samples):
        xx = shuffle_in_time(x)
        yy = shuffle_in_time(y)
        res[i] = corrCoef(xx, yy)
    lower = np.sort(res)[int(samples * quantile)]
    upper = np.sort(res)[samples - int(samples * quantile)]
    return [lower, upper]
