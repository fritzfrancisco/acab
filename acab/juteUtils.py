import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
# from skimage import measure

import pdb
import pandas as pd
# from numba import njit
import shutil
from numpy.linalg import lstsq


def setDefault(x, val):
    if x is None:
        x = val
    return x


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


######################################################################################################
# blobs
######################################################################################################
# measure.label doesnt work with a huge amount of blobs
# so we use this helper functions
# also checkable in (Blob_SplitAndMerge.ipynb

def JoinLabels(lab0, lab1):
    '''
    joins labels of adjacent labeled parts via an overlapping
    section in the first dimension (usually time)
    TODO: probably better way of implementing it
        instead of while loop
    "Problem":
        some labels are skipped, if a label of lab0 is renamed a label-skip appears
        That is not an error but some integers do not have a label
        e.g. np.unique(result) = [0,1,3,4] # integer 2 is missing BUT NO Labeled blob is missing
    '''
    newLabelIds1 = list(np.unique(lab1[0]))
    constAdd = max([lab1.max(), lab0.max()]) + 1
    for Id in newLabelIds1:
        there = np.where(lab1[0] == Id)
        oldId = 1 * lab0[-1, there[0][0], there[1][0]]
        lab1[lab1 == Id] = oldId + constAdd
    # check the other way around:
    renamed_oldId = []
    renamed_newId = []
    diff = lab0[-1] - (lab1[0] - constAdd) # should be everywhere 0 unless missed blob
    diffs = np.unique(diff)
    count = 0
    while len(diffs) > 1:
        for di in diffs:
            if di != 0:
                there = np.where(diff == di)
                oldId = 1 * lab0[-1, there[0][0], there[1][0]]
                newId = 1 * lab1[0, there[0][0], there[1][0]]
                if oldId not in renamed_oldId:
                    newId -= constAdd
                    lab0[lab0 == oldId] = newId
                    renamed_oldId.append(newId)
                elif newId not in renamed_newId:
                    lab1[lab1 == newId] = oldId + constAdd
                    renamed_newId.append(newId)
                else:
                    # rename in lab0: old to new-const AND in lab1: old +const to new
                    lab0[lab0 == oldId] = newId - constAdd
                    lab1[lab1 == oldId + constAdd] = newId
        diff = lab0[-1] - (lab1[0] - constAdd) # should be everywhere 0 unless missed blob
        diffs = np.unique(diff)
        # print('critical counter ', count, '/ 10')
        count += 1
        assert count <= 20, "Too many iterations... probably something wrong. diff: {}, Ids: {}, {}, {}".format(diffs, oldId, newId, constAdd)
    # change non-overlapping blobs
    count = 1
    maxOldId = lab0.max()
    newLabelIds1 = np.unique(lab1[0])
    newLabelIdsAll = np.unique(lab1)
    for Id in newLabelIdsAll:
        if Id not in newLabelIds1:
            lab1[lab1 == Id] = maxOldId + count + constAdd
            count += 1
    lab1[lab1 == 0] = constAdd
    lab1 -= constAdd
    result = np.concatenate((lab0, lab1[1:]), axis=0)
    # now ensure that no label is skipped:
    return result


def JoinLabelsNOTWORKING(lab0, lab1):
    '''
    See JoinLabels: tried to write it more simple.... not working....
    '''
    Overlap0, Overlap1 = 1*lab0[-1], 1*lab0[0]
    overIds0 = np.unique(Overlap0)
    allAssigned1 = np.empty(0)
    constAdd = max([np.max(lab0), np.max(lab1)]) + 1
    Ids1To0 = dict()
    Ids0Merges = dict() # key =
    for Id0 in overIds0:
        there = np.where(Overlap0 == Id0)
        assignedIds1 = np.unique(Overlap1[there])
        alreadyAsigned = np.in1d(assignedIds1, allAssigned1)
        if np.any(alreadyAsigned): # already assigned -> change also lab0-ids 
            alreadyAssignedIds = assignedIds1[alreadyAsigned]
            Id0NeedsMerge = np.unique([Ids1To0[i1] for i1 in alreadyAssignedIds])
            Id0 = Id0NeedsMerge[0]
            if len(Id0NeedsMerge) > 1:
                for id2merge in Id0NeedsMerge[1:]:
                    Ids0Merges[id2merge] = Id0
        for Id1 in assignedIds1:
            Ids1To0[Id1] = Id0
        allAssigned1 = np.append(allAssigned1, assignedIds1)
        # for Id1 in assignedIds1: # ATTENTION
        #     Ids1To0[Id1] = Id0
        #     lab1[lab1 == Id1] = Id0 + constAdd
    # now assign new labels to blobs outside of overlap
    AllIds1 = np.unique(lab1)
    remainingIds = np.setdiff1d(AllIds1, allAssigned1)
    maxId0 = np.max(lab0)
    for i, Id1 in enumerate(remainingIds):
        lab1[lab1 == Id1] = maxId0 + i + constAdd + 1
    # change assigned id1-labels from the overlap
    for Id1 in Ids1To0.keys():
        lab1[lab1 == Id1] = Ids1To0[Id1] + constAdd
    # merge id0-labels from the overlap
    for Id0 in Ids0Merges.keys():
        lab0[lab0 == Id0] = Ids0Merges[Id0]
    lab1 -= constAdd
    # ensure that everything works
    diff = (lab0[-1] - lab1[0]).flatten() # should be everywhere 0 unless missed blob
    assert len(np.unique(diff)) == 1 and diff[0] == 0, 'non-zero entires in diff[0]: {}, np.unique(diff) {}'.format(diff[0], np.unique(diff))
    # join the datasets
    result = np.concatenate((lab0, lab1[1:]), axis=0)
    return result


def labelSplitAndMerge(data, split=None, background=None):
    '''
    Does EXACTLY the same as skimage.measure.label but for much larger data
        If data is too large the skimage.measure.label throws segmentation fault
        This fctn. splits the data in smaller parts runs skimage.measure.label
        on them and merges them together
    ATTENTION chose Split as large as possible, because with each new chunk it merges it the whole past chunks with the new chunk (-> more time needed)
    INPUT:
        data len(shape)=[time, Nx, Ny]
            numpy array with first dimension as time the other 2 are
            space dimensions
    '''
    if split is None:
        split = 100
    time, *_ = data.shape
    borders = np.arange(split, time, step=split)
    # initialize
    dat = data[:borders[0]+1]
    partLabel = partial(measure.label, background=background)
    labelAll = partLabel(dat)
    # merge all but last
    allborders = len(borders)
    for i in range(1, allborders):
        print('time to merge with border ', i, ' / ', allborders, end='\r')
        dat = data[borders[i-1]:borders[i]+1]
        label = partLabel(dat)
        labelAll = JoinLabels(labelAll, label)
    # merge last:
    if borders[-1]+1 != time:
        print('last merge')
        dat = data[borders[-1]:]
        label = partLabel(dat)
        labelAll = JoinLabels(labelAll, label)
    return labelAll


def bin2d(dat, dat2, bins=None, func=None, equalWeight=None):
    '''
    similar to plt.hist2d but computes func(=mean, or median or ...)
    for equally sized  bins BASED ON 'dat' (first input)
        if equalWeight=True: bins are equally space
    INPUT:
        dat
        dat2
        bins
        func(array) -> float
            function which transforms array of arbitrary shape -> float
    OPTIONAL INPUT:
        equalWeight bool
            True: each bin contains same amount of data points
            False: equally distant bins, and therefore can vary in their data content
    '''
    # DEFAULTS-START
    bins = setDefault(bins, int(len(dat)/2))
    func = setDefault(func, np.nanmean)
    equalWeight = setDefault(equalWeight, False)
    # DEFAULTS-END
    dat = np.array(dat).flatten()
    dat2 = np.array(dat2).flatten()
    if equalWeight:
        indicesOfBin = Indices_EqualParts(dat, bins)
    else:
        indicesOfBin = Indices_EquidistantBins(dat, bins)
    # output container
    dat_mean = np.empty(bins) * np.nan
    dat2_mean = np.empty(bins) * np.nan
    Ndat = np.zeros(bins)
    for i, there in enumerate(indicesOfBin):
        if len(there) > 0:
            dat_mean[i] = func(dat[there])
            dat2_mean[i] = func(dat2[there])
            Ndat[i] = len(there)
    return dat_mean, dat2_mean, Ndat


def Indices_EqualParts(data, Npart):
    '''
    split data in "Npart" equal parts and return indices
    of the parts ordered such that parts with smallest values
    is first and part with largest values last
    INPUT:
        data shape=(time)
        Npart int
            number of equal parts the data is split to 
    OUTPUT:
        splitted_data len=Npart
            contains same data 
    '''
    borders = np.linspace(0, 1, Npart+1)[1:-1]  # no 0th, 100th perc.
    borders = np.percentile(data, borders*100)
    splitted_data = []
    # first split
    there = np.where(data <= borders[0])[0]
    splitted_data.append(there)
    # all splits between first and last
    for i in range(0, len(borders)-1):
        there = np.where((data > borders[i]) & (data <= borders[i+1]))[0]
        splitted_data.append(there)
    # last split
    there = np.where(data > borders[-1])[0]
    splitted_data.append(there)
    return splitted_data


def Indices_EquidistantBins(data, Nbins):
    '''
    split data in "Nbins" bins.
    The bins have the same width
    -> the number of data-points in the bin will vary.
    1st. Part contains indices with smallest data
    last Part contains indices with largest data
    INPUT:
        data shape=(time)
        Nbins int
            number of bins the data is assigned to
    OUTPUT:
        splitted_data len=Nbins
            contains same data 
    '''
    dat = np.array(data)
    minn = np.nanmin(dat)
    maxx = np.nanmax(dat)
    dat_mean = np.empty(Nbins) * np.nan
    borders = np.linspace(minn, maxx, Nbins+1)
    indicesOfBin = []
    for i in range(Nbins):
        lb = borders[i]
        ub = borders[i+1]
        there = np.where((lb<dat) & (dat<=ub))[0]
        indicesOfBin.append(there)
    return indicesOfBin


def angle_between_angles(phi0, phi1):
    '''
    returns the smaller angle between two angles (phi0, phi1)
    '''
    angleBetween = np.abs(phi0 - phi1) # actually the abs is not necessary
    angleBetween = np.mod(angleBetween + np.pi, 2*np.pi) - np.pi
    return np.abs(angleBetween)


def standardize(dat, minn=None, maxx=None):
    if minn is None:
        minn = np.min(dat)
    if maxx is None:
        maxx = np.max(dat)
    dat -= minn
    dat /= maxx - minn
    return dat


def treeDict(dic, level=0, name='', maxKeys=5):
    '''
    outputs the structure of nested dictionaries in a tree-like manner
    '''
    string0 = '  '*level + '-{}: '.format(name)
    if type(dic) == dict:
        ks = list(dic.keys())
        if len(ks) > maxKeys:
            string0 += 'Showing only {}[0] of total {} keys ({})'.format(name, len(ks), ks)
            ks = [ks[0]]
        print(string0, ks)
        for k in ks:
            treeDict(dic[k], level=level+1, name=k, maxKeys=maxKeys)
    elif hasattr(dic, 'shape'):
        print(string0, 'shape=', dic.shape)
    elif type(dic) in [list, tuple]:
        print(string0, 'length={}, {}[0]={}'.format(len(dic), name, dic[0]))
    else:
        print(string0, dic)


def funcBetweenPercentiles(func, percentiles, dat):
    '''
    computes the function "func(dat_p1p2)" with
    dat_p1p2 as all datapoints between the two
    percentiles  (= "percentiles[0]", "percentiles[1]") of "dat"
    '''
    perc = np.percentile(dat, percentiles)
    there = np.where((dat >= perc[0]) & (dat <= perc[1]))[0]
    return func(dat[there])


def gaussian_filter_nans(U, sigma, order=0, output=None,
                         mode='reflect', cval=0.0, truncate=4.0):
    '''
    applies scipy.ndimage.gaussian_filter but without
    losing values due to Nans by applying "normalized convolution"
    source:"https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python"
    Better Explanation: "http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#SECTION00020000000000000000"
    Note: It is not strictly a "normalized convolution"
            since we are discarding all resulting averages 
            where NANs where fouund before
          -> to make it proper "normalized convolution" out-comment "Z[np.isnan(U)] = 0"

    INPUT:
        same as for gaussian_filter
    '''
    gau_filter = partial(ndimage.gaussian_filter, order=order, output=output,
                         mode=mode, cval=cval, truncate=truncate)
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = gau_filter(V, sigma)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = gau_filter(W, sigma)

    Z = VV / WW
    Z[np.isnan(U)] = 0
    return Z


def smooth2D(dat, std, min_periods=None, smotype=None):
    assert len(dat.shape) == 2, 'data has wrong dimension' 
    smodat = np.empty(dat.shape, dtype=float)
    for i, da in enumerate(dat.T):
        smodat[:, i] = smooth1D(da, std, min_periods=min_periods,
                                smotype=smotype)
    return smodat


def smooth1D(dat, std=None, window=None, min_periods=None, smotype=None):
    '''
    INPUT:
        dat.shape(T)
        std double
            - standard deviation of gaussian kernel used
            - if window=None: int(6*std) = window size
        window int
            number of datapoints in a moving window
        min_periods int
            - minimum # of datapoints in window
            - if less data than min_periods -> None
        smotype string
            up to now only 'gaussian' is implemented
    '''
    std = setDefault(std, 1)
    window = setDefault(window, int(np.round(6*std)))
    min_periods = setDefault(min_periods, 1)
    smotype = setDefault(smotype, 'gaussian')
    # use pandas to smooth
    smodat = pd.Series(dat)
    smodat = smodat.rolling(window=window, win_type=smotype,
                            center=True, min_periods=int(np.round(min_periods))
                           ).mean(std=std)
    return smodat


def copyIfNeeded(src, dst):
    '''
    copies file or directory to 'dst' if it does not exist
    otherwise
    do not copy
    INPUT:
        src str
            file tobe copied
        dst str
            directory in which the file/dir to be copied
    '''
    p_src = Path(src)
    p_dst = Path(dst) / p_src.parts[-1]
    if not p_dst.exists():
        if p_src.is_dir():
            shutil.copytree( str(p_src), str(p_dst) )
        elif p_src.is_file():
            shutil.copy( str(p_src), str(dst) )


def silentRemove(f_name):
    '''
    for deleting-review see: https://linuxize.com/post/python-delete-files-and-directories/
    TODO:
        if non-empty directory
            shutil.rmtree
            symbolic link = FALSE
        if directory
           pathlib.Path.rmdir
            symbolic link = ????
        if file
           pathlib.Path.unlink
            symbolic link = OK
    '''
    try:
        os.remove(f_name)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def compareDicts(dic0, dic1, excludeTypes=None):
    '''
    compares the values of 2 different dictionaries and the keys
    '''
    excludeTypes = setDefault(excludeTypes, [])
    k0 = np.array(list(dic0.keys()))
    k1 = np.array(list(dic1.keys()))
    k0in1 = np.in1d(k0, k1)
    k1in0 = np.in1d(k0, k1)
    if np.sum(~k0in1) > 0:
        print('dic0-keys: {} are not in dic1'.format(k0[~k0in1]))
    if np.sum(~k1in0) > 0:
        print('dic1-keys: {} are not in dic0'.format(k1[~k1in0]))
    checkKeys = k0[k0in1]
    checkKeys.sort()
    for k in checkKeys:
        val0, val1 = dic0[k], dic1[k]
        if type(val0) not in excludeTypes and val0 != val1:
            print('conflict in {}: {} != {}'.format(k, val0, val1))

def SegmentedLinearReg(X, Y, breakpoints):
    '''
    piecewise regression, or segmented regression which tunes the breakpoints
    -> if you know the breakpoint and want not change it -> do not use
    INPUT:
        X
    Example:
        import matplotlib.pyplot as plt
        
        X = np.linspace( 0, 10, 27 )
        Y = 0.2*X  - 0.3* ramp(X-2) + 0.3*ramp(X-6) + 0.05*np.random.randn(len(X))
        plt.plot( X, Y, 'ok' );
        
        initialBreakpoints = [1, 7]
        plt.plot( *SegmentedLinearReg( X, Y, initialBreakpoints ), '-r' );
        plt.xlabel('X'); plt.ylabel('Y');

    BASED ON 
        [1]: Muggeo, V. M. (2003). Estimating regression models with unknown breakpoints. Statistics in medicine, 22(19), 3055-3071.

    COPIED FROM
        https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression
    '''
    nIterationMax = 10

    ramp = lambda u: np.maximum( u, 0 )
    step = lambda u: ( u > 0 ).astype(float)

    breakpoints = np.sort( np.array(breakpoints) )

    dt = np.min( np.diff(X) )
    ones = np.ones_like(X)

    for i in range( nIterationMax ):
        # Linear regression:  solve A*p = Y
        Rk = [ramp( X - xk ) for xk in breakpoints ]
        Sk = [step( X - xk ) for xk in breakpoints ]
        A = np.array([ ones, X ] + Rk + Sk )
        p =  lstsq(A.transpose(), Y, rcond=None)[0] 

        # Parameters identification:
        a, b = p[0:2]
        ck = p[ 2:2+len(breakpoints) ]
        dk = p[ 2+len(breakpoints): ]

        # Estimation of the next break-points:
        newBreakpoints = breakpoints - dk/ck 

        # Stop condition
        if np.max(np.abs(newBreakpoints - breakpoints)) < dt/5:
            break

        breakpoints = newBreakpoints
    else:
        print( 'maximum iteration reached' )

    # Compute the final segmented fit:
    Xsolution = np.insert( np.append( breakpoints, max(X) ), 0, min(X) )
    ones =  np.ones_like(Xsolution) 
    Rk = [ c*ramp( Xsolution - x0 ) for x0, c in zip(breakpoints, ck) ]

    Ysolution = a*ones + b*Xsolution + np.sum( Rk, axis=0 )

    return Xsolution, Ysolution
