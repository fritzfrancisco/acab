
import gc
import glob
import itertools
import os
import sys
import random
import uuid
import cv2
import _pickle
import h5py
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.animation as animation

from numba import jit
from copy import deepcopy

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from multiprocessing import Pool, get_context

from scipy.interpolate import SmoothSphereBivariateSpline, griddata
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import binned_statistic_2d, binned_statistic, gaussian_kde
from scipy.signal import cwt, morlet, ricker, savgol_filter, correlate
from scipy.spatial.distance import cdist, directed_hausdorff, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import interp1d

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def calculate_katz(focal_id, reference_id, to_origin=False, origin=[0, 0, 0]):
    '''create spherical heatmap based on occurences of reference to focal signal.
    If to_origin is set to True, reference input is ignored and focal coordinates
    are calculated in reference to [0,0,0] origin.'''
    
    if to_origin == True:
        reference_id = {}
        for key in focal_id.keys():
            if np.isin(key, ['X', 'Y', 'Z']) == True:
                idx = int(np.where(np.array(['X', 'Y', 'Z']) == key)[0])
                reference_id[str(key)] = np.repeat(origin[idx],
                                                   len(focal_id[key]))
            else:
                reference_id[str(key)] = focal_id[key]

        focal = focal_id
        ref = reference_id

    else:

        index = np.intersect1d(focal_id['FRAME_IDX'],
                               reference_id['FRAME_IDX'])
        focal_index = np.array(
            [1 if np.isin(f, index) else 0 for f in focal_id['FRAME_IDX']],
            dtype=bool)
        ref_index = np.array(
            [1 if np.isin(f, index) else 0 for f in reference_id['FRAME_IDX']],
            dtype=bool)

        focal = {}
        for key in focal_id.keys():
            focal[key] = focal_id[key][focal_index]

        ref = {}
        for key in reference_id.keys():
            ref[key] = reference_id[key][ref_index]

    sdata = circle_coordinates(focal, ref)

    return sdata[:, 0], sdata[:, 1], sdata[:, 2], sdata[:, 3], sdata[:, 4]


def circle_coordinate(focal_pt, ref_pt):
    '''create azimuth and plunge from two signals (focal and refence)'''
    
    distance = np.sqrt((ref_pt[0] - focal_pt[0])**2 +
                       (ref_pt[1] - focal_pt[1])**2 +
                       (ref_pt[2] - focal_pt[2])**2)

    plunge = np.degrees(np.arcsin((ref_pt[2] - focal_pt[2]) / distance))
    azimuth = np.degrees(
        np.arctan2((ref_pt[0] - focal_pt[0]), (ref_pt[1] - focal_pt[1])))

    r = 1
    x = r * np.sin(plunge) * np.cos(azimuth)
    y = r * np.sin(plunge) * np.sin(azimuth)
    z = r * np.cos(plunge)
    return x, y, z, azimuth, plunge


def circle_coordinates(focal_id, ref_id):
    '''create spherical coordinates from two reference signals'''
    
    focal_pts = np.array([
        (focal_id['X'][i], focal_id['Y'][i], focal_id['Z'][i])
        for i, f in enumerate(focal_id['FRAME_IDX'])
    ]).reshape(-1, 3)

    ref_pts = np.array([(ref_id['X'][i], ref_id['Y'][i], ref_id['Z'][i])
                        for i, f in enumerate(ref_id['FRAME_IDX'])
                        ]).reshape(-1, 3)

    assert (len(focal_pts) == len(ref_pts)
            ), "Focal and reference array don't have same length!"

    spherical_data = np.array(
        [circle_coordinate(f, ref_pts[n]) for n, f in enumerate(focal_pts)])
    return spherical_data


def spherical_heatmap(theta, phi, bins=10, ax=None):
    '''create a spherical heatmap based on spherical 
    theta and phi and in the resolution defined by bins'''

    # Creating the theta and phi values.
    intervals = bins
    ntheta = intervals
    nphi = 2 * intervals

    grid_theta = np.linspace(0, np.pi * 1, ntheta + 1)
    grid_phi = np.linspace(0, np.pi * 2, nphi + 1)

    # Creating the coordinate grid for the unit sphere.
    X = np.outer(np.sin(grid_theta), np.cos(grid_phi))
    Y = np.outer(np.sin(grid_theta), np.sin(grid_phi))
    Z = np.outer(np.cos(grid_theta), np.ones(nphi + 1))

    # Creating a 2D array to be color-mapped on the unit sphere.
    # {X, Y, Z}.shape → (ntheta+1, nphi+1) but c.shape → (ntheta, nphi)
    c = np.zeros((ntheta, nphi))

    # binning data
    bin_means = binned_statistic_2d(theta,
                                    phi,
                                    None,
                                    'count',
                                    bins=[ntheta, nphi]).statistic
    nan_index = np.isnan(bin_means)
    bin_means[nan_index] = 0
    bin_means = gaussian_filter(bin_means, sigma=1)

    # normalize and standardize data
    #     bin_means *= bin_means / np.max(bin_means)
    #     bin_means = bin_means + np.abs(np.min(bin_means))
    #     bin_means = bin_means * (np.max(bin_means)**-1)

    c = bin_means / bin_means.max()

    # The poles are different
    #     c[ :1, :] = 0.8
    #     c[-1:, :] = 0.8
    # as well as the zones across Greenwich
    # c[:,  :1] = 0.0
    # c[:, -1:] = 0.0

    # Creating the colormap thingies.
    cm = mpl.cm.jet
    sm = mpl.cm.ScalarMappable(cmap=cm)
    sm.set_array([])

    # Creating the plot.
    if ax == None:
        fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
        ax = fig.gca(projection='3d')
        fig.colorbar(sm, ax=ax)

    ax = ax.plot_surface(X,
                         Y,
                         Z,
                         rstride=1,
                         cstride=1,
                         facecolors=cm(c),
                         alpha=0.5,
                         shade=False,
                         vmin=0,
                         vmax=1,
                         zorder=10)
    plt.colorbar(sm, fraction=0.011, pad=0.01)
    return ax


def asCartesian(r, theta, phi):
    '''convert spherical coordinates to x,y,z coordinates'''
    r = r
    theta = theta * np.pi / 180  # to radian
    phi = phi[2] * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def asSpherical(x, y, z):
    '''convert x,y,z coordinates to spherical coordinates'''
    
    x = x
    y = y
    z = z
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r) * 180 / np.pi  #to degrees
    phi = np.arctan2(y, x) * 180 / np.pi
    return r, theta, phi


def rotate(angle):
    '''rotation fuction required for plotting rotating plots'''
  
    ax.view_init(azim=angle)


def create_windows(arr, window_size=10):
    '''Split array into windows according to window_size. 
    Windows on both ends of the array are padded with np.nan to fit window_size.
    Window_size musst be even'''
    
    windows = []
    assert window_size % 2 == 0, "window_size must be even!"
    for i, f in enumerate(arr):
        if i <= int(window_size / 2):
            pad = np.repeat(np.nan, int(window_size / 2) - i)
            window = arr[:i + int(window_size / 2)]
            window = np.concatenate([pad, window])
        elif i >= int(len(arr) - (window_size / 2)):
            pad = np.repeat(np.nan, i - (len(arr) - (window_size / 2)))
            window = arr[i - int(window_size / 2):]
            window = np.concatenate([window, pad])
        else:
            window = arr[i - int(window_size / 2):i + int(window_size / 2)]
        windows = np.append(window, windows).reshape(-1, window_size)
    return windows


def get_speed(focal_id, window=11, smooth=False):
    '''calculate speed with cartesian x,y,z coordinates'''

    if 'Z' in focal_id.keys():
        speed = np.sqrt((np.diff(focal_id['X']))**2 +
                        (np.diff(focal_id['Y']))**2 +
                        (np.diff(focal_id['Z']))**2)
    else:
        speed = np.sqrt((np.diff(focal_id['X']))**2 +
                        (np.diff(focal_id['Y']))**2)
    speed = np.append(speed, speed[-1])
    if smooth == True:
        assert window % 2 == 1, 'window length musst be uneven'
        speed = savgol_filter(speed, window, 3)
    return speed


def cooccurrence_index(focal_id, reference_id):
    '''retrieve index of frames in which both signals exist'''
    
    index = np.intersect1d(focal_id['FRAME_IDX'], reference_id['FRAME_IDX'])
    focal_index = np.array(
        [1 if np.isin(f, index) else 0 for f in focal_id['FRAME_IDX']],
        dtype=bool)
    ref_index = np.array(
        [1 if np.isin(f, index) else 0 for f in reference_id['FRAME_IDX']],
        dtype=bool)
    return focal_index, ref_index

@jit(nopython=True, parallel=True)
def correlate_windows(arr1, arr2, tau=50):
    '''small scale correlation of two signals based on a sliding window tau.'''
    
    values = np.zeros(len(arr1))
    lags = np.zeros(len(arr1))
    leads = np.zeros(len(arr1))
    windows = np.zeros((len(arr1),tau*2))
    lag_arr = np.linspace(-0.5 * len(arr1) / tau, 0.5 * len(arr1) / tau,
                            len(arr1))
    lead_arr = np.linspace(-0.5 * len(arr1) / tau, 0.5 * len(arr1) / tau,
                            len(arr1))
    i_count = 0
    for u in arr1:
        window = np.zeros(len(np.arange(i_count - tau, i_count + tau)))
        j_count = 0
        for v in arr2[i_count - tau:i_count + tau]:
            window[j_count] = np.dot(u, v)
            j_count += 1
        windows[i_count] = window
        values[i_count] = np.argmax(window)
        delay = lag_arr[np.argmax(window)]
        lead = lead_arr[np.argmin(window)]
        lags[i_count] = delay
        leads[i_count] = lead
        
        i_count += 1
        
    return windows, values, lags, leads



def read_csv(file):
    data = pd.read_csv(file)
    tracks = {}
    for i, key in data.iteritems():
        if key.name in [
                'frame', 'pos_x', 'pos_y', 'identity', 'tile_center',
                'tile_radius', 'cylinder_center', 'cylinder_radius',
                'frame_height', 'frame_width', 'arena_corner', 'arena_height',
                'arena_width'
        ]:
            tracks[str(key.name)] = np.array(data[i])
    del data
    return tracks


def read_csv2np(file):
    gc.collect()
    data = pd.read_csv(file)
    output = {}
    for i, key in data.iteritems():
        output[str(key.name)] = np.array(data[i])
    del data
    return output


def read_schedule(file):
    schedule = read_csv2np(file)
    output = schedule.copy()
    dates = [
        str(i[:2] + '0' + i[3:]) if len(i) != 10 else i
        for i in schedule['DATE']
    ]
    dates = [
        str(date[-4:] + '-' + date[-7:-5] + '-' + date[:2]) for date in dates
    ]
    times = [time.replace(':', '') for time in schedule['TIME']]
    output['TIME'] = times
    output['DATE'] = dates
    del schedule
    return output


@jit
def distance(x1, x2, y1, y2):
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def remove_duplicates(tracks):
    '''remove duplicated points from tracks'''
    _, x_index = np.unique(tracks['pos_x'], return_index=True)
    _, y_index = np.unique(tracks['pos_y'], return_index=True)
    index = np.intersect1d(x_index, y_index)

    for key in tracks:
        tracks[key] = tracks[key][index]
    return tracks


def interpolate_tracks(tracks,
                       smooth_window=9,
                       n_iter=3,
                       degree=1,
                       masked=False):
    '''interpolate tracks that are in the format used in Francisco & Nuehrenberg 2020.'''
    
    frame_idx = np.arange(tracks['frame'][0], tracks['frame'][-1] + 1)
    if masked:
        mask = deepcopy(tracks['frame'])
    for key in tracks:
        if key in ['pos_x', 'pos_y']:
            interp_key = interp1d(tracks['frame'], tracks[key])
            tracks[key] = interp_key(frame_idx)
    tracks['frame'] = frame_idx
    if smooth_window > 0:
        for key in tracks:
            if key in ['x', 'y']:
                for n in range(n_iter):
                    tracks[key] = savgol_filter(tracks[key], smooth_window,
                                                degree)
    if masked:
        for key in tracks:
            tracks[key] = tracks[key][np.isin(frame_idx, mask)]
    tracks['frame'] = np.unique(tracks['frame']).astype(np.int)
    return tracks


def plot_tracks(tracks):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(tracks['pos_x'], tracks['pos_y'], s=5, marker='.', alpha=0.1)
    ax[1].plot(tracks['pos_x'], tracks['pos_y'])
    for i in range(len(ax)):
        ax[i].axis('equal')
        ax[i].axis('off')
    plt.show()


def get_direction(tracks):
    '''Function to calculate direction for each frame from x- and y-coordinates'''
    
    direction = np.arctan2(np.diff(tracks['pos_y']), np.diff(tracks['pos_x']))
    tracks['direction'] = direction
    return tracks


def get_dist_2_rois(tracks, roi_centers):
    '''Fuction to calculate distances to rois of a single individual for each frame from x- and y-coordinates. 
    Output has same length as roi_centers input'''
    
    xy = [pt for pt in zip(tracks['pos_x'], tracks['pos_y'])]
    dist_list = []
    for center in roi_centers:
        dist = [
            distance(xy[i][0], center[0], xy[i][1], center[1])
            for i, pt in enumerate(xy)
        ]
        dist_list.append(dist)
    dist_list = np.array(dist_list)
    return dist_list


def get_windows(tracks):
    ''' Function to create sliding window, centered around a given frame.
    Padded with np.nan, if the window exceeds the array '''

    windows = {}
    index_list = np.arange(len(tracks['pos_x']))

    for window in np.geomspace(1, 100, 11)[1:]:
        window_list = []
        for c, d in enumerate(tracks['pos_x']):
            index = []
            for w in range(int(window)):
                if c < w:
                    index.append(np.nan)
                elif c > (len(tracks['pos_x']) - w):
                    index.append(np.nan)
                else:
                    index.append(c + w)

            window_list.append(index)
        windows[str(int(window))] = window_list
    tracks['windows'] = windows
    return tracks


def reject_outliers(data, m):
    """
    This function removes any outliers from presented data.

    Parameters
    ----------
    data: pandas.Series
        a column from a pandas dataframe that needs smoothing
    m: float
        standard deviation cutoff beyond which, datapoint is considered as an outlier

    Returns
    -------
    index: ndarray
        an array of indices of points that are not outliers
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return np.where(s < m)


def save(dump, file_name):
    '''Save to a .pkl file
    Parameters
    ----------
    dump : object
        Python object to save
    file_name : str
        File path of saved object
    Returns
    -------
    bool
        Successful save?
    '''

    with open(file_name, 'wb') as fid:
        _pickle.dump(dump, fid)
    return True


def load(file_name):
    '''Loads a python object from a .pkl file
    Parameters
    ----------
    file_name : str
        File path of saved object
    Returns
    -------
    object
        Loaded python object
    '''

    with open(file_name, 'rb') as fid:
        dump = _pickle.load(fid)
    return dump


def filter_tracks(tracks, conversion_factor, iterations=10, threshold=0):
    ''' Filters xy coordinate tracks based on speed threshold in order to remove outliers.
    This is done iteratively to remove as many erroneous points as possible. 
    Parameters
    ----------
    conversion_factor: float
        px/cm conversion
    iterations: int
        number of interations 
    threshold: float
        percentile at which to cut-off
    Returns
    -------
    object 
        filtered tracks object
    '''

    ## TO-DO: Set fixed threshold to standardize over trials!

    gc.collect()
    #     tracks = deepcopy(tracks)
    tracks = clean_tracks(tracks)

    frame_idx = np.arange(tracks['frame'][0], tracks['frame'][-1] + 1)

    for key in ['pos_x', 'pos_y']:
        interp_key = interp1d(tracks['frame'], tracks[key])
        tracks[key] = interp_key(frame_idx)

    tracks['frame'] = frame_idx
    tracks['frame'] = np.unique(tracks['frame']).astype(np.int)

    tracks = get_speed(tracks)

    for i in range(5):
        index = np.array(
            np.where(
                (tracks['speed'] * (30 / conversion_factor) < threshold))[0])
        #         index = np.array(np.where((tracks['speed']*(30/conversion_factor)<np.quantile(tracks['speed']*(30/conversion_factor),threshold)))[0])
        frame_idx = np.arange(tracks['frame'][index][0],
                              tracks['frame'][index][-1] + 1)

        for key in ['pos_x', 'pos_y']:
            interp_key = interp1d(tracks['frame'][index], tracks[key][index])
            tracks[key] = interp_key(frame_idx)

        tracks['frame'] = np.unique(frame_idx).astype(np.int)
        tracks = get_speed(tracks)
    return tracks


def clean_tracks(tracks):
    ''' Fuction derived to remove outliers based on maximum distance (high_distance_filter) 
    and minimum movement (low_distance_filter) between coordinates '''

    gc.collect()
    coords = [tuple(pt) for pt in zip(tracks['pos_x'], tracks['pos_y'])]
    distances = np.array([
        distance(tracks['pos_x'][i], tracks['pos_x'][i + 1],
                 tracks['pos_y'][i], tracks['pos_y'][i + 1])
        for i, pt in enumerate(tracks['frame'][:-1])
    ])
    median = np.median(distances)
    perc = np.percentile(distances, 10)
    stdev = np.std(distances)
    high_distance_filter = np.where((distances > perc))[0]
    index = high_distance_filter

    for key in tracks:
        mask = np.ones(tracks[key].size, dtype=bool)
        mask[index] = False
        tracks[key] = tracks[key][mask]


#     tracks = remove_duplicates(tracks)
    return tracks


def delete_borders(tracks, inside_height, inside_width, height, width):
    ''' collapsing all points to the border, due to perspective distortion
    Parameters
    ----------
    tracks: obj
        track object
    Returns
    -------
    object
        tracks with outliers removed
    '''
    x_ind = []
    y_ind = []

    for i, x in enumerate(tracks['pos_x']):
        if x <= int((width - inside_width) * 0.5):
            x_ind.append(i)
#                     tracks['pos_x'][i] = int((width - inside_width)*0.5)
        if x > int(inside_width + (width - inside_width) * 0.5):
            x_ind.append(i)
#                     tracks['pos_x'][i] = int(inside_width + (width -  inside_width)*0.5)

    for j, y in enumerate(tracks['pos_y']):
        if y <= int((height - inside_height) * 0.5):
            y_ind.append(j)


#                     tracks['pos_y'][j] = int((height - inside_height)*0.5)
        if y > int(inside_height + (height - inside_height) * 0.5):
            y_ind.append(j)
            tracks['pos_y'][j] = int(inside_height +
                                     (height - inside_height) * 0.5)

    del_idx = np.array(np.concatenate([x_ind, y_ind]), dtype=np.int32)

    ## ensure arrays all have same length:
    fixed_values = [
        'tile_center', 'tile_radius', 'cylinder_center', 'cylinder_radius',
        'frame_height', 'frame_width', 'arena_corner', 'arena_height',
        'arena_width'
    ]

    for key in fixed_values:
        if np.isin(tracks.keys(), key, invert=True).any():
            continue
        if tracks[key].size != tracks['frame'].size:
            tracks[key] = np.repeat(tracks[key][0], tracks['frame'].size)

    for key in tracks.keys():
        if np.isin(['speed', 'direction'], key).any():
            continue

        mask = np.ones(tracks[key].size, dtype=bool)
        mask[del_idx] = False
        tracks[key] = tracks[key][mask]

    return tracks


def get_info(file, schedule):
    ''' Function returning all relevant information from csv files and a given schedule'''
    
    f_date = file[97:107]
    f_time = file[108:114]
    f_identity = int(file[-13:-11])

    date_index = [np.array(schedule['DATE']) == f_date][0]
    time_index = [np.array(schedule['TIME']) == f_time[:4]][0]
    id_index = [np.array(schedule['ID'] == f_identity)][0]
    index = np.intersect1d(
        np.where(time_index == True)[0],
        np.where(date_index == True)[0])
    index = np.intersect1d(np.where(id_index == True)[0], index)

    if len(index) != 0:
        species = str(schedule['SPECIES'][index][0])
        treatment = str(schedule['TREATMENT'][index][0][0])
        identity = str(f_identity)
        return species, treatment, identity, f_date
    else:
        return None, None, None


def dir2rois(tracks, centers):
    ''' Calculate directions in reference to given rois '''
    
    directions = []
    for center in centers:
        dv = [
            np.arctan2(((center[1] - xy[1]) / 2), ((center[0] - xy[0]) / 2))
            for xy in zip(tracks['pos_x'], tracks['pos_y'])
        ]
        directions.append(dv)
    return np.array(directions)


def read_filter_return_df(file, idx):
    tracks = read_csv(file)
    tracks = filter_tracks(tracks,
                           conversion_factor=5.2,
                           iterations=1,
                           threshold=30)
    df_out = pd.DataFrame(
        np.array([
            tracks['pos_x'], tracks['pos_y'],
            np.repeat(idx, len(tracks['pos_x'])), tracks['frame']
        ]).astype(np.int).T)
    return df_out


def get_sorted_df(file_list, schedule, ref_id):
    ''' Function return DataFrame with X, Y, FRAME values
    sorted by trial dates according to the given schedule '''
    
    gc.collect()
    files = file_list
    id_trials = []
    print('accumulating data.')
    for j, file in enumerate(files):
        try:
            s, t, i, d = get_info(file, schedule)
            if int(i) == int(ref_id):
                id_trials.append(j)
        except Exception as e:
            print(e)

    id_trials = np.array(id_trials)
    sorted_dates_idx = np.argsort([file[-31:-21] for file in files[id_trials]])
    _, _, dates = get_idxNid(file_list, schedule, s, t)
    df = pd.DataFrame()

    for idx, file in enumerate(files[id_trials][sorted_dates_idx]):
        df_out = read_filter_return_df(
            file,
            int(np.where(np.array([file.find(d) for d in dates]) > 0)[0]))
        df = df.append(df_out)
    print('finished creating dataset!')
    return df


def get_idxNid(file_list, schedule, species, treatment, debug=False):
    ''' Create index for file_list according to species and treatment '''
    
    gc.collect()
    identities = []
    index = []
    dates = []
    files = file_list
    for n, file in enumerate(files):
        try:
            s, t, i, d = get_info(file, schedule)
            if s == species and t == treatment:
                if np.isin(i, identities) == False:
                    identities.append(i)
                index.append(n)
                dates.append(d)
        except Exception as e:
            if debug == True:
                print(e)
    identities = np.array(identities)
    index = np.array(index)
    dates = np.array(sorted(np.unique(dates)))
    return index, identities, dates


def DBSCAN_clustering(tracks):
    '''DBSCAN clustering on xy-coordinates to achieve indentities over cluster association.'''
    
    model = DBSCAN(eps=0.03)
    points = np.array([tracks['pos_x'], tracks['pos_y']]).T
    model.fit(points)
    centers = []
    for i in np.unique(model.labels_):
        centers = np.append([
            np.mean(tracks['pos_x'][model.labels_ == i]),
            np.mean(tracks['pos_y'][model.labels_ == i])
        ], centers)
    centers = centers.reshape(len(np.unique(model.labels_)), 2)
    return model.labels_, centers


def rmv_out_pts(tracks, xmin=100, xmax=300, ymin=0, ymax=1000, absolute=True):
    '''Removing outlier points crudely by setting upper and lower x and y boudaries.'''
    
    if absolute == True:
        x = tracks['pos_x']
        y = tracks['pos_y']
    else:
        x = tracks['pos_x'] * tracks['frame_width'][0]
        y = tracks['pos_y'] * tracks['frame_height'][0]
    x_index = np.where((x > xmin) & (x < xmax))[0]
    y_index = np.where((y > ymin) & (y < ymax))[0]
    index = list(set(x_index) & set(y_index))
    for key in ['pos_x', 'pos_y', 'frame', 'speed', 'direction']:
        try:
            tracks[key] = tracks[key][index]
        except:
            continue
    return tracks


def simple_filter(data, threshold=8, iterations=1):
    '''filter tracks by speed threshold and IF_outlier_removal()'''
    
    for i in range(iterations):
        if (np.isin(np.array(list(data.keys())),
                    'speed').any() == True) == False:
            data = get_speed(data)
        index = np.array(np.where((data['speed'] < threshold))[0])
        frame_idx = np.arange(data['frame'][index][0],
                              data['frame'][index][-1] + 1)

        id_tracks = IF_outlier_removal(data)

        for key in ['speed', 'pos_x', 'pos_y']:
            interp_key = interp1d(data['frame'][index], data[key][index])
            data[key] = interp_key(frame_idx)

        data['frame'] = frame_idx
        data['frame'] = np.unique(data['frame']).astype(np.int)
    return data


def IF_outlier_removal(data):
    '''Remove outliers using the IsolationForest algorithm.'''
    
    points = np.array([data['pos_x'], data['pos_y']]).T
    clf = IsolationForest(random_state=0, contamination=0.1).fit(points)
    inliers = clf.predict(points)
    for key in ['pos_x', 'pos_y', 'frame', 'speed', 'direction', 'identity']:
        if np.isin(data.keys(), key, invert=True).any():
            continue
        data[key] = data[key][inliers > 0]
    return data


def dictfromh5(file, key_idx=0):
    '''Create numpy dictionary from .h5 file.
    Individual identities are dictionary keys. 
    key_idx specifies which key to read from .h5 file'''
    
    tracks = read_h5(file, key_idx)
    df = {}
    for i in np.unique(tracks[:, 3]):
        df[str(int(i))] = {
            'frame':
            np.array(tracks[:, 0]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]],
            'pos_x':
            np.array(tracks[:, 1]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]],
            'pos_y':
            np.array(tracks[:, 2]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]],
            'frame_width':
            np.array(tracks[:, 4]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]][0],
            'frame_height':
            np.array(tracks[:, 5]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]][0],
            'cylinder_x':
            np.array(tracks[:, 6]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]][0],
            'cylinder_y':
            np.array(tracks[:, 7]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]][0],
            'cylinder_r':
            np.array(tracks[:, 8]).astype(
                np.float)[np.where(np.array(tracks[:, 3]) == i)[0]][0]
        }
    return df


def read_h5(file, key_idx=None):
    '''Reading data from tracking output in .h5 format'''
    
    f = h5py.File(file, 'r')
    if key_idx != None:
        key = list(f.keys())[key_idx]
        f = f[key]
    return f


def visualize_vid_h5(video, h5_file):
    '''Visualize raw data overlaid on video.
    .h5 key and os.path.basename(video)[:-4] must match.'''
    
    f = h5py.File(h5_file, 'r')
    data = f[str(os.path.basename(video)[:-4])]
    identities = np.unique(data[:, 3])

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    count = 0
    cap = cv2.VideoCapture(file)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        this = cap.get(1)

        if ret == True:

            for i in identities:
                if data[:, 7][int(count)] < 0:
                    continue
                cv2.circle(frame, (data[:, 1][data[:, 3] == i][int(count)],
                                   data[:, 2][data[:, 3] == i][int(count)]), 2,
                           (0, 255, 0), 2)
                cv2.circle(frame, (data[:, 6][data[:, 3] == i][int(count)],
                                   data[:, 7][data[:, 3] == i][int(count)]),
                           data[:, 8][data[:, 3] == i][int(count)],
                           (0, 0, 255), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
    cv2.destroyAllWindows()


def get_incident(keys, value='2020-05-25'):
    '''finds index of date value in .h5 file keys, 
    corresponding to test dates.'''
    
    index = []
    for key in keys:
        i = np.where(key.find(value) > 0)[0]
        if len(i) > 0:
            index.append(True)
        else:
            index.append(False)
    incident = min(np.where(np.array(index) == True)[0])
    return incident


def get_all_identities(results):
    ''' get all occurring identities from get_instances() output'''
    
    identities = []
    for i in results:
        for j in np.unique(np.array(list(i.keys()))):
            identities = np.append(j, identities)
    identities = sorted(np.unique(identities).astype(np.int))
    return identities

def create_colors(n, cmap='jet'):
    '''create color array containing n colors'''
    
    cm = plt.get_cmap(cmap)
    colors = []
    for i in range(n):
        color = cm(1.*i/2)
        colors.append(color)
    colors = np.array(colors)
    return colors


def show_wait_destroy(winname, img):
    '''Open CV2 window with winname and show img. 
    Window closes after pressing any key'''
    
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(1)
    cv2.destroyWindow(winname)


def euclidean(x1, x2, y1, y2):
    '''Calculate euclidean distance between two points.'''
    
    return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


def createColDict(identities, col_dict):
    '''Create color dictionary for plotting of snippet identities.'''
    
    for e in np.unique(identities):
        if e not in col_dict:
            col_dict[e] = (random.randint(100, 255), random.randint(100, 255),
                           random.randint(100, 255))


def save_data(data):
    '''Function to store data to pandas DataFrame. 
    Data is assumed to be an NxM array.'''
    
    df = pd.DataFrame(data, columns=['pos_x', 'pos_x', 'ID', 'frame'])
    return df


def reject_outliers(data):
    '''Reject outliers within given data based on the Isolation Forest Algorithm.
    Returns array of inliers. Contamination value represents a distance metric within
    which points are expected to be clustered together.'''
    
    clf = IsolationForest(random_state=0, behaviour='new',
                          contamination=0.23).fit(data)
    inliers = clf.predict(data)
    data = np.array(data)[inliers > 0]


def get_division(file, max_frames=10000, draw=False):
    '''Divides frame according to vertical lines found across frames up until max_frames.
    Vertical lines are found using the cv2.HoughLinesP() function. '''
    
    cap = cv2.VideoCapture(file)
    count = 0
    disection_id = []
    disection_lines = []

    minLineLength = 50
    maxLineLength = 300
    try:
        while (True):
            ret, frame = cap.read()
            if ret == True:
                if count < 1:
                    img_center = np.array(
                        [int(frame.shape[1] / 2),
                         int(frame.shape[0] / 2)])
                if count > max_frames:
                    break

                if ret == True:
                    gray = cv2.cvtColor(frame[:frame.shape[0] - 20, :],
                                        cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                    lines = cv2.HoughLinesP(image=edges,
                                            rho=1,
                                            theta=np.pi,
                                            threshold=60,
                                            lines=np.array([]),
                                            minLineLength=minLineLength,
                                            maxLineGap=80)
                    if lines is None:
                        count += 1
                        continue
                    a, b, c = lines.shape

                    for i in range(a):
                        p1 = np.array([lines[i][0][0], lines[i][0][1]])
                        p2 = np.array([lines[i][0][2], lines[i][0][3]])
                        if np.sqrt(
                                pow(p2[1] - p1[0], 2) +
                                pow(p2[1] - p1[1], 2)) > maxLineLength:
                            count += 1
                            continue
                        p3 = img_center
                        d = np.linalg.norm(np.cross(
                            p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

                        if d < 35:
                            if np.isin(d, disection_id) == True:
                                continue
                            disection_id = np.append(d, disection_id)
                            disection_lines.append(lines[i])

                    if draw == True:
                        if len(disection_lines) <= 1:
                            count += 1
                            continue
                        for line in disection_lines:
                            cv2.line(frame, (line[0][0], line[0][1]),
                                     (line[0][2], line[0][3]), (0, 0, 255), 1,
                                     cv2.LINE_AA)
                        cv2.circle(frame, (int(
                            (np.mean(np.array(disection_lines)[:][:, 0, 0]) +
                             np.mean(np.array(disection_lines)[:][:, 0, 2])) /
                            2), int(frame.shape[0] / 2)), 1, (255, 0, 0),
                                   cv2.LINE_AA)
                        cv2.imshow('frame', frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            break
                    count += 1

        division_point = (int(
            (np.mean(np.array(disection_lines)[:][:, 0, 0]) +
             np.mean(np.array(disection_lines)[:][:, 0, 2])) / 2) /
                          frame.shape[1],
                          int(frame.shape[0] / 2) / frame.shape[0])
        cap.release()
        if draw == True:
            cv2.destroyAllWindows()

        return division_point

    except Exception as e:
        division_point = np.array([0, 0])
        print(file, '\nException:', e)


def find_cylinder(file, draw=False, division=None, max_frames=1000):
    '''Function to find circular obstacles in given scene using the HoughCircles functionality.'''
    
    cap = cv2.VideoCapture(file)
    count = 0
    cylinders = [[], []]
    if division == None:
        division = get_division(file, max_frames=1000)
    try:
        while (True):
            ret, frame = cap.read()
            if ret == True:
                if count < 1:
                    img_center = np.array(
                        [int(frame.shape[1] / 2),
                         int(frame.shape[0] / 2)])
                if count >= max_frames:
                    cap.release()
                    break

                if ret == True:
                    gray = cv2.cvtColor(frame[:frame.shape[0] - 20, :],
                                        cv2.COLOR_BGR2GRAY)

                    circles = cv2.HoughCircles(gray,
                                               cv2.HOUGH_GRADIENT,
                                               1,
                                               20,
                                               param1=50,
                                               param2=30,
                                               minRadius=8,
                                               maxRadius=15)
                    try:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            d = np.sqrt(
                                pow(division[0] - i[0] / gray.shape[0], 2) +
                                pow(division[1] - i[1] / gray.shape[1], 2))
                            if d < 1 and len(cylinders) <= 500:
                                # draw the outer circle
                                if draw == True:
                                    cv2.circle(frame, (i[0], i[1]), i[2],
                                               (0, 255, 0), 2)
                                    # draw the center of the circle
                                    cv2.circle(frame, (i[0], i[1]), 2,
                                               (0, 0, 255), 3)
                                    # draw the center of the division plane
                                    cv2.circle(
                                        frame,
                                        (int(division[0] * gray.shape[1]),
                                         int(division[1] * gray.shape[0])), 2,
                                        (0, 0, 255), 3)

                                if i[0] / gray.shape[1] < division[0]:
                                    cylinders[0].append(i)
                                    identity = 0
                                else:
                                    cylinders[1].append(i)
                                    identity = 1

                    except Exception as e:
                        count += 1
                        continue

                    if draw == True:
                        cv2.imshow('frame', frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            print('Cylinder 0: x',
                                  np.mean(np.array(cylinders[0])[:, 0]), 'y ',
                                  np.mean(np.array(cylinders[0])[:, 1]), 'r ',
                                  np.mean(np.array(cylinders[0])[:, 2]))
                            print('Cylinder 1: x',
                                  np.mean(np.array(cylinders[1])[:, 0]), 'y ',
                                  np.mean(np.array(cylinders[1])[:, 1]), 'r ',
                                  np.mean(np.array(cylinders[1])[:, 2]))
                            break
                    count += 1

        cap.release()
        if draw == True:
            cv2.destroyAllWindows()

    except Exception as e:
        print(file, '\nException:', e)

    if len(cylinders[0]) > 0 and len(cylinders[1]) > 0:
        cylinder0 = np.array([
            np.mean(np.array(cylinders[0])[:, i])
            for i in range(cylinders[0][0].shape[0])
        ])
        cylinder1 = np.array([
            np.mean(np.array(cylinders[1])[:, i])
            for i in range(cylinders[1][0].shape[0])
        ])
    else:
        cylinder0 = np.array([-1, -1, -1])
        cylinder1 = np.array([-1, -1, -1])
    return cylinder0, cylinder1


def get_identity(x, division_point):
    '''define identity based on whether side x is in respect to the division_point'''
    
    try:
        division_point
    except NameError:
        id_exists = False
    else:
        id_exists = True

    if id_exists:
        if x < division_point[0]:
            identity = 0
        else:
            identity = 1
    else:
        identity = -1
    return identity


def mem_track(file, draw=False, save=True, write=False):
    '''Tracking function with incoporated positional memory of previous points.
    Trajectroy snippets are computed, based on this memory and given distinct IDs.'''

    output_dir = '/home/user/'
    output_filepath = str(output_dir +
                          os.path.splitext(os.path.basename(file))[0] +
                          '_tracks.csv')

    msec_start = 30000  ## Discard first 30 seconds

    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=450)
    #     bg = cv2.bgsegm.createBackgroundSubtractorCNT()
    if draw == True:
        window_name = os.path.basename(file)
        cv2.namedWindow(str(window_name), cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(file)
    cap.set(cv2.CAP_PROP_POS_MSEC, msec_start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if write == True:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('/home/fritz/Desktop/out.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (frame_width, frame_height))

    count = int(fps * (msec_start / 1000))
    division_point = get_division(file)

    circle_list = []
    df = []
    last = 0
    kernel1 = (3, 3)
    kernel2 = (5, 5)
    kernel3 = (9, 9)

    ## Individual location(s) measured in the last and current step
    n_inds = 2
    meas_last = list(np.zeros((n_inds, 2)))
    meas_now = list(np.zeros((n_inds, 2)))

    ## initialize memory:
    mem_x = []
    mem_y = []
    mem_i = []
    mem_frame = []
    max_ind = 0

    point = []
    inds = []
    pre = []
    col_dict = {}
    df = []

    ## Print filename:
    print('----------------------------------------------------')
    print('Tracking: ', file)
    print('----------------------------------------------------')

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        this = cap.get(1)

        if ret == True:

            # memory storage:
            current = []

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame[:frame.shape[0] - 20, :],
                                cv2.COLOR_BGR2GRAY)

            background = bg.apply(gray)
            blur = background
            closing = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel3)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
            erosion = cv2.erode(opening, kernel1, iterations=1)
            dilation = cv2.dilate(erosion, kernel1, iterations=1)
            blur = cv2.GaussianBlur(dilation, kernel2, 0)
            ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            filtered_countours = np.array([
                i for i in contours
                if cv2.contourArea(i) > 0 and cv2.contourArea(i) < 50
            ])

            for cnt in filtered_countours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                fish_x = float(x + w / 2) / float(gray.shape[1])
                fish_y = float(y + h / 2) / float(gray.shape[0])
                current.append([fish_x, fish_y, np.nan])

            if np.array(mem_x).size != 0:  ### Limit memory to last 10 frames:
                flist = [10 > (count - d) for d in mem_frame]
                mem_x = mem_x[flist]
                mem_y = mem_y[flist]
                mem_i = mem_i[flist]
                mem_frame = mem_frame[flist]

            for element in current:
                if len(pre) > 0 and count != 0:

                    for previous_element in pre:
                        dist_euclidean = euclidean(
                            int(previous_element[0] * gray.shape[1]),
                            int(element[0] * gray.shape[1]),
                            int(previous_element[1] * gray.shape[0]),
                            int(element[1] * gray.shape[0])
                        )  ### Calculating euclidean distance between points:
                        if dist_euclidean < 10:  ### Check distance to objects in previous frame
                            element[2] = previous_element[2]
                            point.append(
                                [element[0], element[1], element[2], count])
                            df.append([
                                count, element[0], element[1],
                                get_identity(element[0], division_point),
                                frame.shape[0], frame.shape[1]
                            ])

                        else:
                            reassigned = 0
                            for i in range(len(mem_i)):
                                dist_euclidean = euclidean(
                                    int(mem_x[i] * gray.shape[1]),
                                    int(element[0] * gray.shape[1]),
                                    int(mem_y[i] * gray.shape[0]),
                                    int(element[1] * gray.shape[0]))
                                if dist_euclidean < 20 and reassigned == 0:  ### Check distance to object in memory
                                    element[2] = int(mem_i[i])
                                    reassigned = 1
                                    break

                            if reassigned == 0:
                                element[2] = max(np.array(point)[:, 2]) + 1

                elif len(point) == 0:
                    element[2] = 0
                    point.append([element[0], element[1], element[2], count])
                    df.append([
                        count, element[0], element[1],
                        get_identity(element[0], division_point),
                        frame.shape[0], frame.shape[1]
                    ])

                else:
                    element[2] = max(np.array(point)[:, 2]) + 1

                list_id = [x[2] for x in current]
                for previous_element in pre:
                    if previous_element[2] not in list_id:
                        mem_x = np.append(mem_x, previous_element[0])
                        mem_y = np.append(mem_y, previous_element[1])
                        mem_i = np.append(mem_i, previous_element[2])
                        mem_frame = np.append(mem_frame, count)

            if draw == True:
                if np.array(point).size != 0:
                    createColDict([x[2] for x in point], col_dict)

                for i in point:
                    # if np.isfinite(i[2]) and (i[3] > self.frame_count - 40): ### trailing track for last 40 frames:
                    cv2.circle(
                        frame,
                        (int(i[0] * gray.shape[1]), int(i[1] * gray.shape[0])),
                        1,
                        (0, 255, 0),
                        #                         col_dict[i[2]],
                        1,
                        lineType=cv2.LINE_AA)
            pre = current

        if last >= this:
            break

        last = this

        count += 1
        if write == True:
            # Write the frame into the file
            out.write(frame)

        if draw == True:
            #       # Display the resulting frame
            cv2.imshow(str(window_name), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    df = pd.DataFrame(np.matrix(df),
                      columns=[
                          'frame', 'pos_x', 'pos_y', 'identity',
                          'frame_height', 'frame_width'
                      ])
    if save == True:
        df.to_csv(output_filepath, sep=',')

    # When everything done, release the capture
    cap.release()
    out.release()
    if draw == True:
        cv2.destroyAllWindows()
        
        
def plt_h5(input_file, cmap='plasma'):
    '''Function for plotting trajectories accumulated in a .h5 file.
    Individual tests should be defined as keys within this file.'''
    
    f = h5py.File(input_file, 'r')
    cmap = matplotlib.cm.get_cmap(cmap)
    for key in f.keys():
        fig, ax = plt.subplots()
        for j, i in enumerate(np.unique(np.array(f[key][:, 3]))):
            x_track = np.array(f[key][:, 1]).astype(
                np.float)[np.where(np.array(f[key][:, 3]) == i)[0]]
            y_track = np.array(f[key][:, 2]).astype(
                np.float)[np.where(np.array(f[key][:, 3]) == i)[0]]
            x = np.array(f[key][:,
                                6])[np.where(np.array(f[key][:,
                                                             3]) == i)[0][0]]
            y = np.array(f[key][:,
                                7])[np.where(np.array(f[key][:,
                                                             3]) == i)[0][0]]
            r = np.array(f[key][:,
                                8])[np.where(np.array(f[key][:,
                                                             3]) == i)[0][0]]
            cylinder = plt.Circle(
                (x, y + 20), r, color='r'
            )  ## added 20 since image was cropped by 20 px along frame.shape[0]
            print(i)
            ax.scatter(x_track, y_track, s=0.7, c=np.array([cmap(0.7 * j)]))
            ax.add_artist(cylinder)
            ax.set_aspect('equal')
        plt.show()


def track2h5(
    input_file,
    save=False,
    plot=False,
    output_dir='/home/user/',
    trial = 0):
    '''Tracking function with incoporated positional memory of previous points.
    Trajectroy snippets are computed, based on this memory and given distinct IDs.
    Parameters
    ----------
    input_file: obj
        video file input
    Returns
    -------
    object
        trajectories in .h5 format. 
        Column headers: FRAME,X,Y,ID,
        FRAME_WIDTH,FRAME_HEIGHT,CYLINDER_X,
        CYLINDER_Y,CYLINDER_R
    '''

    output_filepath = str(
        output_dir + 'trial_' + str(trial) + '_' +
        os.path.splitext(os.path.basename(input_file))[0][-5:] + '_tracks.h5')

    filename = os.path.splitext(os.path.basename(input_file))[0]

    if save == True:
        output_file = h5py.File(output_filepath, 'a')

    msec_start = 30000  ## Discard first 30 seconds
    max_frame = 13000  ## crop analysis to 13000 frames for all videos

    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=450)

    cap = cv2.VideoCapture(input_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, msec_start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = int(fps * (msec_start / 1000))

    division_point = get_division(input_file, max_frames=800)
    cylinders = np.array(
        find_cylinder(input_file, division_point, max_frames=1000))

    last = 0
    kernel1 = (3, 3)
    kernel2 = (5, 5)
    kernel3 = (9, 9)

    ## initialize memory:
    mem_x = []
    mem_y = []
    mem_i = []
    mem_frame = []
    max_ind = 0

    point = []
    inds = []
    pre = []
    col_dict = {}

    ## Print filename:
    print(
        '--------------------------------------------------------------------------------------------------------'
    )
    print('Tracking: ', input_file)

    try:

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            this = cap.get(1)

            if ret == True:

                if count % 500 == 0:
                    print(np.round(np.round(count / length, 2) * 100, 1),
                          '%',
                          flush=True)

                # memory storage:
                current = []

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame[:frame.shape[0] - 20, :],
                                    cv2.COLOR_BGR2GRAY)

                background = bg.apply(gray)
                blur = background
                closing = cv2.morphologyEx(background, cv2.MORPH_CLOSE,
                                           kernel3)
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
                erosion = cv2.erode(opening, kernel1, iterations=1)
                dilation = cv2.dilate(erosion, kernel1, iterations=1)
                blur = cv2.GaussianBlur(dilation, kernel2, 0)
                ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                filtered_countours = np.array([
                    i for i in contours
                    if cv2.contourArea(i) > 0 and cv2.contourArea(i) < 50
                ])

                for cnt in filtered_countours:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    fish_x = float(x + w / 2) / float(gray.shape[1])
                    fish_y = float(y + h / 2) / float(gray.shape[0])                    
                    current.append([fish_x, fish_y, np.nan])

                if np.array(
                        mem_x).size != 0:  ### Limit memory to last 10 frames:
                    flist = [10 > (count - d) for d in mem_frame]
                    mem_x = mem_x[flist]
                    mem_y = mem_y[flist]
                    mem_i = mem_i[flist]
                    mem_frame = mem_frame[flist]

                for element in current:
                    if len(pre) > 0 and count != 0:

                        for previous_element in pre:
                            dist_euclidean = euclidean(
                                int(previous_element[0] * gray.shape[1]),
                                int(element[0] * gray.shape[1]),
                                int(previous_element[1] * gray.shape[0]),
                                int(element[1] * gray.shape[0])
                            )  ### Calculating euclidean distance between points:
                            if dist_euclidean < 10:  ### Check distance to objects in previous frame
                                element[2] = previous_element[2]
                                point.append([
                                    element[0], element[1], element[2], count
                                ])
                                ## hard coded to specific file ending!
                                idx = get_identity(element[0], division_point)
                                identity = input_file[int(-9 + (
                                    3 * idx)):int(-7 + (3 * idx))]
                                data = np.array([
                                    float(count),
                                    float(element[0] * frame.shape[1]),
                                    float(element[1] * frame.shape[0]),
                                    float(int(identity)),
                                    float(frame.shape[0]),
                                    float(frame.shape[1]),
                                    float(cylinders[idx][0]),
                                    float(cylinders[idx][1]),
                                    float(cylinders[idx][2])
                                ])
                                                                
                                if save == True:
                                    if np.isin(str(filename),
                                               list(output_file.keys())).all(
                                               ) == False:
                                        print(' \ncreated dataset', str(filename),
                                              list(output_file.keys()))
                                        print('\ncylinder:', cylinders[idx])
                                        output_file.require_dataset(
                                            str(filename),
                                            data=data,
                                            shape=(1, 9),
                                            dtype='f',
                                            maxshape=(
                                                None,
                                                None,
                                            ),
                                            chunks=(5000, 9),
                                            compression="gzip",
                                            compression_opts=9)
                                    else:
                                        output_file[str(filename)].resize(
                                            (output_file[str(filename)].shape[0] +
                                             data.shape[0]),
                                            axis=0)
                                        output_file[str(
                                            filename)][-data.shape[0]:] = data

                            else:
                                reassigned = 0
                                for i in range(len(mem_i)):
                                    dist_euclidean = euclidean(
                                        int(mem_x[i] * gray.shape[1]),
                                        int(element[0] * gray.shape[1]),
                                        int(mem_y[i] * gray.shape[0]),
                                        int(element[1] * gray.shape[0]))
                                    if dist_euclidean < 20 and reassigned == 0:  ### Check distance to object in memory
                                        element[2] = int(mem_i[i])
                                        reassigned = 1
                                        continue

                                if reassigned == 0:
                                    element[2] = max(np.array(point)[:, 2]) + 1

                    elif len(point) == 0:
                        element[2] = 0
                        point.append(
                            [element[0], element[1], element[2], count])
                        ## hard coded to specific file ending!
                        idx = get_identity(element[0], division_point)
                        identity = input_file[int(-9 +
                                                  (3 * idx)):int(-7 +
                                                                 (3 * idx))]
                        
                        data = np.array([
                            float(count),
                            float(element[0] * frame.shape[1]),
                            float(element[1] * frame.shape[0]),
                            float(int(identity)),
                            float(frame.shape[0]),
                            float(frame.shape[1]),
                            float(cylinders[idx][0]),
                            float(cylinders[idx][1]),
                            float(cylinders[idx][2])
                        ])
                        if save == True:
                            if np.isin(str(filename),
                                       list(output_file.keys())).all() == False and save == True:
                                print(' \ncreated dataset', str(filename),
                                      list(output_file.keys()))
                                print('\ncylinder:', cylinders[idx])
                                output_file.require_dataset(str(filename),
                                                            data=data,
                                                            shape=(1, 9),
                                                            dtype='f',
                                                            maxshape=(
                                                                None,
                                                                None,
                                                            ),
                                                            chunks=(5000, 9),
                                                            compression="gzip",
                                                            compression_opts=9)
                            else:
                                output_file[str(filename)].resize(
                                    (output_file[str(filename)].shape[0] +
                                     data.shape[0]),
                                    axis=0)
                                output_file[str(filename)][-data.shape[0]:] = data
                        if plot == True:
                            gray = cv2.circle(
                                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                (int(data[1]), int(data[2])), 1, (255, 0, 0))

                    else:
                        element[2] = max(np.array(point)[:, 2]) + 1

                    list_id = [x[2] for x in current]
                    for previous_element in pre:
                        if previous_element[2] not in list_id:
                            mem_x = np.append(mem_x, previous_element[0])
                            mem_y = np.append(mem_y, previous_element[1])
                            mem_i = np.append(mem_i, previous_element[2])
                            mem_frame = np.append(mem_frame, count)
                            
                pre = current

                if plot == True:
                    cv2.imshow('frame', gray)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        cap.release()
                        break

            if (last >= this) or (count >= length) or (count >= max_frame):
                print(np.round(np.round(count / length, 2) * 100, 1),
                      '%',
                      flush=True)
                if save == True:
                    output_file.close()
                cap.release()
                print('Reached end of video file.')
                break

            last = this
            point = point[-11:]
            count += 1
            
        if save == True:
            output_file.close()
        cap.release()
        if plot == True:
            cv2.destroyAllWindows()
        print('Exited normally.')

    except Exception as e:
        if save == True:
            output_file.close()
        print(e)
    if save == True:
        output_file.close()
    

def tracks2h5(file_chunk):
    '''track videos in file_chunk using track2h5()'''
    
    for video in np.array(file_chunk):
        try:
            track2h5(video)
        except Exception as e:
            print(e)
    return True


def chunk_vid_list(input_arr):
    '''create input chuncks for multiprocessing from input array.'''
    
    ids = np.array([i[-9:-4] for i in input_arr])
    file_chunks = []
    for i in np.unique(ids):
        file_chunks.append(
            np.array(input_arr)[np.where(ids == i)[0]].astype('str'))
    file_chunks = np.array(file_chunks)
    return file_chunks

def calculate_detection_probability(h5_files, max_frame=13000, plot=True, ax=None):
    '''Calculate detection probability as fraction of total recording time (max_frame)'''
    
    detection_probability = []
    for file in h5_files:
        f = h5py.File(file,'r')
        keys = sorted(np.array(list(f.keys())))
        for key in keys:
            time_range = np.arange(max_frame)
            for identity in np.unique(np.array(f[key])[:,3]):
                id_time_range = np.unique(np.array(f[key])[:,0][np.array(f[key])[:,3] == identity])
                detection_probability = np.append(detection_probability,np.round(len(id_time_range)/len(time_range),2))
        f.close()
    
    if plot == True:
        if ax != None:
            sns.distplot(detection_probability,ax=ax)
            return ax
        else:
            fig,ax = plt.subplots()
            ax = sns.distplot(detection_probability, ax=ax)
            return fig, ax
    else:
        return detection_probability
    
def calculate_hd(id_tracks1,id_tracks2):
    ''' Calculate Hausdorff distance between two, size-matched trajectories '''
    
    min_idx = int(max(min(id_tracks1['frame']),min(id_tracks2['frame'])))
    max_idx = int(min(max(id_tracks1['frame']),max(id_tracks2['frame'])))
    
    index1 = [(id_tracks1['frame'] > min_idx) & (id_tracks1['frame'] < max_idx)][0]
    index2 = [(id_tracks2['frame'] > min_idx) & (id_tracks2['frame'] < max_idx)][0]

    arr1 = np.array((id_tracks1['pos_x'][index1],id_tracks1['pos_y'][index1])).T
    arr2 = np.array((id_tracks2['pos_x'][index2],id_tracks2['pos_y'][index2])).T

    dhd = max(directed_hausdorff(arr1, arr2)[0], directed_hausdorff(arr2, arr1)[0])   
    
    return dhd

def get_hdistances(file,save=True,plot=False):
    '''return Hausdorff distances from .h5 file as returned by track2h5().'''
    
    output_dir = '/home/user/'
    f = h5py.File(file,'r')
    keys = np.array(list(f.keys()))
    f.close()
    name = file.replace('.h5','')
    outfile = os.path.basename(name).replace('tracks','hdistances')
    outfile = str(output_dir+outfile)

    hdistances = []
    for j,k in enumerate(keys):
        print(np.round((j/len(keys))*100,1),'%')
        tracks = dictfromh5(file,j)
        identities = np.array([int(file[-15+(3*idx):-13+(3*idx)]) for idx in [0,1]])
        ## make sure trajectories for both IDs exists:
        if (len(tracks.keys()) != 2) or (np.array([tracks[str(i)]['cylinder_x'] for i in tracks]).any() == -1):
            continue
        for i in tracks:
            id_tracks = tracks[str(i)]
            id_tracks = simple_filter(id_tracks,threshold=4)
            id_tracks = rmv_out_pts(id_tracks)
            tracks[str(i)] = id_tracks
        dhd = calculate_hd(tracks[str(identities[0])],tracks[str(identities[1])])
        hdistances.append(np.round(dhd))
    hdistances = np.array(hdistances)

    if save==True:
        np.save(outfile, hdistances)

    if plot==True:
        fig,ax = plt.subplots()
        ax.plot(hdistances)
        ax.axvline(get_incident(keys), c='r')
        plt.show()
        
    return hdistances

def get_incident(keys, value='2020-05-25'):
    '''finds index of date value in .h5 file keys, 
    corresponding to test dates.'''
    
    index = []
    for key in keys:
        i = np.where(key.find(value) > 0)[0]
        if len(i) > 0:
            index.append(True)
        else:
            index.append(False)
    if np.where(np.array(index) == True)[0].size == 0:
        incident = np.nan
    else:
        incident = min(np.where(np.array(index) == True)[0])
    return incident

def get_instances(file,
                  save=False,
                  plot=False,
                  output_dir='/home/fritz/Desktop/data_20200715/'):
    '''Function to retrieve instances where xy coordinates are within a specified range.
    In this case it is specifically designed for .h5 input retrieved through track2h5()
    which contains cylinder coordinates and radii. 
    These were collected using the find_cylinder() function.'''
    
    instances = {}
    f = h5py.File(file, 'r')
    keys = np.array(list(f.keys()))
    name = file.replace('.h5', '')

    for key in keys:
        for i in np.unique(f[key][:, 3]):
            instances[str(int(i))] = {}
    f.close()
    identities = np.unique(np.array(list(instances.keys())))
    
    for j, key in enumerate(keys):
        print(os.path.basename(name), np.round((j / len(keys)) * 100, 1), '%')
        tracks = dictfromh5(file, j)
        
         ## make sure trajectories for both IDs exists:
        if (len(tracks.keys()) != 2) or (np.array([tracks[str(i)]['cylinder_x'] for i in tracks]).any() == -1):
            continue
            
        for i in identities:

            id_tracks = tracks[str(int(i))]
            id_tracks = simple_filter(id_tracks, threshold=4)
            id_tracks = rmv_out_pts(id_tracks)

            x = id_tracks['pos_x']
            y = id_tracks['pos_y']
            cx = id_tracks['cylinder_x']
            cy = id_tracks['cylinder_y']
            cr = id_tracks['cylinder_r']

            if cr.any() < 0:
                continue
            else:
                distances = np.sqrt((x - cx)**2 + (y - (cy+20))**2) ## 20 added due to cropping along frame.shape[0]
                boolean = np.where(distances <= cr)[0]
                instances[str(int(i))][key] = {
                    'distances': distances,
                    'boolean': boolean
                }
            if plot == True:
                fig, ax = plt.subplots()
                circle = plt.Circle((int(cx), int(cy)+20), ## 20 added due to cropping along frame.shape[0]
                                    int(cr),
                                    color='r',
                                    alpha=0.8)
                ax.add_artist(circle)
                ax.scatter(x, y, s=0.5)
                ax.set_aspect('equal')
                ax.set_axis_off()
                plt.show()

    if plot == True:
        for i in instances.keys():
            instance = []
            for key in instances[str(i)].keys():
                instance.append(len(instances[str(i)][key]['boolean']))
            plt.plot(instance, label=str(i))
        plt.legend()
        plt.show()

    if save == True:
        outfile = os.path.basename(name).replace('tracks', 'instances')
        outfile = str(output_dir + outfile)
        np.save(outfile, instances)
    return instances

def collect_trex(directory,identities=[0,1,2,3]):
    '''combine trex.run output for a given directory and number of individuals'''
    
    area_inner_tank = 423112 ## px measured at bottom of tank
    r = np.sqrt(area_inner_tank/np.pi)
    px2cm = r/30
    trex2px = 2046/30
    threshold = 13.7
    center = np.array([15,15])
    data_stacks = []
    canvas = np.zeros((int(2040*(2040/2046)),2046,3), np.uint8)

    for identity in np.array(identities):
        data_stack = []
        file_list = sorted(glob.glob(str(directory + '/data/*fish'+str(identity)+'.npz')))
        for j,file in enumerate(file_list):
            data = np.load(file,allow_pickle=True)
            x = data['X']
            y = data['Y']
            distance = np.sqrt(np.power(x - center[0],2)+np.power(y-center[1],2))

            index = np.where((y!=np.inf) & (x!=np.inf) & (distance <= threshold))[0]

            x = (x[index] - center[0])
            y = (y[index] - center[1])
            
            if j == 0:
                time = data['frame']
            else:
                time = data['frame']+max(frame_idx)
                
            frame_idx = time.astype(np.int32)        
            time = time[index]/50  

            out = np.array([time,x,y]).T
            data_stack = np.append(data_stack, out)
        data_stack = data_stack.reshape(-1, out.shape[1])
        
        data_stacks.append(data_stack)
        del data_stack, out
        
    out = {}
    for i in np.array(identities):
        out['TIME%s'%i] = data_stacks[i][:,0]
        out['X%s'%i] = data_stacks[i][:,1]
        out['Y%s'%i] = data_stacks[i][:,2]

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in out.items() ]))  
    del out
    
    cv2.circle(canvas,(int(canvas.shape[0]/2),int(canvas.shape[1]/2)),int(13.7*trex2px),(255,255,255),-1,cv2.LINE_AA)
    
    for i in np.array(identities):
        for j, p in enumerate(df[str('X%s'%i)]):
            if np.isnan(p):
                continue
            else:
                canvas = cv2.circle(canvas, (int((df[str('X%s'%i)][j]+center[0])*trex2px),int((df[str('Y%s'%i)][j]+center[1])*trex2px)),int(5), colors[i]*255, -1, cv2.LINE_AA)
    return df, canvas


def full_sig_xlags(tracks, asarray=True, exclude_ids=[]):
    '''calculate mean correlation based leader-follower value from the entire signal'''
    
    pairs = itertools.permutations(tracks['IDENTITIES'], 2)
    pairs = np.array(list(pairs))
    fps = 20
    lags = {}
    for i in np.delete(tracks['IDENTITIES'],np.array(exclude_ids, dtype=np.int)):
        lags[str(i)] = 0

    for pair in pairs:
        if str(pair[0]) not in lags or str(pair[1]) not in lags:
            continue
        index = cooccurrence_index(tracks[str(pair[0])],tracks[str(pair[1])])
        xsig = calc_xcorr(tracks[str(pair[0])]['SPEED'][index[0]],tracks[str(pair[1])]['SPEED'][index[1]])
        lags[str(pair[0])] += xsig[0][np.argmax(xsig[1])]/fps

    for i in lags:
        lags[str(i)] = np.round(lags[str(i)]/int(len(lags.keys())-1),2)
        
    if asarray == True:
        lags = np.array([lags[str(i)] for i in lags.keys()])
    return lags

def calc_distance_stack(tracks):
    '''calculate distance stack, as stack of matrices.
    The resulting stacks shape[0] is length of FRAME_IDX 
    and shape[1] and shape[2] are N individuals '''
    
    frame_idx = tracks['FRAME_IDX']
    pts = np.array([np.array([tracks[str(i)]['X'],tracks[str(i)]['Y']]) for i in tracks['IDENTITIES']])
    pts = np.rot90(pts).T

    # create stack of len(time) with distance matrix for all four individuals:
    dist_stack = np.zeros((len(frame_idx),4,4))

    for t, frame in enumerate(frame_idx):
        dist_stack[int(t)] = cdist(pts[t],pts[t])
    return dist_stack

def calc_pairwise_distances(tracks,asarray=True):
    '''calculate pairwise distances'''
    
    dist_stack = calc_distance_stack(tracks)
    pairwise_distances = {}
    for i in tracks['IDENTITIES']:
        pairwise_distances[str(i)] = {}
        maximum = np.concatenate(
            [dist_stack[:, int(i), j] for j in np.arange(dist_stack.shape[2])])
        maximum[np.isinf(maximum)] = 0
        maximum[np.isnan(maximum)] = 0
        maximum = np.max(maximum)

        for n, j in enumerate(
                np.array(tracks['IDENTITIES'])[np.array(
                    np.array(tracks['IDENTITIES']) != i)]):
            sig = dist_stack[:, int(i), int(j)]
            pairwise_distances[str(i)][str(j)] = np.mean(sig[np.isfinite(sig)] /
                                                         maximum)
    if asarray == True:
        out = []
        for i in list(pairwise_distances.keys()):
            for n, j in enumerate(np.array(list(pairwise_distances.keys()))[np.array(np.array(list(pairwise_distances.keys())) != i)]):
                out = np.append(out,pairwise_distances[str(i)][str(j)])
        pairwise_distances = out.reshape(len(pairwise_distances.keys()),len(pairwise_distances.keys())-1)
    return pairwise_distances

def trex2tracks(files, identities = np.arange(4), interpolate=True, start_idx = 1200, end_idx = 72000, threshold = 14):
    '''create tracks{} dictionary from trex.run output.
    Function also filter outliers by thresholding distance to center of arena and interpolate x,y'''

    tracks = {'IDENTITIES': identities}
    
    for i in identities:
        tracks[str(i)] = {}
        data = np.load(files[i])
        index = np.where((data['frame'] >= start_idx) & (data['frame'] < end_idx))[0]
        distance = np.sqrt(
            np.power(data['X'][index] - 15, 2) + np.power(data['Y'][index] - 15, 2))
        if interpolate==True:
            x_itpd, _ = interpolate_signal(data['X'][index][distance < threshold],
                                           data['frame'][index][distance < threshold])
            y_itpd, frame_idx = interpolate_signal(data['Y'][index][distance < threshold],
                                                   data['frame'][index][distance < threshold])

            tracks[str(i)]['X'] = np.array(x_itpd).astype(float)
            tracks[str(i)]['Y'] = np.array(y_itpd).astype(float)
        else:
            tracks[str(i)]['X'] = np.array(data['X'][index][distance < threshold]).astype(float)
            tracks[str(i)]['Y'] = np.array(data['Y'][index][distance < threshold]).astype(float)
            
        tracks[str(i)]['SPEED'] = get_speed(tracks[str(i)])
        tracks[str(i)]['FRAME_IDX'] = np.array(frame_idx).astype(float)
    tracks = get_direction(tracks)
    del data
    
    frame_idx = np.unique(np.concatenate([np.load(files[i])['frame'] for i in identities]))
    index = np.where((frame_idx >= start_idx) & (frame_idx < end_idx))[0]
    frame_idx = np.arange(np.min(frame_idx[index]),np.max(frame_idx[index]))
    tracks['FRAME_IDX'] = frame_idx.astype(np.int32)
    ret = np.array([[tracks[str(i)]['X'],tracks[str(i)]['Y']] for i in tracks['IDENTITIES']])
    try:
        assert len(ret.shape) == 3
        ret = True
        return ret, tracks

    except AssertionError as e:
#         print("No tracking data retrieved: \n", os.path.dirname(files[0]))
        ret = False
        return ret, tracks
    
def check_corruption(files):
    '''check trex.run output files (.npz) for corruptions'''
    
    corrupted = []
    for file in files:
        try:
            data = np.load(file, allow_pickle=True)
            corrupted = np.append(corrupted, False)
        except:
            corrupted = np.append(corrupted, True)
            continue
    del data
    
    index = np.array(corrupted==False)
    uncorrupted_files = np.array(files)[np.array(index)]
    del corrupted
    return uncorrupted_files
    
def group_polarization(tracks, smoothing_window = 501):
    '''calculate polarization of group as sum of unit direction vectors for each individual'''
    
    d = []
    for i in tracks['IDENTITIES']:
        d.append(tracks[str(i)]['ANGLE'])
    d = np.array(d)
    x = np.sum(np.cos(d),axis=0)/len(tracks['IDENTITIES'])
    y = np.sum(np.sin(d),axis=0)/len(tracks['IDENTITIES'])
    polarization = np.sqrt(np.power(x,2) + np.power(y,2))
    if smoothing_window != 0:
        polarization = savgol_filter(polarization, int(smoothing_window), 3)
    return polarization

def group_speed(tracks, smoothing_window = 501, fps = 20, px2m = 0.055751029873265766):
    '''calculate group speed as speed of group centroid.
    px2m depends highly on tracking output.'''
    
    x = []
    y = []

    for i in tracks['IDENTITIES']:
        x.append(tracks[str(i)]['X'])
        y.append(tracks[str(i)]['Y'])
    x = np.mean(np.array(x).T,axis=1)
    y = np.mean(np.array(y).T,axis=1)
    speed = np.sqrt(np.power(np.diff(x),2)+np.power(np.diff(y),2))*px2m*int(fps)
    speed = np.append(speed,speed[-1])
    speed = savgol_filter(speed,smoothing_window,3)
    return speed 

def get_activity(file,
                  save=False,
                  plot=False,
                  output_dir='/home/ffrancisco/Desktop/data_20200715/'):
    '''Function to retrieve instances where xy coordinates are within a specified range.
    In this case it is specifically designed for .h5 input retrieved through track2h5()
    which contains cylinder coordinates and radii. 
    These were collected using the find_cylinder() function.'''
    activity = {}
    f = h5py.File(file, 'r')
    keys = np.array(list(f.keys()))
    name = file.replace('.h5', '')

    for key in keys:
        for i in np.unique(f[key][:, 3]):
            activity[str(int(i))] = {}
    f.close()
    identities = np.unique(np.array(list(activity.keys())))
    
    for j, key in enumerate(keys):
        print(os.path.basename(name), np.round((j / len(keys)) * 100, 1), '%')
        tracks = dictfromh5(file, j)
        
         ## make sure trajectories for both IDs exists:
        if (len(tracks.keys()) != 2) or (np.array([tracks[str(i)]['cylinder_x'] for i in tracks]).any() == -1):
            continue
            
        for i in identities:

            id_tracks = tracks[str(int(i))]
            id_tracks = simple_filter(id_tracks, threshold=4)
            id_tracks = rmv_out_pts(id_tracks)

            x = id_tracks['pos_x']
            y = id_tracks['pos_y']
            cr = id_tracks['cylinder_r']
            s = np.sqrt(np.power(np.diff(x),2) + np.power(np.diff(y),2))

            if cr.any() < 0:
                continue
            else:
                activity[str(int(i))][key] = {
                    'speeds': s,
                }


    if save == True:
        outfile = os.path.basename(name).replace('tracks', 'instances')
        outfile = str(output_dir + outfile)
        np.save(outfile, activity)
    return activity

def create_uuids(arr):
    '''create universally unique identifiers for each unique value of arr.
    Function returns masked array or len(arr) containing all UUIDs'''

    x = np.array([p for p in arr])
    uniques, idx = np.unique(x, axis=0, return_index=True)
    uuids = []

    while len(np.unique(uuids)) != len(uniques):
        uuids = np.array([str(uuid.uuid4().hex) for i in uniques])
    
    if len(np.array(x).shape)>1: 
        uuids = np.concatenate([uuids[np.where(np.mean(np.array([uniques==u])[0],axis=1)==1)[0]] for u in x])    
    else:
        uuids = np.concatenate([uuids[np.where(uniques==u)[0]] for u in x])    
        
    return uuids

def interpolate_signal(arr1, arr2):
    '''linear interpolation of arr1 based on arr2'''

    frame_idx = np.arange(arr2[0], arr2[-1] + 1)
    interp_key = interp1d(arr2, arr1)
    new_arr1 = interp_key(frame_idx)
    new_arr2 = np.unique(frame_idx).astype(np.int)
    return new_arr1, new_arr2

def generate_colors(n): 
    '''generate color array containing n colors in rgb and hex'''
    
    rgb_values = [] 
    hex_values = [] 
    r = int(random.random() * 256) 
    g = int(random.random() * 256) 
    b = int(random.random() * 256) 
    step = 256 / n 
    for _ in range(n): 
        r += step 
        g += step 
        b += step 
        r = int(r) % 256 
        g = int(g) % 256 
        b = int(b) % 256 
        r_hex = hex(r)[2:] 
        g_hex = hex(g)[2:] 
        b_hex = hex(b)[2:] 
        hex_values.append('#' + r_hex + g_hex + b_hex) 
        rgb_values.append((r/255,g/255,b/255,1)) 
    rgb_values = np.array(rgb_values)
    return rgb_values, hex_values 

    
def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized.
    plot using: extrapolatedtrendline=[a*index + b for index in range(len(ratios))]
    """
    
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
