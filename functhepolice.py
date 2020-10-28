from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.interpolate import SmoothSphereBivariateSpline, griddata
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import binned_statistic_2d, binned_statistic, gaussian_kde

import matplotlib as mpl
import matplotlib.animation as animation

import itertools

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
