"""This module contains analytical function(s) to assess the characteristics of NumpyPointClouds."""

# local modules
from modules.normals import normalize_vector_array, einsum_angle_between

# basic imports
import numpy as np

# advanced functionality
import scipy.spatial

# plot imports
import matplotlib.pyplot as plt


def plot_histogram (data, numer_of_bins ):
    # the histogram of the data
    n, bins, patches = plt.hist(data, numer_of_bins, density=False, range=(0, 180), facecolor='g', alpha=0.75 )

    plt.xlabel ('angle' )
    plt.ylabel ('count' )
    plt.title ('Histogram of Angle Differences from correspondence to reference' )
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis ([0, numer_of_bins, 0, max (n) + 1000] )
    plt.grid (True )
    plt.show ()


def show_normal_angle_differences (reference_pointcloud, corresponding_pointcloud ):
    """
    Takes two clouds and displays the distribution of angle differences by creating a histogram. Each point of the
    corresponding_pointcloud is checked for it's angle difference to it's nearest neighbor in the reference_pointcloud.
    """

    # extract normals
    normals_numpy_cloud = reference_pointcloud.get_normals ()
    normals_corresponding_cloud = corresponding_pointcloud.get_normals ()

    # normalize
    normals_numpy_cloud = normalize_vector_array (normals_numpy_cloud )
    normals_corresponding_cloud = normalize_vector_array (normals_corresponding_cloud )

    # build a kdtree and query it
    kdtree = scipy.spatial.cKDTree (reference_pointcloud.points[:, 0:3] )
    distances, correspondences = kdtree.query (corresponding_pointcloud.points[:, 0:3], k=1 )

    # get the angle differences between the normal vectors
    angle_differences = einsum_angle_between (normals_numpy_cloud[correspondences, :],
                                              normals_corresponding_cloud ) * (180/np.pi)

    # plot
    plot_histogram (angle_differences, 180 )
