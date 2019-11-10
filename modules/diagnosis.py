"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

This module contains analytical function(s) to assess the characteristics of NumpyPointClouds.
"""

# local modules
from modules.normals import normalize_vector_array, einsum_angle_between

# basic imports
import numpy as np

# advanced functionality
import scipy.spatial

# plot imports
import matplotlib.pyplot as plt


def cloud2cloud_distance_sum (reference_pointcloud, aligned_pointcloud, translation=(0, 0, 0) ):
    """Returns the sum of all nearest neighbor distances from aligned_pointcloud to reference_pointcloud"""

    # make a tree an get a list of distances to the nearest neigbor and his index (which is not needed)
    # but only take the x,y,z fields into consideration (reference_cloud.get_xyz_coordinates ())
    scipy_kdtree = scipy.spatial.cKDTree (reference_pointcloud.get_xyz_coordinates () )

    # translate aligned_pointcloud
    aligned_pointcloud.points[:, 0:3] += translation

    # query the three, but only take the x,y,z fields into consideration
    c2c_distances, indices = scipy_kdtree.query (aligned_pointcloud.get_xyz_coordinates (), k=1 )

    return np.sum (c2c_distances)


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
