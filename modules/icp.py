"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

Import this for a simple iterative closest point algorithm
"""

# basic imports
import numpy as np
import time

# advanced functionality
import scipy.spatial


def icp (numpy_reference_cloud, numpy_aligned_cloud, accuracy=0.000000001, verbose=False ):
    """
    Iterative closest point algorithm that computes the translation from numpy_aligned_cloud to numpy_reference_cloud.

    Input:
        input_reference_cloud: (np.ndarray)       Input Cloud. This Cloud will stay fixed in place.
        input_aligned_cloud: (np.ndarray)         Input Cloud. This Cloud will be moved to match the reference cloud.
        accuracy: (float)                         The desired accuracy of alignment in meters

    Output:
        translation: ((x, y, z) tuple)          The estimated translation between aligned_cloud and reference_cloud.
        mean_squared_error: ((x, y, z) tuple)   The remaining MSE in x, y and z direction
    """

    start_time = time.time()    # measure time

    # clouds
    reference_cloud = numpy_reference_cloud[:, 0:3]           # no copy
    aligned_cloud = numpy_aligned_cloud[:, 0:3].copy ()     # this cloud will be translated

    # preparing variables
    cloud_size = aligned_cloud.shape[0]
    translation = [0, 0, 0]
    iterations = 1

    # build a kdtree
    #tree = sklearn.neighbors.kd_tree.KDTree (reference_cloud, leaf_size=40, metric='euclidean')
    scipy_kdtree = scipy.spatial.cKDTree (reference_cloud )

    # mean difference between the clouds
    # (set to inf to avoid exit criterium in the first iteration)
    clouds_delta = np.array([np.inf, 0, 0])
    clouds_delta_previous_iteration = np.array([0, 0, 0])

    # iterate while there is a change in the position of the shifted cloud
    while (np.linalg.norm (clouds_delta - clouds_delta_previous_iteration) > accuracy and iterations < 3000):
        if (verbose):
            print ('\n----------------------------------------------')
            print ('iteration nb.: ' + str(iterations) + ', diff: '
                   + str(np.linalg.norm (clouds_delta - clouds_delta_previous_iteration) ))

        # get correspondences (e.g. nearest neighbors) of aligned_cloud's points in reference_cloud
        distances, correlations = scipy_kdtree.query(aligned_cloud, k=1 )

        # Compute translation between each point pair and sum it up
        point_deltas = np.subtract (reference_cloud[correlations, :], aligned_cloud )
        delta_sum = np.sum (point_deltas, axis=0 )

        # mean transformation of this step
        clouds_delta_previous_iteration = clouds_delta   # store old mean transform for cancellation criteria
        clouds_delta = delta_sum / cloud_size

        # appy transform to aligned point cloud
        aligned_cloud = aligned_cloud + clouds_delta
        translation = translation + clouds_delta  # add transformation to build up final transformation

        # compute mean squared error
        mean_squared_error = (delta_sum**2) / cloud_size
        if (verbose):
            print ('mean_squared_error: ' + str(mean_squared_error ))

        iterations = iterations + 1

    if (iterations >= 3000 ):
        print ("\nToo many iterations (3000). Aborting.")
    else:
        print('\nICP finished in ' + str(time.time() - start_time) + ' seconds' )

    return (translation[0], translation[1], translation[2]), \
           (mean_squared_error[0], mean_squared_error[1], mean_squared_error[2])
