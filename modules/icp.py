"""Import this for a simple iterative closest point algorithm"""

import sklearn.neighbors    # kdtree
import numpy as np
import time                 # measure time
import itertools            # speed improvement when making a [list] out of a [list of [lists]]


def icp (numpy_reference_cloud, numpy_aligned_cloud, accuracy=0.000001, verbose=False ):
    '''
    Iterative closest point algorithm that computes the translation from numpy_aligned_cloud to numpy_reference_cloud.

    Input:
        input_reference_cloud (np.array):       Input Cloud. This Cloud will stay fixed in place.
        input_aligned_cloud (np.array):         Input Cloud. This Cloud will be moved to match the reference cloud.
        accuracy (float):                       The desired accuracy of alignment in meters

    Output:
        translation ((x, y, z) tuple):          The estimated translation between aligned_cloud and reference_cloud.
        mean_squared_error ((x, y, z) tuple):   The remaining MSE in x, y and z direction
    '''

    start_time = time.time()    # measure time

    # clouds
    reference_cloud = numpy_reference_cloud[:, 0:3]           # no copy
    aligned_cloud = numpy_aligned_cloud[:, 0:3].copy ()     # this cloud will be translated

    # preparing variables
    cloud_size = aligned_cloud.shape[0]
    translation = [0, 0, 0]
    iterations = 1

    # build a kdtree
    tree = sklearn.neighbors.kd_tree.KDTree (reference_cloud, leaf_size=40, metric='euclidean')

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
        # list(itertools.chain(*tree.query())  --> sklearn returns a list of lists,
        # which needs to be reduced to a list to be used in this context
        correlations = list(itertools.chain(*tree.query (aligned_cloud, k=1, return_distance=False )))

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
        print ("\nToo many iterations (100000). Aborting.")
    else:
        print('\nICP finished in ' + str(time.time() - start_time) + ' seconds' )

    return (translation[0], translation[1], translation[2]), \
           (mean_squared_error[0], mean_squared_error[1], mean_squared_error[2])
