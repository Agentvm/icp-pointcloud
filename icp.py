"""Import this for a simple iterative closest point algorithm"""

import input_output
import sklearn.neighbors    # kdtree
import numpy as np
import time                 # measure time
import itertools            # speed improvement when making a [list] out of a [list of [lists]]


def icp (input_numpy_cloud_1, input_numpy_cloud_2, kd_tree_of_cloud_2, verbose = False ):
    '''
    Iterative closest point algorithm that computes the translation from input_numpy_cloud_1 to input_numpy_cloud_2.

    Input:
        input_numpy_cloud_1 (np.array):                 Input Cloud. This Cloud will be moved to match cloud_2.
        input_numpy_cloud_2 (np.array):                 Input Cloud. This Cloud will stay fixed in place.
        kd_tree_of_cloud_2 (sklearn.neighbors.KDTree):  A kd search tree build with the sklern library.

    Output:
        translation ((x, y, z) tuple):          The estimated translation between input_numpy_cloud_1
                                                and input_numpy_cloud_2.
        mean_squared_error ((x, y, z) tuple):   The remaining MSE in x, y and z direction
    '''

    start_time = time.time()    # measure time

    # clouds
    numpy_cloud_1 = input_numpy_cloud_1.copy ()     # this cloud will be translated
    numpy_cloud_2 = input_numpy_cloud_2             # no copy

    # preparing variables
    cloud_size = numpy_cloud_1.shape[0]
    translation = [0, 0, 0]
    iterations = 1

    # mean difference between the clouds
    # (set to inf to avoid exit criterium in the first iteration)
    clouds_delta = np.array([np.inf, 0, 0])
    clouds_delta_previous_iteration = np.array([0, 0, 0])

    # iterate while there is a change in the position of the shifted cloud
    while (np.linalg.norm (clouds_delta - clouds_delta_previous_iteration) > 1*10**(-12 )):
        if (verbose):
            print ('\n----------------------------------------------')
            print ('iteration nb.: ' + str(iterations) + ', diff: '
                   + str(np.linalg.norm (clouds_delta - clouds_delta_previous_iteration) ))

        # get correspondences (e.g. nearest neighbors) of numpy_cloud_1's points in numpy_cloud_2
        # list(itertools.chain(*tree.query())  --> sklearn returns a list of lists,
        # which needs to be reduced to a list to be used in this context
        correlations = list(itertools.chain(*tree.query (numpy_cloud_1, k=1, return_distance=False )))

        # Compute translation between each point pair and sum it up
        point_deltas = np.subtract (numpy_cloud_2[correlations, :], numpy_cloud_1 )
        delta_sum = np.sum (point_deltas, axis=0 )

        # mean transformation of this step
        clouds_delta_previous_iteration = clouds_delta   # store old mean transform for cancellation criteria
        clouds_delta = delta_sum / cloud_size

        # appy transform to first point cloud
        numpy_cloud_1 = numpy_cloud_1 + clouds_delta
        translation = translation + clouds_delta  # add transformation to build up final transformation

        # compute mean squared error
        mean_squared_error = (delta_sum**2) / cloud_size
        if (verbose):
            print ('mean_squared_error: ' + str(mean_squared_error ))

        iterations = iterations + 1

    print('ICP finished in ' + str(time.time() - start_time) + ' seconds' )

    return translation, mean_squared_error


if __name__ == "__main__":

    # load ply files
    numpy_cloud_1 = input_output.load_ply_file ('clouds/laserscanning/', 'plane1.ply')    # 3806 points
    numpy_cloud_2 = numpy_cloud_1 + [0, 2.1, 1]

    # print clouds
    # print ('numpy_cloud_1:\n' + str(numpy_cloud_1 ) + '\n')
    # print ('numpy_cloud_2:\n' + str(numpy_cloud_2 ) + '\n')

    # test scipy tree   #############################################################
    # leafsize=200: 79.21 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # leafsize=20: 85.56 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # leafsize=10: 86.81 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # leafsize=5: 97.40 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # tree = scipy.spatial.kdtree.KDTree (numpy_cloud_2, leafsize=200 )
    # dists, corr_l_r = tree.query (numpy_cloud_1 )
    # print ('corr_l_r:\n' + str(corr_l_r[:3] ) + '\n')
    # print ('numpy_cloud_1[corr_l_r]:\n' + str(numpy_cloud_1[corr_l_r[:3], :] ) + '\n')

    # # test sklearn tree   #########################################################
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud_2, leaf_size=40)
    # correlations = list(itertools.chain(*tree.query (numpy_cloud_1, k=1, return_distance=False )))
    # print ("correlations: " + str (correlations[:3] ))
    # print ('numpy_cloud_1[correlations]:\n' + str(numpy_cloud_1[correlations[:3], :] ) + '\n')

    # do icp    #####################################################################
    translation, mean_squared_error = icp (numpy_cloud_1, numpy_cloud_2, tree )
    print ("\nSweet Success.")
    print ('translation: ' + str(translation ))
    print ('mean_squared_error: ' + str(mean_squared_error ))
