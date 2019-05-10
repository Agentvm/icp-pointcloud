import input_output
import scipy.spatial.kdtree
import numpy as np
import time


def icp (input_numpy_cloud_1, input_numpy_cloud_2, kd_tree_of_cloud_2, verbose = False ):
    '''
    Iterative closest point algorithm
    '''

    start_time = time.time()    # measure time

    # clouds
    numpy_cloud_1 = input_numpy_cloud_1.copy ()     # this cloud will be translated
    numpy_cloud_2 = input_numpy_cloud_2             # no copy

    cloud_size = numpy_cloud_1.shape[0]
    translation = [0, 0, 0]  # translation
    clouds_delta = np.array([np.inf, 0, 0])
    clouds_delta_previous_iteration = np.array([0, 0, 0])

    # begin iterations -------------------------------
    iterations = 1

    # iterate while there is a change in the position of the shifted cloud
    while (np.linalg.norm (clouds_delta - clouds_delta_previous_iteration) > 1*10**(-6 )):
        if (verbose):
            print ('\n----------------------------------------------')
            print ('iteration nb.: ' + str(iterations) + ', diff: '
                   + str(np.linalg.norm (clouds_delta - clouds_delta_previous_iteration) ))

        # get correspondences (e.g. nearest neighbors) of numpy_cloud_1's points in numpy_cloud_2
        neighbor_distances, correlations = kd_tree_of_cloud_2.query (numpy_cloud_1 )

        # Compute translation between each point pair ---------
        point_deltas = np.subtract (numpy_cloud_2[correlations, :], numpy_cloud_1 )
        delta_sum = np.sum (point_deltas, axis=0 )

        # mean transformation of this step
        clouds_delta_previous_iteration = clouds_delta   # store old mean transform for cancellation criteria
        clouds_delta = delta_sum / cloud_size

        # appy transform to left point cloud ---------
        numpy_cloud_1 = numpy_cloud_1 + clouds_delta
        translation = translation + clouds_delta  # add transformation to build up final transformation

        # compute mean squared error ---------
        mean_squared_error = (delta_sum**2) / cloud_size
        if (verbose):
            print ('mean_squared_error: ' + str(mean_squared_error ))

        iterations = iterations + 1

    print('\nICP finished in ' + str(time.time() - start_time) + ' seconds' )

    return translation, mean_squared_error


if __name__ == "__main__":

    # load ply files
    numpy_cloud_1 = input_output.load_ply_file ('clouds/laserscanning/', 'plane1.ply')    # 3806 points
    numpy_cloud_2 = numpy_cloud_1 + [0, 2.1, 1]

    # print ('numpy_cloud_1:\n' + str(numpy_cloud_1 ) + '\n')
    # print ('numpy_cloud_2:\n' + str(numpy_cloud_2 ) + '\n')

    # leafsize=200: 79.21 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # leafsize=20: 85.56 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # leafsize=10: 86.81 seconds, translation: [0.02902296 1.99756838 0.90591257]
    # leafsize=5: 97.40 seconds, translation: [0.02902296 1.99756838 0.90591257]
    tree = scipy.spatial.kdtree.KDTree (numpy_cloud_2, leafsize=200 )
    #dists, corr_l_r = tree.query (numpy_cloud_1 )
    # print ('corr_l_r:\n' + str(corr_l_r ) + '\n')
    # print ('numpy_cloud_1[corr_l_r]:\n' + str(numpy_cloud_1[corr_l_r, :] ) + '\n')

    translation, mean_squared_error = icp (numpy_cloud_1, numpy_cloud_2, tree )

    print ("\nSweet Success.")
    print ('translation: ' + str(translation ))
    print ('mean_squared_error: ' + str(mean_squared_error ))
