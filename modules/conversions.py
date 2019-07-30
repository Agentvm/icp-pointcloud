"""
This module contains various pointcloud operations including masking certain rows, sampling and reducing the x, y
coordinated to zero
"""

# local modules
from modules import consensus

# basic imports
import numpy as np
import random

# advanced functionality
import scipy.spatial


def mask_cloud_rows (numpy_cloud, condition_column ):
    """
    Masks numpy_cloud rows depending on the True/False Values defined in condition_column

    --- Usage example:
    from modules import conversions


    numpy_cloud = np.array([[1.1, 2.1, 3.1],
                            [1.2, 2.2, 3.2],
                            [1.3, 2.3, 3.3],
                            [1.4, 2.4, 3.4],
                            [1.5, 2.5, 3.5],
                            [1.6, 2.6, 3.6]] )

    print (conversions.mask_cloud_rows (numpy_cloud, numpy_cloud[:, 0] < 1.4 ))
    --- Example end
    """

    # prepare the mask of boolean_values
    mask_boolean_values = boolean_values = condition_column.reshape (-1, 1)

    # spread the mask along axis=1 of the cloud, until it is the same shape as the cloud
    for i in range (numpy_cloud.shape[1] - 1 ):
        mask_boolean_values = np.concatenate ((mask_boolean_values, boolean_values ), axis=1 )

    return np.ma.masked_array (numpy_cloud, mask=mask_boolean_values )


def mask_cloudpoints_without_correspondence (ref_pointcloud, corr_pointcloud, radius = 0.5 ):
    """
    Remove points in ref_pointcloud and corr_pointcloud which have no neighbor in the other cloud that is nearer than
    radius

    Input:
        ref_pointcloud (NumpyPointCloud):    NumpyPointCloud object containing a numpy array and it's data labels
        corr_pointcloud (NumpyPointCloud):   The cloud corresponding to ref_pointcloud. Both clouds will be modified.
    """

    # build trees
    scipy_kdtree_ref = scipy.spatial.cKDTree (ref_pointcloud[:, 0:3] )
    scipy_kdtree_corr = scipy.spatial.cKDTree (corr_pointcloud[:, 0:3] )

    # # determine the consensus in the current aligment of clouds
    # reference: ref_pointcloud -> consensus_vector: consensus_vector_corr
    _, consensus_vector_corr = consensus.point_distance_cloud_consensus (
        scipy_kdtree_ref, ref_pointcloud, corr_pointcloud, radius )

    # reference: corr_pointcloud  -> consensus_vector: consensus_vector_ref
    _, consensus_vector_ref = consensus.point_distance_cloud_consensus (
        scipy_kdtree_corr, corr_pointcloud, ref_pointcloud, radius )

    # attach the consensus_vector to the clouds
    ref_pointcloud.add_field (consensus_vector_ref, "Consensus" )
    corr_pointcloud.add_field (consensus_vector_corr, "Consensus" )

    # delete all points that did not contribute to the consensus
    truth_vector = ref_pointcloud.get_fields (["Consensus"] ) == 0
    ref_pointcloud.points = mask_cloud_rows (ref_pointcloud.points, truth_vector )
    truth_vector = corr_pointcloud.get_fields (["Consensus"] ) == 0
    corr_pointcloud.points = mask_cloud_rows (corr_pointcloud.points, truth_vector )

    return ref_pointcloud, corr_pointcloud


def sample_cloud (numpy_cloud, sample_divisor, deterministic_sampling=False ):
    '''
    Samples a cloud by a given divisor. If sample_divisor=4, cloud is 4 times as small after sampling.
    '''
    previous_length = numpy_cloud.shape[0]

    # deterministic sampling
    if (deterministic_sampling ):
        numpy_cloud = numpy_cloud[::sample_divisor].copy ()
    # random sampling
    else:
        indices = random.sample(range(0, numpy_cloud.shape[0] ), int (numpy_cloud.shape[0] / sample_divisor ))
        numpy_cloud = numpy_cloud[indices, :].copy ()

    print ("Cloud sampled, divisor: "
           + str(sample_divisor )
           + ". Cloud size / previous cloud size: "
           + str(numpy_cloud.shape[0] )
           + "/"
           + str (previous_length))

    return numpy_cloud


def reduce_cloud (input_cloud_numpy, copy=True, return_transformation=False, return_as_float32=False ):

    # copy to avoid reference disaster
    if (copy ):
        numpy_cloud = input_cloud_numpy.copy ()
    else:
        numpy_cloud = input_cloud_numpy

    # get the minimum x and y values (center of mass would be even better)
    min_x_coordinate = np.min (numpy_cloud[:, 0] )
    min_y_coordinate = np.min (numpy_cloud[:, 1] )

    # reduce coordinates to ensure that the precisison of float32 is enough
    numpy_cloud[:, 0] = numpy_cloud[:, 0] - min_x_coordinate
    numpy_cloud[:, 1] = numpy_cloud[:, 1] - min_y_coordinate

    if (return_as_float32 and return_transformation ):
        return numpy_cloud.astype (np.float32 ), min_x_coordinate, min_y_coordinate
    elif (return_as_float32 ):
        return numpy_cloud.astype (np.float32 )
    elif (return_transformation ):
        return numpy_cloud, min_x_coordinate, min_y_coordinate

    return numpy_cloud


# set the random seed for both the numpy and random module, if it is not already set.
if (random.seed != 1337 or np.random.seed != 1337):
    random.seed = 1337
    np.random.seed = 1337
    print ("Random Seed set to: " + str(random.seed ))
