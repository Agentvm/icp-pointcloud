"""
This module contains various pointcloud operations including pruning clouds by removing certain points, masking certain
columns, sampling and reducing the x, y coordinates to zero
"""

# local modules
from modules import consensus
from modules.normals import normalize_vector_array, einsum_angle_between

# basic imports
import numpy as np
import random

# advanced functionality
import scipy.spatial


def get_distance_consensus (ref_pointcloud, corr_pointcloud, radius):
    """
    Show which points of ref_pointcloud and corr_pointcloud have no neighbor in the other cloud that is nearer than
    radius

    Input:
        ref_pointcloud: (NumpyPointCloud)           NumpyPointCloud object containing a numpy array and it's data labels
        corr_pointcloud: (NumpyPointCloud)          The cloud corresponding to ref_pointcloud.
        radius: (float)                             Max neighbor distance

    Output:
        consensus_vector_ref: ([n, 1] np.array)     Shows 1 where points of ref_pointcloud have a neighbor, 0 otherwise
        consensus_vector_corr: ([n, 1] np.array)    Shows 1 where points of corr_pointcloud have a neighbor, 0 otherwise
    """

    # build trees
    scipy_kdtree_ref = scipy.spatial.cKDTree (ref_pointcloud.points[:, 0:3] )
    scipy_kdtree_corr = scipy.spatial.cKDTree (corr_pointcloud.points[:, 0:3] )

    # # determine the consensus in the current aligment of clouds
    # reference: ref_pointcloud -> consensus_vector: consensus_vector_corr
    _, consensus_vector_corr = consensus.point_distance_cloud_consensus (
        scipy_kdtree_ref, corr_pointcloud, (0, 0, 0), radius )

    # reference: corr_pointcloud  -> consensus_vector: consensus_vector_ref
    _, consensus_vector_ref = consensus.point_distance_cloud_consensus (
        scipy_kdtree_corr, ref_pointcloud, (0, 0, 0), radius )

    return consensus_vector_ref, consensus_vector_corr


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


def mask_cloudpoints_without_correspondence (ref_pointcloud, corr_pointcloud, radius=0.5 ):
    """
    Mask points in ref_pointcloud and corr_pointcloud which have no neighbor in the other cloud that is nearer than
    radius

    Input:
        ref_pointcloud: (NumpyPointCloud)   NumpyPointCloud object containing a numpy array and it's data labels
        corr_pointcloud: (NumpyPointCloud)  The cloud corresponding to ref_pointcloud. Both clouds will be modified.
        radius: (float)                     Max neighbor distance

    Output:
        ref_pointcloud: (NumpyPointCloud)   The altered reference pointcloud
        corr_pointcloud: (NumpyPointCloud)  The altered corresponding pointcloud
    """

    # get consensus column
    consensus_vector_ref, consensus_vector_corr = get_distance_consensus (ref_pointcloud, corr_pointcloud, radius )

    # attach the consensus_vector to the clouds
    ref_pointcloud.add_field (consensus_vector_ref, "Consensus" )
    corr_pointcloud.add_field (consensus_vector_corr, "Consensus" )

    # mask all points that did not contribute to the consensus
    truth_vector = ref_pointcloud.get_fields (["Consensus"] ) == 0
    ref_pointcloud.points = mask_cloud_rows (ref_pointcloud.points, truth_vector )
    truth_vector = corr_pointcloud.get_fields (["Consensus"] ) == 0
    corr_pointcloud.points = mask_cloud_rows (corr_pointcloud.points, truth_vector )

    return ref_pointcloud, corr_pointcloud


def prune_model_outliers (ref_pointcloud, corr_pointcloud, distance=0.5 ):
    """
    Remove points in ref_pointcloud and corr_pointcloud which have no neighbor in the other cloud that is nearer than
    distance (point distance filtering).

    Input:
        ref_pointcloud: (NumpyPointCloud)   NumpyPointCloud object containing a numpy array and it's data labels
        corr_pointcloud: (NumpyPointCloud)  The cloud corresponding to ref_pointcloud. Both clouds will be modified.
        distance: (float)                   Max neighbor distance

    Output:
        ref_pointcloud: (NumpyPointCloud)   The altered reference pointcloud
        corr_pointcloud: (NumpyPointCloud)  The altered corresponding pointcloud
    """

    # get consensus column
    consensus_vector_ref, consensus_vector_corr = get_distance_consensus (ref_pointcloud, corr_pointcloud, distance )

    # delete all points that did not contribute to the consensus
    truth_vector = consensus_vector_ref != 0
    ref_pointcloud.points = ref_pointcloud.points[truth_vector.reshape (-1, ), :]
    truth_vector = consensus_vector_corr != 0
    corr_pointcloud.points = corr_pointcloud.points[truth_vector.reshape (-1, ), :]

    return ref_pointcloud, corr_pointcloud


def prune_cloud_borders (numpy_cloud, clearance=1.2 ):
    """Delete points at the clouds' borders in range of distance, restricting the x-y plane (ground)"""

    # get min/max of cloud
    cloud_max_x = np.max (numpy_cloud[:, 0])
    cloud_min_x = np.min (numpy_cloud[:, 0])
    cloud_max_y = np.max (numpy_cloud[:, 1])
    cloud_min_y = np.min (numpy_cloud[:, 1])

    # define 4 borders
    borders = [cloud_max_x - clearance, cloud_min_x + clearance,
               cloud_max_y - clearance, cloud_min_y + clearance]

    # index all points within borders
    numpy_cloud = numpy_cloud[numpy_cloud[:, 0] < borders[0]]
    numpy_cloud = numpy_cloud[numpy_cloud[:, 0] > borders[1]]
    numpy_cloud = numpy_cloud[numpy_cloud[:, 1] < borders[2]]
    numpy_cloud = numpy_cloud[numpy_cloud[:, 1] > borders[3]]

    return numpy_cloud


def prune_sigma_quality (numpy_pointcloud, max_sigma_value=0.05 ):
    """Fetch the Sigma field of the given cloud and remove all points with a sigma value higher than max_sigma_value"""

    if (numpy_pointcloud.has_fields ("Sigma" )):
        numpy_pointcloud.points = numpy_pointcloud.points[numpy_pointcloud.get_fields ("Sigma" ) <= max_sigma_value]
        return numpy_pointcloud
    return numpy_pointcloud


def remove_water_classes (numpy_pointcloud ):
    """Fetch the Classification field of the given cloud and remove all points with a value == 9"""

    if (numpy_pointcloud.has_fields ("Classification" )):
        numpy_pointcloud.points = numpy_pointcloud.points[numpy_pointcloud.get_fields ("Classification" ) != 9]
        return numpy_pointcloud
    return numpy_pointcloud


def prune_normal_vectors (reference_pointcloud, corresponding_pointcloud, max_angle_difference=32 ):
    """
    Compare the normal vectors of the given cloud pair and remove points whose normal vector angle difference to
    their nearest neighbor is higher than max_angle_difference
    """

    # extract normals
    reference_normals = reference_pointcloud.get_normals ()
    correspondence_normals = corresponding_pointcloud.get_normals ()

    # normalize
    reference_normals = normalize_vector_array (reference_normals )
    correspondence_normals = normalize_vector_array (correspondence_normals )

    # build a kdtree of each cloud
    kdtree_ref = scipy.spatial.cKDTree (reference_pointcloud.points[:, 0:3] )
    kdtree_corr = scipy.spatial.cKDTree (corresponding_pointcloud.points[:, 0:3] )

    # and query it with the other cloud, respectively (dists do not matter)
    dists, corr_indices = kdtree_ref.query (corresponding_pointcloud.points[:, 0:3], k=1 )
    dists, ref_indices = kdtree_corr.query (reference_pointcloud.points[:, 0:3], k=1 )

    # get the angle differences between the normal vectors
    corr_angle_differences = einsum_angle_between (reference_normals[corr_indices, :],
                                                   correspondence_normals ) * (180/np.pi)
    ref_angle_differences = einsum_angle_between (correspondence_normals[ref_indices, :],
                                                  reference_normals ) * (180/np.pi)

    # prune points that do not match
    reference_pointcloud.points[ref_angle_differences > max_angle_difference]
    corresponding_pointcloud.points[corr_angle_differences > max_angle_difference]

    return reference_pointcloud, corresponding_pointcloud


def prune_cloud_pair (reference_pointcloud, corresponding_pointcloud, translation = (0, 0, 0),
                      prune_borders=True, borders_clearance=1.2,
                      prune_water_bodies=True,
                      prune_sigma=True, max_sigma_value=0.05,
                      prune_outliers=True, max_outlier_distance=0.5,
                      prune_normals=True, max_angle_difference=32 ):
    """
    Utilises several pruning methods to remove unwanted points that would otherwise hinder the alignment of two
    corresponding clouds.

    Input:
        reference_pointcloud: (NumpyPointCloud)     NumpyPointCloud object containing a numpy array and it's data labels
        corresponding_pointcloud: (NumpyPointCloud) Pointcloud corresponding to reference_pointcloud
        translation: (3-tuple)                      This translation will be applied to corresponding_pointcloud
        prune_borders: (boolean)                    If true, removes part of the borders of corresponding_pointcloud
        borders_clearance: (float)                  Sets how much of the bordes will be removed (in meters)
        prune_water_bodies: (boolean)               If true and Classification field is present, water is removed
        prune_sigma: (boolean)                      If true and Sigma field is present, high sigma values are removed
        max_sigma_value: (float)                    Sets the highest sigma value to be kept. Higer values are removed
        prune_outliers: (boolean)                   Removed points that have no neigbor in the other cloud
        max_outlier_distance: (float)               Sets the distance that defines outliers
        prune_normals: (boolean)                    If true, normal vectors are compared and those that differ, removed
        max_angle_difference: (float)               Sets the maximum difference between two normal vectors (in degree)

    Output:
        reference_pointcloud: (NumpyPointCloud)     The pruned NumpyPointCloud object of the reference pointcloud
        corresponding_pointcloud: (NumpyPointCloud) The pruned NumpyPointCloud object of the corresponding pointcloud
    """

    # translate
    corresponding_pointcloud.points[:, 0:3] += translation

    # prune cloud outliers
    if (prune_outliers ):
        reference_pointcloud, corresponding_pointcloud = \
            prune_model_outliers (reference_pointcloud, corresponding_pointcloud, max_outlier_distance )

    # # start removing points that are likely to disturb cloud alignment
    if (prune_borders ):
        # only prune the borders of one cloud, so it can be fitted in the reference cloud, avoiding biases introduced
        # through edge phenomenons
        corresponding_pointcloud.points = prune_cloud_borders (corresponding_pointcloud.points, borders_clearance )

    if (prune_water_bodies ):
        reference_pointcloud = remove_water_classes (reference_pointcloud )
        corresponding_pointcloud = remove_water_classes (corresponding_pointcloud )

    if (prune_sigma ):
        reference_pointcloud = prune_sigma_quality (reference_pointcloud, max_sigma_value )
        corresponding_pointcloud = prune_sigma_quality (corresponding_pointcloud, max_sigma_value )

    if (prune_normals ):
        reference_pointcloud, corresponding_pointcloud = \
            prune_normal_vectors (reference_pointcloud, corresponding_pointcloud, max_angle_difference )

    return reference_pointcloud, corresponding_pointcloud


def sample_cloud (numpy_cloud, sample_divisor, deterministic_sampling=False ):
    """Samples a numpy cloud by a given divisor. If sample_divisor=4, cloud is 4 times as small after sampling."""

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


def reduce_cloud (input_cloud_numpy, copy=True, return_transformation=False ):
    """
    Alters cloud coordinates so that the lowest point is at (0, 0, z). High Distances are reduced to values close to
    zero

    Input:
        input_cloud_numpy: (numpy.ndarray)  The Point Cloud
        copy: (boolean)                     Wheather to copy the array before alteration
        return_transformation: (boolean)    If true, numpy_cloud, min_x_coordinate and min_y_coordinate are returned

    Output:
        numpy_cloud: (numpy.ndarray)        The altered points
    """

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

    # return
    if (return_transformation ):
        return numpy_cloud, min_x_coordinate, min_y_coordinate

    return numpy_cloud


# set the random seed for both the numpy and random module, if it is not already set.
random.seed (1337 )
np.random.seed (1337 )
