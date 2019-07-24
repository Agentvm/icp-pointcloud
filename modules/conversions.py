#import pcl
import numpy as np
import random
from modules import consensus
import scipy.spatial


def get_fields (numpy_cloud, field_labels_list, requested_fields ):
    '''
    Input:
        requested_fields (list(string)):    The names of the fields to be returned, in a list
    '''

    # remove any spaces around the labels
    field_labels_list = [label.strip () for label in field_labels_list]

    if (requested_fields is not None
       and all(field in field_labels_list for field in requested_fields ) ):
        indices = []
        for field in requested_fields:
            indices.append (field_labels_list.index(field ))
    else:
        raise ValueError ("This Cloud is missing one of the requested fields: "
                          + str(requested_fields)
                          + ".\nCloud fields are: " + str(field_labels_list ))

    return numpy_cloud[:, indices]


def add_field (numpy_cloud, numpy_cloud_field_labels, field, field_name ):
    numpy_cloud = np.concatenate ((numpy_cloud, field), axis=1 )
    numpy_cloud_field_labels += [field_name]

    return numpy_cloud, numpy_cloud_field_labels


def add_fields (numpy_cloud, numpy_cloud_field_labels, field, field_name ):
    raise NotImplementedError


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


def mask_cloudpoints_without_correspondence (ref_cloud, ref_labels,
                                             corr_cloud, corr_labels,
                                             radius = 0.5):

    # build trees
    scipy_kdtree_ref = scipy.spatial.cKDTree (ref_cloud[:, 0:3] )
    scipy_kdtree_corr = scipy.spatial.cKDTree (corr_cloud[:, 0:3] )

    # # determine the consensus in the current aligment of clouds
    # reference: ref_cloud -> consensus_vector: consensus_vector_corr
    _, consensus_vector_corr = consensus.point_distance_cloud_consensus (
        scipy_kdtree_ref, ref_cloud, corr_cloud, radius )

    # reference: corr_cloud  -> consensus_vector: consensus_vector_ref
    _, consensus_vector_ref = consensus.point_distance_cloud_consensus (
        scipy_kdtree_corr, corr_cloud, ref_cloud, radius )

    # attach the consensus_vector to the clouds
    ref_cloud, ref_labels = add_field (ref_cloud, ref_labels, consensus_vector_ref, "Consensus" )
    corr_cloud, corr_labels = add_field (corr_cloud, corr_labels, consensus_vector_corr, "Consensus" )

    # delete all points that did not contribute to the consensus
    truth_vector = get_fields (ref_cloud, ref_labels, ["Consensus"] ) == 0
    ref_cloud = mask_cloud_rows (ref_cloud, truth_vector )
    truth_vector = get_fields (corr_cloud, corr_labels, ["Consensus"] ) == 0
    corr_cloud = mask_cloud_rows (corr_cloud, truth_vector )

    return ref_cloud, ref_labels, corr_cloud, corr_labels


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


# def numpy_to_pcl (input_cloud_numpy):
#     """
#     Takes a numpy array and returns a pcl cloud, and the values that have been subtracted from the cloud to fit the
#     values into a float32.
#
#     Input:
#         input_cloud_numpy (np.array): numpy array with data points and Intensity or RGB values
#
#     Output:
#         pcl_cloud (pcl.PointCloudXYZ):  A pcl cloud
#         min_x (float):                  The amount by which the x coordinates have been reduced
#         min_y (float):                  The amount by which the y coordinates have been reduced
#     """
#
#     # get number of different values XYZRGB, XYZI or XYZ
#     numpy_colums = input_cloud_numpy.shape[1]
#     if (numpy_colums < 3 ):
#         print ('In numpy_to_pcl: Inserted numpy cloud only has ' + str(numpy_colums)
#                + ' channels. Returning empty cloud.')
#         return pcl.PointCloudXYZ ()     # abort
#
#     # reduce coordinates to ensure that the precision of float32 is enough
#     input_cloud_numpy, min_x_coordinate, min_y_coordinate = reduce_cloud (input_cloud_numpy,
#                                                                          return_transformation=True )
#     # DIM cloud, with RGB
#     if (numpy_colums == 6):
#         # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
#         pcl_cloud = pcl.PointCloudXYZRGB(np.array(input_cloud_numpy, dtype=np.float32 ))
#     # ALS cloud, with intensity
#     elif (numpy_colums == 4):
#         # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
#         pcl_cloud = pcl.PointCloud_PointXYZI (np.array(input_cloud_numpy, dtype=np.float32 ))
#     # some other cloud
#     else:
#         first_three_colums = input_cloud_numpy[:, 0:3]
#         # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
#         pcl_cloud = pcl.PointCloudXYZ(np.array(first_three_colums, dtype=np.float32 ))
#
#     return pcl_cloud, min_x_coordinate, min_y_coordinate


# def pcl_to_numpy (pcl_cloud ):
#     """
#     Wraps PCL's to_array function to return an np.array.
#     """
#     return pcl_cloud.to_array (pcl_cloud)


if (random.seed != 1337 or np.random.seed != 1337):
    random.seed = 1337
    np.random.seed = 1337
    print ("Random Seed set to: " + str(random.seed ))
