import pcl
import numpy as np
import random


def get_fields (numpy_cloud, field_labels_list, requested_fields ):

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
                          + ". Compute Normals first.")

    return numpy_cloud[:, indices]


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


def numpy_to_pcl (input_cloud_numpy):
    """
    Takes a numpy array and returns a pcl cloud, and the values that have been subtracted from the cloud to fit the
    values into a float32.

    Input:
        input_cloud_numpy (np.array): numpy array with data points and Intensity or RGB values

    Output:
        pcl_cloud (pcl.PointCloudXYZ):  A pcl cloud
        min_x (float):                  The amount by which the x coordinates have been reduced
        min_y (float):                  The amount by which the y coordinates have been reduced
    """

    # get number of different values XYZRGB, XYZI or XYZ
    numpy_colums = input_cloud_numpy.shape[1]
    if (numpy_colums < 3 ):
        print ('In numpy_to_pcl: Inserted numpy cloud only has ' + str(numpy_colums)
               + ' channels. Returning empty cloud.')
        return pcl.PointCloudXYZ ()     # abort

    # reduce coordinates to ensure that the precision of float32 is enough
    input_cloud_numpy, min_x_coordinate, min_y_coordinate = reduce_cloud (input_cloud_numpy,
                                                                         return_transformation=True )
    # DIM cloud, with RGB
    if (numpy_colums == 6):
        # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
        pcl_cloud = pcl.PointCloudXYZRGB(np.array(input_cloud_numpy, dtype=np.float32 ))
    # ALS cloud, with intensity
    elif (numpy_colums == 4):
        # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
        pcl_cloud = pcl.PointCloud_PointXYZI (np.array(input_cloud_numpy, dtype=np.float32 ))
    # some other cloud
    else:
        first_three_colums = input_cloud_numpy[:, 0:3]
        # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
        pcl_cloud = pcl.PointCloudXYZ(np.array(first_three_colums, dtype=np.float32 ))

    return pcl_cloud, min_x_coordinate, min_y_coordinate


def pcl_to_numpy (pcl_cloud ):
    """
    Wraps PCL's to_array function to return an np.array.
    """
    return pcl_cloud.to_array (pcl_cloud)


if (random.seed != 1337):
    random.seed = 1337
    print ("Random Seed set to: " + str(random.seed ))
