import pcl
import numpy as np


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
    min_x_coordinate = input_cloud_numpy[:, 0].min
    min_y_coordinate = input_cloud_numpy[:, 1].min
    input_cloud_numpy[:, 0] = input_cloud_numpy[:, 0] - min_x_coordinate
    input_cloud_numpy[:, 1] = input_cloud_numpy[:, 1] - min_y_coordinate

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
