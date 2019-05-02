import pcl
import numpy as np


def numpy_to_pcl (input_cloud_numpy):
    """
    Takes a numpy array and returns a pcl cloud

    Input:
        input_cloud_numpy: numpy array with data points and Intensity or RGB values

    Output:
        pcl_cloud: A pcl cloud
    """

    # get number of different values XYZRGB, XYZI or XYZ
    numpy_colums = input_cloud_numpy.shape[1]
    if (numpy_colums < 3 ):
        return pcl.PointCloudXYZ ()     # abort

    if (numpy_colums == 6):     # DIM cloud, with RGB
        # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
        pcl_cloud = pcl.PointCloudXYZRGB(np.array(input_cloud_numpy, dtype=np.float32 ))
    if (numpy_colums == 4):     # ALS cloud, with intensity
        # Python understands float as Float64, which C++ understands as Double, therefore, a conversion is needed.
        pcl_cloud = pcl.PointCloud_PointXYZI (np.array(input_cloud_numpy, dtype=np.float32 ))
    else:   # some other cloud
        first_three_colums = input_cloud_numpy[:, 0:3]
        pcl_cloud = pcl.PointCloudXYZ(np.array(first_three_colums, dtype=np.float32 ))

    return pcl_cloud


def pcl_to_numpy (pcl_cloud ):
    return pcl_cloud.to_array (pcl_cloud)
