# for testing and demonstration of point cloud workflows

# os
import sys
#import os

# data transformation
import numpy as np
import pcl

# load
from laspy.file import File

# visualization
#from matplotlib import pyplot
#from mpl_toolkits.mplot3d import Axes3D


def load_las_file (dir_in, filename, dtype=None):
    """
    Loads .las data as numpy array

    Inputs:
        dir_in: string; directory in
        filename: String; name of the .las tile (incl. .las)
        dtype: String;
        if dtype = 'als', then the function will return points as [x, y, z, intensity, class]
        if dtype = 'dim', then the function will return points as [x, y, z, r, g, b, class]
        if dtype = None, then the function will return points as [x, y, z, class]
        default: dtype = None

    Outputs:
        points: np array; contains n points with different columns depending on dtype
    """

    # load tile
    print('Loading file ' + file_name + ' ...')
    with File(dir_in + filename, mode = 'r') as inFile:
        x = np.reshape(inFile.x.copy(), (-1, 1))    # create colums
        y = np.reshape(inFile.y.copy(), (-1, 1))
        z = np.reshape(inFile.z.copy(), (-1, 1))
        raw_class = np.reshape(inFile.raw_classification.copy(), (-1, 1))

        if dtype == 'dim':
            red = np.reshape(inFile.red.copy(), (-1, 1))    # add rgb
            green = np.reshape(inFile.green.copy(), (-1, 1))
            blue = np.reshape(inFile.blue.copy(), (-1, 1))
            points = np.concatenate((x, y, z, red, green, blue, raw_class), axis = -1)
        elif dtype == 'als':
            intensity = np.reshape(inFile.intensity.copy(), (-1, 1))    # add intensity
            #num_returns = inFile.num_returns    # number of returns
            #return_num = inFile.return_num      # this points return number
            points = np.concatenate((x, y, z, intensity, raw_class), axis = -1)
        else:
            points = np.concatenate((x, y, z, raw_class), axis = -1)

    print ('Cloud imported.\nNumber of points: ' + str(points.size ) + '\n')
    return points


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


if __name__ == '__main__':

    print ('executed with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) + "\n" )

    # Files
    #clouds_folder = 'clouds/ALS2014/'
    #file_name = '33314059950_05_2014-01-03.las'

    clouds_folder = 'clouds/DIM2016/'
    file_name = 'DSM_Cloud_333140_59950.las'

    #clouds_folder = 'clouds/'
    #file_name = 'laserscanning.ply'

    # Loading
    numpy_cloud = load_las_file (clouds_folder, file_name )

    print (str(numpy_cloud ) + "\n\n" )

    pcl_cloud = numpy_to_pcl (numpy_cloud )
    #pcl_cloud = pcl.load(str(clouds_folder + file_name ))

    print (str(pcl_cloud[:, :, :] ) + "\n\n" ) # how to print a pcl cloud??

    #pcl.save (pcl_cloud, str(clouds_folder) + 'output.ply')
