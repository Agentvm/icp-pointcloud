# load
from laspy.file import File
import numpy as np
import pcl
import time
from open3d import read_point_cloud, set_verbosity_level, VerbosityLevel


def load_ply_file (dir_in, file_name ):
    '''
    Takes a directory path and a filename, then loads a .ply pointcloud file and returns it as numpy array.

    Input:
        dir_in (String):     The relative path to the folder that the file to be loaded is in
        file_name (String):  The name of the file to be loaded, including it's file type extension (.ply)

    Output:
        points (np.array):   The numpy array containing the loaded points is of shape (n, 3).
    '''

    # Load a file
    start_time = time.time()    # measure time
    print('Loading file ' + file_name + ' ...')

    # Set Debug log to Error, so it doesn't print a messy loading bar, then read the file content
    set_verbosity_level(VerbosityLevel.Error)
    open3d_point_cloud = read_point_cloud(dir_in + file_name )

    # convert to numpy array
    points = np.asarray(open3d_point_cloud.points )

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ) + '\n')

    return points


def load_las_file (dir_in, file_name, dtype=None):
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

    # Load a file
    start_time = time.time()    # measure time
    print('Loading file ' + file_name + ' ...')

    with File(dir_in + file_name, mode = 'r') as inFile:
        # add points by adding xyz channels. Reshape to create colums
        x = np.reshape(inFile.x.copy(), (-1, 1))
        y = np.reshape(inFile.y.copy(), (-1, 1))
        z = np.reshape(inFile.z.copy(), (-1, 1))

        # add classification channel
        raw_class = np.reshape(inFile.raw_classification.copy(), (-1, 1))

        if dtype == 'dim':
            # add rgb color channels
            red = np.reshape(inFile.red.copy(), (-1, 1))
            green = np.reshape(inFile.green.copy(), (-1, 1))
            blue = np.reshape(inFile.blue.copy(), (-1, 1))
            points = np.concatenate((x, y, z, red, green, blue, raw_class), axis = -1)  # join all values in an np.array
        elif dtype == 'als':
            # add LIDAR intensity
            intensity = np.reshape(inFile.intensity.copy(), (-1, 1))
            #num_returns = inFile.num_returns    # number of returns
            #return_num = inFile.return_num      # this points return number
            points = np.concatenate((x, y, z, intensity, raw_class), axis = -1)  # join all values in one np.array
        else:
            points = np.concatenate((x, y, z, raw_class), axis = -1)  # join all values in one np.array

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ) + '\n')

    return points


def pcl_load (dir_in, file_name, format = None):
    '''
    Takes a directory path and a filename, then loads a pointcloud file and returns it as pcl cloud.

    Input:
        dir_in (String):            The relative path to the folder that the file to be loaded is in
        file_name (String):         The name of the file to be loaded, including it's file type extension
        format (String):            Default is None. If format=None, file format is determined automatically

    Output:
        points (pcl.PointCloudXYZ): The pcl object containing the loaded points
    '''

    # Load a file
    start_time = time.time()    # measure time
    print('Loading file ' + file_name + ' ...')

    #points = pcl.load(dir_in + file_name, format)
    pcl_cloud = pcl.load (dir_in + file_name, format)

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(pcl_cloud.size ) + '\n')

    return pcl_cloud
