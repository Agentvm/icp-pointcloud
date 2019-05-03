# load
from laspy.file import File
import numpy as np
import pcl
import time
from open3d import read_point_cloud, set_verbosity_level, VerbosityLevel


def load_ply_file (dir_in, file_name ):

    start_time = time.time()
    print('Loading file ' + file_name + ' ...')
    set_verbosity_level(VerbosityLevel.Error)   # so it doesn't print a messy loading bar
    open3d_point_cloud = read_point_cloud(dir_in + file_name )
    points = np.asarray(open3d_point_cloud.points )     # convert to numpy array

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

    # load tile
    start_time = time.time()
    print('Loading file ' + file_name + ' ...')
    with File(dir_in + file_name, mode = 'r') as inFile:
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

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ) + '\n')

    return points


def pcl_load (dir_in, file_name, format = None):
    # Loading
    start_time = time.time()
    print('Loading file ' + file_name + ' ...')
    points = pcl.load(dir_in + file_name, format)

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ) + '\n')

    return points
