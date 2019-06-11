# load
from laspy.file import File
import numpy as np
import pcl
import time
from os.path import isfile
from open3d import io, PointCloud, Vector3dVector, read_point_cloud, set_verbosity_level, VerbosityLevel
from custom_clouds import CustomCloud


def save_ascii_file (numpy_cloud, field_labels_list, path ):
    '''
    Saves Pointcloud as ASCII file

    Input:
        numpy_cloud (np.array):             Data array
        field_names_list ([str, str, ...]): List of strings containing the labels of the pointcloud columns. These will
                                            be written to the header of the ascii file
        path (str):                         The path to the file to save
    '''

    print('\nSaving file ' + path )

    # "%.2f %.2f %.2f %.8f %.8f %.8f %.0f %.0f"
    # "%.8f %.8f %.8f %.2f %.2f %.2f"
    format = "%.8f %.8f %.8f"
    for i in range (numpy_cloud.shape[1] - 3 ):
        format = format + " %.2f"

    field_names_list = ['{0} '.format(name) for name in field_labels_list]
    leading_line = "//" + ''.join(field_names_list)

    np.savetxt(path,  # pfad + name
    numpy_cloud,  # numpy array
    header=leading_line,
    comments='',
    fmt = format)  # format


def load_ascii_file (path, return_custom_cloud=False ):

    start_time = time.time()    # measure time
    print('\nLoading file ' + path + ' ...')
    numpy_cloud = np.loadtxt (path, comments='//')

    if (return_custom_cloud ):
        # get the first line of the file, extracting the field labels
        with open(path) as f:
            field_labels_list = f.readline().strip ('//').split ()
        custom_cloud = CustomCloud (numpy_cloud, field_labels_list )

        return custom_cloud

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(numpy_cloud.shape[0] ))

    return numpy_cloud


def check_for_file (path ):
    return isfile(path )


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
    print('\nLoading file ' + file_name + ' ...')

    # Set Debug log to Error, so it doesn't print a messy loading bar, then read the file content
    set_verbosity_level(VerbosityLevel.Error)
    open3d_point_cloud = read_point_cloud(dir_in + file_name )

    # convert to numpy array
    points = np.asarray(open3d_point_cloud.points )

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ))

    return points


def save_ply_file (numpy_cloud, file_name ):
    '''
    Takes a directory path and a filename, then loads a .ply pointcloud file and returns it as numpy array.

    Input:
        numpy_cloud (np.array): The cloud to save
        dir_in (String):        The relative path to the folder that the file to be loaded is in
        file_name (String):     The name of the file to be loaded, including it's file type extension (.ply)
    '''

    # Save a file
    print('\nSaving file ' + file_name )
    if (numpy_cloud.shape[0] == 0):
        print ("This Cloud has no points. Aborting")
        return

    # convert to open3d cloud
    open3d_cloud = PointCloud()
    open3d_cloud.points = Vector3dVector(numpy_cloud[:, 0:3] )

    # Set Debug log to Error, so it doesn't print a messy loading bar, then write out
    set_verbosity_level(VerbosityLevel.Error)
    io.write_point_cloud(file_name, open3d_cloud, write_ascii=True )


def load_las_file (file_path, dtype=None, return_custom_cloud=False ):
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
    print('\nLoading file ' + file_path + ' ...')

    with File(file_path, mode = 'r') as inFile:
        # add points by adding xyz channels. Reshape to create colums
        x = np.reshape(inFile.x.copy(), (-1, 1))
        y = np.reshape(inFile.y.copy(), (-1, 1))
        z = np.reshape(inFile.z.copy(), (-1, 1))

        # add classification channel
        raw_class = np.reshape(inFile.raw_classification.copy(), (-1, 1))

        if dtype == 'dim':
            # add rgb color channels and convert them to float
            red = np.reshape(inFile.red.copy() / 65535.0, (-1, 1))
            green = np.reshape(inFile.green.copy() / 65535.0, (-1, 1))
            blue = np.reshape(inFile.blue.copy() / 65535.0, (-1, 1))
            points = np.concatenate((x, y, z, red, green, blue, raw_class), axis = -1)  # join all values in an np.array
            if (return_custom_cloud ):
                points = CustomCloud (points, ['X', 'Y', 'Z',
                                               'Rf', 'Gf', 'Bf',
                                               'Classification'])

        elif dtype == 'als':
            # add LIDAR intensity
            intensity = np.reshape(inFile.intensity.copy(), (-1, 1))
            num_returns = np.reshape(inFile.num_returns.copy(), (-1, 1))    # number of returns
            return_num = np.reshape(inFile.return_num.copy(), (-1, 1))      # this points' return number
            point_src_id = np.reshape(inFile.pt_src_id.copy(), (-1, 1))    # this points' file origin id

            # join all values in one np.array
            points = np.concatenate((x, y, z, intensity, num_returns, return_num, point_src_id, raw_class), axis = -1)
            if (return_custom_cloud ):
                points = CustomCloud (points,
                                      ['x', 'y', 'z',
                                       'Intensity',
                                       'Number_of_Returns',
                                       'Return_Number',
                                       'Point_Source_ID',
                                       'Classification'])

        else:
            points = np.concatenate((x, y, z, raw_class), axis = -1)  # join all values in one np.array
            if (return_custom_cloud ):
                points = CustomCloud (points, ['x', 'y', 'z', 'raw_class'])

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ))

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
    print('\nLoading file ' + file_name + ' ...')

    #points = pcl.load(dir_in + file_name, format)
    pcl_cloud = pcl.load (dir_in + file_name, format)

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(pcl_cloud.size ))

    return pcl_cloud
