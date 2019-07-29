"""
Loading and saving Point Cloud .las and ASCII (.asc) files
"""

# load
from laspy.file import File
import numpy as np
#import pcl
import time
from open3d import io, PointCloud, Vector3dVector, read_point_cloud, set_verbosity_level, VerbosityLevel
from os import listdir, walk
from os.path import isfile, join, splitext
import pickle


def save_ascii_file (numpy_cloud, field_labels_list, path ):
    '''
    Saves Pointcloud as ASCII file

    Input:
        numpy_cloud (np.ndarray):             Data array
        field_names_list ([str, str, ...]): List of strings containing the labels of the pointcloud columns. These will
                                            be written to the header of the ascii file
        path (str):                         The path to the file to save
    '''

    print('\nSaving file ' + path )

    # "%.2f %.2f %.2f %.8f %.8f %.8f %.0f %.0f"
    # "%.8f %.8f %.8f %.2f %.2f %.2f"
    format = "%.8f %.8f %.8f"
    for i in range (numpy_cloud.shape[1] - 3 ):
        format = format + " %.6f"

    field_names_list = ['{0} '.format(name) for name in field_labels_list]
    leading_line = "//" + ''.join(field_names_list)

    np.savetxt(path,  # pfad + name
    numpy_cloud,  # numpy array
    header=leading_line,
    comments='',
    fmt = format)  # format


def load_ascii_file (path ):

    start_time = time.time()    # measure time
    print('\nLoading file ' + path + ' ...')
    numpy_cloud = np.loadtxt (path, comments='//')
    with open(path) as f:
        field_labels_list = f.readline().strip ('//').split ()

    print ("Field labels: " + str (field_labels_list ))
    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(numpy_cloud.shape[0] ))

    return numpy_cloud, field_labels_list


def save_obj (object, name ):
    '''Saves a python object into data/ folder, using pickle'''
    print ("\n Saving file " + 'data/' + name + '.pkl')
    with open('data/' + name + '.pkl', 'wb') as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)


def load_obj (name ):
    '''Loads a python object from the data/ folder, using pickle'''
    with open('data/' + name + '.pkl', 'rb') as file:
        return pickle.load(file )


def join_saved_dictionaries (list_of_dict_names, output_name ):
    resulting_dictionary = {}

    for dict_name in list_of_dict_names:
        resulting_dictionary.update (load_obj (dict_name ))

    save_obj (resulting_dictionary, output_name )

    return True


def check_for_file (path ):
    return isfile(path )


def get_matching_filenames(filename):
    """
    finds matching filenames from one point cloud to another.
    Matches ALS16 to DIM16
    Inputs:
    filename: string; filename of the original file to split
    Outputs:
    [s1, s2, s3, s4]: list of string;
    s1: xmin and ymin
    s2: xmin and ymean
    s3: xmean and ymin
    s4: xmean and ymean
    """
    # get filenames
    s = filename.split('_')[0]

    # get minimal and mean values
    xmin = int(s[0:5] + '0')
    ymin = int(s[5:])
    xmean = xmin + 5
    ymean = ymin + 5

    # build file strings
    prep = 'DSM_Cloud_'
    ending = '.las'
    s1 = "".join([prep, str(xmin), '_', str(ymin), ending])
    s2 = "".join([prep, str(xmin), '_', str(ymean), ending])
    s3 = "".join([prep, str(xmean), '_', str(ymin), ending])
    s4 = "".join([prep, str(xmean), '_', str(ymean), ending])
    return [s1, s2, s3, s4]

# y >= 40 ###
# x = 30
#print (get_matching_filenames ("3331359940_1_2016-11-28.las" ))     # Y/YZ (Train)
#print (get_matching_filenames ("3331359950_1_2016-11-28.las" ))    # Forest
#print (get_matching_filenames ("3331359960_1_2016-11-28.las" ))

# x = 40
#print (get_matching_filenames ("3331459940_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331459950_1_2016-11-28.las" ))    # DIM showcase, missing Building
#print (get_matching_filenames ("3331459960_1_2016-11-28.las" ))

# x = 50
#print (get_matching_filenames ("3331559940_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331559950_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331559960_1_2016-11-28.las" ))

# x = 60
#print (get_matching_filenames ("3331659940_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331659950_1_2016-11-28.las" ))    # XZ, everything, XYZ, Acker, XY, Fahrbahn
#print (get_matching_filenames ("3331659960_1_2016-11-28.las" ))


def get_all_files_in_subfolders (path_to_folder, permitted_file_extension=None ):
    '''
    Finds all files inside the folders below the given folder (1 level below)
    '''

    # find all directories below path_to_folder
    dirnames = []
    f = []
    for (dirpath, dirnames, file_names) in walk(path_to_folder):
        f.extend(file_names)
        break

    # append the directories to the input directory
    # add the input directory itself, so files in there will be found
    full_directories = [path_to_folder.strip ('/')]
    for dir in dirnames:
        dir = path_to_folder + dir
        full_directories.append (dir)

    #for every directory found, find all files inside and append the resulting path to each file to full_paths
    full_paths = []
    for dir_count, directory in enumerate (full_directories ):
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        for file in onlyfiles:
            full_paths.append (full_directories[dir_count] + '/' + file )

    # if specified, remove all file extensions that do not match the specified extension
    paths_to_remove = []
    if (permitted_file_extension is not None ):
        for path in full_paths:
            file_name, file_extension = splitext(path )
            if (file_extension != permitted_file_extension):
                paths_to_remove.append (path )
    full_paths = [path for path in full_paths if path not in paths_to_remove]

    return full_paths


def get_folder_and_file_name (path ):

    # mash up the string
    folder = str(path.split ('/')[-2])
    list_of_filename_attributes = path.split ('/')[-1].split ('_')[0:3]
    list_of_filename_attributes = ['{0}_'.format(element) for element in list_of_filename_attributes]
    file_name = ''.join(list_of_filename_attributes)

    return folder, file_name


def load_ply_file (dir_in, file_name ):
    '''
    Takes a directory path and a filename, then loads a .ply pointcloud file and returns it as numpy array.

    Input:
        dir_in (String):     The relative path to the folder that the file to be loaded is in
        file_name (String):  The name of the file to be loaded, including it's file type extension (.ply)

    Output:
        points (np.ndarray):   The numpy array containing the loaded points is of shape (n, 3).
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
        numpy_cloud (np.ndarray): The cloud to save
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


def load_las_file (file_path, dtype=None ):
    """
    Loads .las data as numpy array

    Inputs:
        dir_in (String): directory in
        filename (String): name of the .las tile (incl. .las)
        dtype (String):
        if dtype = 'als', then the function will return points as [x, y, z, intensity, class]
        if dtype = 'dim', then the function will return points as [x, y, z, r, g, b, class]
        if dtype = None, then the function will return points as [x, y, z, class]
        default: dtype = None

    Outputs:
        points: np array; contains n points with different columns depending on dtype
        field_labels_list:
    """

    # Load a file
    start_time = time.time()    # measure time
    print('\nLoading file ' + file_path + ' ...')

    field_labels_list = []
    with File(file_path, mode = 'r') as inFile:
        # add points by adding xyz channels. Reshape to create colums
        x = np.reshape(inFile.x.copy(), (-1, 1))
        y = np.reshape(inFile.y.copy(), (-1, 1))
        z = np.reshape(inFile.z.copy(), (-1, 1))

        if dtype == 'dim':
            # add rgb color channels and convert them to float
            red = np.reshape(inFile.red.copy() / 65535.0, (-1, 1))
            green = np.reshape(inFile.green.copy() / 65535.0, (-1, 1))
            blue = np.reshape(inFile.blue.copy() / 65535.0, (-1, 1))

            # join all values in one np.array and update the field labels to allow safe access of colums
            points = np.concatenate((x, y, z, red, green, blue), axis = -1)  # join all values in an np.array
            field_labels_list += ['X', 'Y', 'Z', 'Rf', 'Gf', 'Bf']


        elif dtype == 'als':
            # extract the scalar fields of the .las cloud
            intensity = np.reshape(inFile.intensity.copy(), (-1, 1))            # add LIDAR intensity
            num_returns = np.reshape(inFile.num_returns.copy(), (-1, 1))        # number of returns
            return_num = np.reshape(inFile.return_num.copy(), (-1, 1))          # this points' return number
            point_src_id = np.reshape(inFile.pt_src_id.copy(), (-1, 1))         # this points' file origin id
            raw_class = np.reshape(inFile.raw_classification.copy(), (-1, 1))   # add classification channel

            # join all values in one np.array and update the field labels to allow safe access of colums
            points = np.concatenate((x, y, z, intensity, num_returns, return_num, point_src_id, raw_class), axis = -1)
            field_labels_list += [
                'X', 'Y', 'Z', 'Intensity', 'Number_of_Returns', 'Return_Number', 'Point_Source_ID', 'Classification']


        else:
            points = np.concatenate((x, y, z, raw_class), axis = -1)  # join all values in one np.array

    print ("Field labels: " + str (field_labels_list ))
    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ))

    return points, field_labels_list


# def pcl_load (dir_in, file_name, format = None):
#     '''
#     Takes a directory path and a filename, then loads a pointcloud file and returns it as pcl cloud.
#
#     Input:
#         dir_in (String):            The relative path to the folder that the file to be loaded is in
#         file_name (String):         The name of the file to be loaded, including it's file type extension
#         format (String):            Default is None. If format=None, file format is determined automatically
#
#     Output:
#         points (pcl.PointCloudXYZ): The pcl object containing the loaded points
#     '''
#
#     # Load a file
#     start_time = time.time()    # measure time
#     print('\nLoading file ' + file_name + ' ...')
#
#     #points = pcl.load(dir_in + file_name, format)
#     pcl_cloud = pcl.load (dir_in + file_name, format)
#
#     print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
#            + str(pcl_cloud.size ))
#
#     return pcl_cloud


def conditionalized_load (file_path ):
    '''
    Loads .las and .asc files.

    Input:
        file_path (string):     The path to the file to load. Include file extension.

    Output:
        numpy_cloud (np.ndarray): The cloud values, fitted in a numpy nd array
        field_labels_list:      The header of the file, containing the labels of the cloud fields (column titles)
    '''

    numpy_cloud = None
    #field_labels_list = []
    file_name, file_extension = splitext(file_path )

    # # load the file
    if (file_extension == '.las'):
        if ("DSM_Cloud" in file_path):
            # Load DIM cloud
            numpy_cloud, field_labels_list = load_las_file (file_path, dtype="dim" )
            #field_labels_list += labels_list_loaded
        else:
            # Load ALS cloud
            numpy_cloud, field_labels_list = load_las_file (file_path, dtype="als")
            #field_labels_list += labels_list_loaded

    elif (file_extension == '.asc'):
        # load ASCII cloud
        numpy_cloud, field_labels_list = load_ascii_file (file_path )
        # with open(file_path) as f:
        #     field_labels_list += f.readline().strip ('//').split ()

    return numpy_cloud, field_labels_list
