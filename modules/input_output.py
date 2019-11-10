"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

Input and output of Point Cloud formats. Loads ASCII (.asc), .ply and .las files. Saves .las and .asc files. Also
contains functions for saving python objects like dictionaries and convenience functions for filepath discovery.
"""

# local modules
from modules.np_pointcloud import NumpyPointCloud

# basic imports
import numpy as np
import time
from os import listdir, walk
import os.path

# advanced functionality
from laspy.file import File
from open3d import io, PointCloud, Vector3dVector, read_point_cloud, set_verbosity_level, VerbosityLevel
import pickle


def save_ascii_file (numpy_cloud, field_labels, file_path ):
    """
    Saves Pointcloud as ASCII file

    Input:
        numpy_cloud: (np.ndarray)           Input Cloud. Minimum size is (n, 3) for the X, Y, Z point data
        field_labels: (list(string))        Labels of the columns of this cloud, describing the type of data
        file_path: (string)                 The full path including filename and extension
    """

    print('\nSaving file ' + file_path )

    # define coordinate precision
    format = "%.8f %.8f %.8f"

    # define precision of the other data fields
    for i in range (numpy_cloud.shape[1] - 3 ):
        format = format + " %.6f"

    # format the header of the ascii file containing the descriptions of data column content
    field_names_list = ['{0} '.format(name) for name in field_labels]
    leading_line = "//" + ''.join(field_names_list)

    # save
    np.savetxt(file_path, numpy_cloud, header=leading_line, comments='', fmt=format )

    return True


def load_ascii_file (file_path, return_separate=False ):
    """
    Loads an ASCII pointcloud and returns a NumpyPointCloud object

    Input:
        file_path: (string)                 The full path including filename and extension
        return_separate: (boolean)          If True, returns a numpy.ndarray and a list of field labels

    Output:
        NumpyPointCloud, containing:
            points: (np.ndarray)            Point Cloud data. Point coordinates and additional colums (fields)
            field_labels: (list(string))    Labels of the columns of the cloud, describing the type of data
    """

    start_time = time.time()    # measure time

    print('\nLoading file ' + file_path + ' ...')

    # load the file, extracting the header of the file as field labels list
    numpy_cloud = np.loadtxt (file_path, comments='//')
    with open(file_path) as f:
        field_labels_list = f.readline().strip ('//').split ()

    print ("Field labels: " + str (field_labels_list ))
    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(numpy_cloud.shape[0] ))

    if (return_separate ):
        return numpy_cloud, field_labels_list
    return NumpyPointCloud(numpy_cloud, field_labels_list )


def save_ply_file (numpy_cloud, file_path ):
    """
    Takes a file_path, then loads a .ply pointcloud file and returns it as numpy array.

    Input:
        numpy_cloud: (np.ndarray)   The cloud to save
        file_path: (string)         The relative path to the folder that the file to be loaded is in
        file_name: (String)         The name of the file to be loaded, including it's file type extension (.ply)
    """

    # Save a file
    print('\nSaving file ' + file_path )
    if (numpy_cloud.shape[0] == 0):
        print ("This Cloud has no points. Aborting")
        return

    # convert to open3d cloud
    open3d_cloud = PointCloud()
    open3d_cloud.points = Vector3dVector(numpy_cloud[:, 0:3] )

    # Set Debug log to Error, so it doesn't print a messy loading bar, then write out
    set_verbosity_level(VerbosityLevel.Error)
    io.write_point_cloud(file_path, open3d_cloud, write_ascii=True )


def load_ply_file (file_path ):
    """
    Takes a file_path, then loads a .ply pointcloud file and returns it as numpy array.

    Input:
        file_path: (string)     The full path including filename and extension

    Output:
        points: (np.ndarray)    The numpy array containing the loaded points is of shape (n, 3).
    """

    # Load a file
    start_time = time.time()    # measure time
    print('\nLoading file ' + file_path + ' ...')

    # Set Debug log to Error, so it doesn't print a messy loading bar, then read the file content
    set_verbosity_level(VerbosityLevel.Error)
    open3d_point_cloud = read_point_cloud(file_path )

    # convert to numpy array
    points = np.asarray(open3d_point_cloud.points )

    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: '
           + str(points.shape[0] ))

    return points


def load_las_file (file_path, dtype=None, return_separate=False ):
    """
    Copyright Florian Politz

    Loads .las data and returns it as NumpyPointCloud

    Inputs:
        file_path: (string)                 The path to the file, including extension
        return_separate: (boolean)          If True, returns a numpy.ndarray and a list of field labels
        dtype (String):                     Method of scan
            if dtype = 'als': function will return points as ['X', 'Y', 'Z', I, Nr, Rn, Id, 'Classification']
            if dtype = 'dim': function will return points as ['X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Classification']
            if dtype = None: function will return points as ['X', 'Y', 'Z', 'Classification']
            default: dtype = None

    Output:
        NumpyPointCloud, containing:
            points: (np.ndarray)            Point Cloud data. Point coordinates and additional colums (fields)
            field_labels: (list(string))    Labels of the columns of the cloud, describing the type of data
    """

    start_time = time.time()    # measure time
    print('\nLoading file ' + file_path + ' ...')

    # Load a file
    with File(file_path, mode = 'r') as inFile:

        # add points by adding xyz channels. Reshape to create colums
        x = np.reshape(inFile.x.copy(), (-1, 1))
        y = np.reshape(inFile.y.copy(), (-1, 1))
        z = np.reshape(inFile.z.copy(), (-1, 1))

        # add the classification channel
        raw_class = np.reshape(inFile.raw_classification.copy(), (-1, 1))

        # Load Dense Image matching cloud
        if dtype == 'dim':

            # add rgb color channels and convert them to float
            red = np.reshape(inFile.red.copy() / 65535.0, (-1, 1))
            green = np.reshape(inFile.green.copy() / 65535.0, (-1, 1))
            blue = np.reshape(inFile.blue.copy() / 65535.0, (-1, 1))

            # join all values in one np.array and update the field labels to allow safe access of colums
            points = np.concatenate((x, y, z, red, green, blue, raw_class), axis = -1)  # join all values in an np.array
            field_labels_list = ['X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Classification']

        # Load Airborne Laserscanning cloud
        elif dtype == 'als':

            # extract the scalar fields of the .las cloud
            intensity = np.reshape(inFile.intensity.copy(), (-1, 1))            # add LIDAR intensity
            num_returns = np.reshape(inFile.num_returns.copy(), (-1, 1))        # number of returns
            return_num = np.reshape(inFile.return_num.copy(), (-1, 1))          # this points' return number
            point_src_id = np.reshape(inFile.pt_src_id.copy(), (-1, 1))         # this points' file origin id

            # join all values in one np.array and update the field labels to allow safe access of colums
            points = np.concatenate((x, y, z, intensity, num_returns, return_num, point_src_id, raw_class), axis = -1)
            field_labels_list = [
                'X', 'Y', 'Z', 'Intensity', 'Number_of_Returns', 'Return_Number', 'Point_Source_ID', 'Classification']

        # Load some other cloud
        else:

            # join all values in one np.array and update the field labels to allow safe access of colums
            field_labels_list = ['X', 'Y', 'Z', 'Classification']
            points = np.concatenate((x, y, z, raw_class), axis = -1)

    print ("Field labels: " + str (field_labels_list ))
    print ('Cloud loaded in ' + str(time.time() - start_time) + ' seconds.\nNumber of points: ' + str(points.shape[0] ))

    if (return_separate ):
        return points, field_labels_list
    return NumpyPointCloud (points, field_labels_list )


def conditionalized_load (file_path, return_separate=False ):
    """
    Loads .las and .asc files.

    Input:
        file_path: (String)                 The path to the file, including extension
        return_separate: (boolean)          If True, returns a numpy.ndarray and a list of field labels

    Output:
        NumpyPointCloud, containing:
            points: (np.ndarray)            Point Cloud data. Point coordinates and additional colums (fields)
            field_labels: (list(string))    Labels of the columns of the cloud, describing the type of data
    """

    file_name, file_extension = os.path.splitext(file_path )

    # # load the file
    if (file_extension == '.las'):
        if ("DSM_Cloud" in os.path.basename (file_path ) or "_dim" in os.path.basename (file_path )):
            # Load DIM cloud
            np_pointcloud = load_las_file (file_path, dtype="dim" )
        else:
            # Load ALS cloud
            np_pointcloud = load_las_file (file_path, dtype="als")

    elif (file_extension == '.asc'):
        # load ASCII cloud
        np_pointcloud = load_ascii_file (file_path )

    if (return_separate ):
        return np_pointcloud.points, np_pointcloud.field_labels_list
    return np_pointcloud


def save_obj (object, name ):
    """Saves a python object into data/ folder, using pickle"""

    print ("\n Saving file " + 'data/' + name + '.pkl')
    with open('data/' + name + '.pkl', 'wb') as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)


def load_obj (name ):
    """Loads a python object from the data/ folder, using pickle"""

    with open('data/' + name + '.pkl', 'rb') as file:
        return pickle.load(file )


def join_saved_dictionaries (list_of_dict_names, output_name ):
    """Takes a list of dictionary names that are saved under 'data/' and saved their combined contents under
    'data/output_name'

    Input:
        list_of_dict_names: (list(String))  A list of dictionary names to load, excluding their extensions
        output_name: (String)               The dictionary name to save the results under, excluding extension
    """

    resulting_dictionary = {}

    # simply update the empty dictionary, effectively overwriting differing entries with the later dictionaries
    for dict_name in list_of_dict_names:
        resulting_dictionary.update (load_obj (dict_name ))

    save_obj (resulting_dictionary, output_name )

    return True


def print_reference_dict (reference_dictionary_name ):
    """
    Load a reference dictionary in the data/ folder by name (excluding extension) and print it's innards

    Input:
        reference_dictionary_name: (String)   Name of a dict located in 'data/' of shape {("",""); ((x,y,z), mse)}
    """

    # parse the reference values saved in a file
    reference_dictionary = load_obj (reference_dictionary_name )
    print ("\n" + str (reference_dictionary_name ) + ":" )

    # iterate through the keys (path pairs) of the dictionary
    for path_tuple in sorted(reference_dictionary ):

        # disassemble the key
        reference_path, aligned_path = path_tuple
        results_tuple = reference_dictionary[path_tuple]

        # folder should be the the same
        folder, reference_file_name = get_folder_and_file_name (reference_path)
        folder, aligned_file_name = get_folder_and_file_name (aligned_path)

        # unpack values
        ref_translation, ref_mse = results_tuple
        if (type (ref_mse) is not tuple ):
            ref_mse = (ref_mse, 0, 0)

        # print comparison
        print ("'" + aligned_file_name
               + "' aligned to '" + reference_file_name + "'"
               + ';{: .8f}'.format(ref_translation[0])
               + '\n;{: .8f}'.format(ref_translation[1])
               + '\n;{: .8f}'.format(ref_translation[2])
               + '\n;{: .8f}'.format(ref_mse[0]))


def check_for_file (file_path ):
    """Checks if a file is present"""
    return os.path.isfile(file_path )


def get_all_files_in_subfolders (path_to_folder, permitted_file_extension=None ):
    """Finds all files inside the folders below the given folder (1 level below)"""

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
        onlyfiles = [f for f in listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for file in onlyfiles:
            full_paths.append (full_directories[dir_count] + '/' + file )

    # if specified, remove all file extensions that do not match the specified extension
    paths_to_remove = []
    if (permitted_file_extension is not None ):
        for path in full_paths:
            file_name, file_extension = os.path.splitext(path )
            if (file_extension != permitted_file_extension):
                paths_to_remove.append (path )
    full_paths = [path for path in full_paths if path not in paths_to_remove]

    return full_paths


def get_folder_and_file_name (file_path ):
    """Extracts folder and base name of a cloud"""

    # extract the folder
    cloud_folder = str(file_path.split ('/')[-2])

    # extract the name of the cloud and split it by it's attributes, or tags
    list_of_cloud_tags = os.path.basename(file_path.strip ('/' )).split ('_' )[0:2]

    # add '_' and join the tags
    list_of_cloud_tags = ['{0}_'.format(element) for element in list_of_cloud_tags]
    cloud_name = ''.join(list_of_cloud_tags).strip('_')

    return cloud_folder, cloud_name


def get_matching_filenames(filename ):
    """
    Copyright Florian Politz

    Finds matching filenames from one point cloud to another. Matches ALS16 to DIM16

    Inputs:
        filename: (string) filename of the original file to split

    Outputs:
        [s1, s2, s3, s4]: (list of string)
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
