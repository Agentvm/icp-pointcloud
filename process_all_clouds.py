from modules import input_output
from modules import normals
from modules import icp
from data import reference_transformations
from os import listdir, walk
from os.path import isfile, join, splitext
from collections import OrderedDict
import sklearn.neighbors    # kdtree
import numpy as np
import random
import psutil


def get_all_files_in_subfolders (path_to_folder, permitted_file_extension=None ):
    '''
    Finds all files inside the folders below the given folder (1 level below)
    '''

    # find all directories below path_to_folder
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


def compute_normals (numpy_cloud, file_path, field_labels_list, query_radius ):
    '''
    Computes Normals for a Cloud and concatenates the newly computed colums with the Cloud.
    '''

    # build a kdtree
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')

    additional_values = np.zeros ((numpy_cloud.shape[0], 4 ))
    success = True

    # compute normals for each point
    for index, point in enumerate (numpy_cloud ):

        # check memory usage
        if (psutil.virtual_memory().percent > 95.0):
            print (print ("!!! Memory Usage too high: "
                          + str(psutil.virtual_memory().percent)
                          + "%. Skipping cloud. There still are "
                          + str (numpy_cloud.shape[0] - index)
                          + " normal vectors left to compute. Reduction process might be lost."))
            success = False
            break

        if (index % int(numpy_cloud.shape[0] / 10) == 0):
            print ("Progress: " + "{:.1f}".format ((index / numpy_cloud.shape[0]) * 100.0 ) + " %" )

        # kdtree radius search
        point_neighbor_indices = tree.query_radius(point.reshape (1, -1), r=query_radius )

        # just get all indices in the point radius
        point_neighbor_indices = [nested_value for value in point_neighbor_indices for nested_value in value]

        # you can't estimate a cloud with less than three neighbors
        if (len (point_neighbor_indices) < 3 ):
            continue

        # do a Principal Component Analysis with the plane points obtained by a RANSAC plane estimation
        normal_vector, sigma, mass_center = normals.PCA (
                    normals.ransac_plane_estimation (numpy_cloud[point_neighbor_indices, :],   # point neighbors
                                                     threshold=0.3,  # max point distance from the plane
                                                     fixed_point=numpy_cloud[index, :],
                                                     w=0.6,         # probability for the point to be an inlier
                                                     z=0.90)        # desired probability that plane is found
                                                     [1] )          # only use the second return value, the points

        # join the normal_vector and sigma value to a 4x1 array and write them to the corresponding position
        additional_values[index, :] = np.append (normal_vector, sigma )

    # delete normals if already computed
    if ('Nx' in field_labels_list
       and 'Ny' in field_labels_list
       and 'Nz' in field_labels_list
       and 'Sigma' in field_labels_list ):
        indices = []
        indices.append (field_labels_list.index('Sigma' ))
        indices.append (field_labels_list.index('Nz' ))
        indices.append (field_labels_list.index('Ny' ))
        indices.append (field_labels_list.index('Nx' ))

        print ("Found previously computed normals. Removing the following fields: Nx, Ny, Nz and Sigma")

        field_labels_list = [label for label in field_labels_list if field_labels_list.index(label) not in indices]
        numpy_cloud = np.delete (numpy_cloud, indices, axis=1 )

    # add the newly computed values to the cloud
    numpy_cloud = np.concatenate ((numpy_cloud, additional_values ), axis=1 )
    field_labels_list.append('Nx ' 'Ny ' 'Nz ' 'Sigma' )

    return numpy_cloud, field_labels_list, success


def sample_cloud (numpy_cloud, sample_factor, deterministic_sampling=False ):
    '''
    Samples a cloud by a given factor.
    '''
    previous_length = numpy_cloud.shape[0]

    # deterministic sampling
    if (deterministic_sampling ):
        numpy_cloud = numpy_cloud[::sample_factor]
    # random sampling
    else:
        indices = random.sample(range(0, numpy_cloud.shape[0] ), int (numpy_cloud.shape[0] / sample_factor ))
        numpy_cloud = numpy_cloud[indices, :]

    print ("DIM Cloud sampled, factor: "
           + str(sample_factor )
           + ". Cloud size / previous cloud size: "
           + str(numpy_cloud.shape[0] )
           + "/"
           + str (previous_length))

    return numpy_cloud


def do_icp (full_path_of_reference_cloud, full_path_of_aligned_cloud ):

    # load reference cloud
    reference_cloud = input_output.load_ascii_file (full_path_of_reference_cloud )

    if ("DSM_Cloud" in full_path_of_reference_cloud):
        reference_cloud = sample_cloud (reference_cloud, 5, deterministic_sampling=False )

    # load aligned clouds
    aligned_cloud = input_output.load_ascii_file (full_path_of_aligned_cloud )

    # sample DIM Clouds
    if ("DSM_Cloud" in full_path_of_aligned_cloud):
        aligned_cloud = sample_cloud (aligned_cloud, 5, deterministic_sampling=False )

    translation, mean_squared_error = icp.icp (reference_cloud, aligned_cloud, verbose=False )

    dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud): (translation, mean_squared_error)}

    return dictionary_line


def get_folder_and_file_name (path ):

    # mash up the string
    folder = str(path.split ('/')[-2])
    list_of_filename_attributes = path.split ('/')[-1].split ('_')[0:3]
    list_of_filename_attributes = ['{0}_'.format(element) for element in list_of_filename_attributes]
    file_name = ''.join(list_of_filename_attributes)

    return folder, file_name


def compare_icp_results (icp_results ):

    reference_dict = reference_transformations.translations

    # # sort the results
    # create a list of tuples from reference and aligned cloud file paths
    unssorted_results = []
    for paths in icp_results:
        unssorted_results.append ((paths, icp_results[paths]) )

    sorted_results = sorted(unssorted_results )

    for paths, translation_value in sorted_results:

        if (paths in reference_dict ):

            # disassemble the key
            reference_path, aligned_path = paths

            folder, reference_file_name = get_folder_and_file_name (reference_path)
            folder, aligned_file_name = get_folder_and_file_name (aligned_path)     # folder should be the the same

            # unpack values
            ref_translation, ref_mse = reference_dict [paths]
            icp_translation, icp_mse = translation_value

            # print comparison
            print ('\n' + folder + "/"
                   + "\nreference cloud:\t" + reference_file_name
                   + "\naligned cloud:\t\t" + aligned_file_name
                   + "\n\tdata alignment:\t" + '({: .8f}, '.format(ref_translation[0])
                                             + '{: .8f}, '.format(ref_translation[1])
                                             + '{: .8f}), '.format(ref_translation[2])
                                             + ' {: .8f}, '.format(ref_mse)
                   + "\n\ticp alignment:\t" + '({: .8f}, '.format(icp_translation[0])
                                            + '{: .8f}, '.format(icp_translation[1])
                                            + '{: .8f}), '.format(icp_translation[2])
                                            + '({: .8f}, '.format(icp_mse[0])
                                            + '{: .8f}, '.format(icp_mse[1])
                                            + '{: .8f}) '.format(icp_mse[2]))


def use_icp_on_dictionary (icp_paths_dictionary ):
    '''
    Uses a dictionary of reference cloud file_paths as keys
    and a list of corresponding aligned cloud file_paths as values
    '''

    # before start, check if files exist
    for key in icp_paths_dictionary:
        if (input_output.check_for_file (key ) is False):
            print ("File " + key + " was not found. Aborting.")
            return False
        for aligned_cloud_path in icp_paths_dictionary[key]:
            if (input_output.check_for_file (aligned_cloud_path ) is False):
                print ("File " + aligned_cloud_path + " was not found. Aborting.")
                return False

    icp_results = {}    # dictionary

    # create a list of tuples from reference and aligned cloud file paths
    for reference_cloud_path in icp_paths_dictionary:
        for aligned_cloud_path in icp_paths_dictionary[reference_cloud_path]:   # multiple aligned clouds possible
            # do the icp
            icp_results.update (do_icp (reference_cloud_path, aligned_cloud_path ))

    # prints the values computed along with the ground truth in the dictionary
    compare_icp_results (icp_results )

    return True


def use_icp_on_folder (path_to_folder,
                       reference_file_tag,
                       aligned_file_tag=None,
                       permitted_file_extension=None ):

    print ("WARNING: use_icp_on_folder() function currently not working. Use use_icp_on_dictionary() instead.")

    # crawl path
    full_paths = get_all_files_in_subfolders (path_to_folder, permitted_file_extension )
    print ("full_paths: " + str (full_paths ))

    # before start, check if files exist
    for file_path in full_paths:
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # filter full paths for reference and aligned paths (leading to the clouds used as reference and aligned in icp)
    full_paths_reference = [path for path in full_paths if (reference_file_tag in path)]
    full_paths_aligned = [path for path in full_paths if (aligned_file_tag in path or aligned_file_tag is None)]

    # check if there are files containing reference_file_tag
    if len(full_paths_reference) == 0:
        print ("No files containing "
               + str (reference_file_tag )
               + " were found. No reference clouds could be selected. Aborting.")
        return False

    # build a dictionary of reference cloud paths as key
    # and a list of corresponding aligned cloud paths as values
    icp_paths_dictionary = {}

    # read all reference cloud paths and accumulate the corresponding align cloud paths in the same folder
    for reference_file_path in full_paths_reference:

        # # treat clouds folder-specific
        # find folder name
        if (len(reference_file_path.split ('/')) == 1):
            current_folder = "no folder"     # no folder
        else:
            current_folder = reference_file_path.split ('/')[-2]

        # read all aligned cloud paths and find the ones in the same folder
        corresponding_align_paths = [path for path in full_paths_aligned
                              if (current_folder in path
                              or len(path.split ('/')) == 1)]

        # write a new entry in the dictionary
        icp_paths_dictionary.update ({reference_file_path: corresponding_align_paths})

    print ("icp_paths_dictionary:\n" + str (icp_paths_dictionary ).replace (',', ',\n'))

    return use_icp_on_dictionary (icp_paths_dictionary )


def get_reduction (numpy_cloud ):
    '''
    Compute the min x and min y coordinate
    '''

    min_x_coordinate = np.min (numpy_cloud[:, 0] )
    min_y_coordinate = np.min (numpy_cloud[:, 1] )

    return min_x_coordinate, min_y_coordinate


def apply_reduction (numpy_cloud, min_x_coordinate, min_y_coordinate ):

    # reduce the cloud, so all points are closer to origin
    numpy_cloud[:, 0] = numpy_cloud[:, 0] - min_x_coordinate
    numpy_cloud[:, 1] = numpy_cloud[:, 1] - min_y_coordinate

    return numpy_cloud


def conditionalized_load (file_path ):
    '''
    Loads .las and .asc files.

    Input:
        file_path (string):     The path to the file to load. Include file extension.

    Output:
        numpy_cloud (np.array): The cloud values, fitted in a numpy nd array
        field_labels_list:      The header of the file, containing the labels of the cloud fields (column titles)
    '''

    field_labels_list = ['X', 'Y', 'Z']
    file_name, file_extension = splitext(file_path )

    # # load the file
    if (file_extension == '.las'):
        if ("DSM_Cloud" in file_path):
            # Load DIM cloud
            numpy_cloud = input_output.load_las_file (file_path, dtype="dim" )
            field_labels_list.append ('Rf ' 'Gf ' 'Bf ' 'Classification')
        else:
            # Load ALS cloud
            numpy_cloud = input_output.load_las_file (file_path, dtype="als")
            field_labels_list.append('Intensity '
                                     'Number_of_Returns '
                                     'Return_Number '
                                     'Point_Source_ID '
                                     'Classification')
    elif (file_extension == '.asc'):
        # load ASCII cloud
        numpy_cloud = input_output.load_ascii_file (file_path )
        with open(file_path) as f:
            field_labels_list = f.readline().strip ('//').split ()

    return numpy_cloud, field_labels_list


def process_clouds_in_folder (path_to_folder,
                              permitted_file_extension=None,
                              string_to_ignore="",
                              reduce_clouds=False,
                              do_normal_calculation=False,
                              normals_computation_radius=2.5 ):
    '''
    Loads all .las files in a given folder. At the users choice, this function also reduces their points so they are
    closer to zero, computes normals for all points and then saves them again with a different name, or applies an icp
    algorithm to files in the same folder (on of the files in the folder must have "_reference" in it's name)
    '''

    # crawl path
    full_paths = get_all_files_in_subfolders (path_to_folder, permitted_file_extension )
    #print ("full_paths: " + str (full_paths ))

    # # just print paths and quit, if no task was selected
    # if (not reduce_clouds and not do_normal_calculation ):
    #     return True

    # before start, check if files exist
    print ("The following files will be processed:\n" )
    for file_path in full_paths:
        #print (str (file_path ))
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # set process variables
    previous_folder = ""    # for folder comparison

    steps = 3
    steps = 11 * steps

    print (full_paths[(-3 - steps):(-steps)])

    # process clouds
    for complete_file_path in full_paths[(-3 - steps):(-steps)]:
        print ("\n\n-------------------------------------------------------")

        # # split path and extension
        #file_name, file_extension = splitext(complete_file_path )

        # skip files containing string_to_ignore
        if (string_to_ignore is not None and string_to_ignore in complete_file_path):
            continue

        # # load
        numpy_cloud, field_labels_list = conditionalized_load (complete_file_path )

        # # treat clouds folder-specific
        # find folder name
        if (len(complete_file_path.split ('/')) == 1):
            current_folder = ""     # no folder
        else:
            current_folder = complete_file_path.split ('/')[-2]

        # check if the folder changed
        if (current_folder != previous_folder and reduce_clouds):

            # all clouds in one folder should get the same trafo
            min_x, min_y = get_reduction (numpy_cloud )

        # # # alter cloud
        cloud_altered = False

        # # reduce cloud
        if (reduce_clouds ):
            numpy_cloud = apply_reduction (numpy_cloud, min_x, min_y )
            cloud_altered = True

        # # compute normals on cloud
        if (do_normal_calculation ):
            numpy_cloud, field_labels_list, success = compute_normals (numpy_cloud,
                                                                       complete_file_path,
                                                                       field_labels_list,
                                                                       normals_computation_radius )

            # don't change the cloud unless all normals have been computed
            cloud_altered = success

        # save the cloud again
        if (cloud_altered):
            input_output.save_ascii_file (numpy_cloud, field_labels_list, complete_file_path )
            #input_output.save_ascii_file (numpy_cloud, field_labels_list, "clouds/tmp/normals_fixpoint_test.asc" )

        # set current to previous folder for folder-specific computations
        previous_folder = current_folder

    print ("\n\nDone.")
    return True


# refactor
def print_files_dict_of_folder (folder, permitted_file_extension=None ):
    full_paths = get_all_files_in_subfolders (folder, permitted_file_extension )

    dict = OrderedDict ()
    for path in full_paths:
        dict.update ({('?reference?', path): ((0.0, 0.0, 0.0), 0.0)} )

    print ("Paths in Folder "
           + folder
           + ':\n\n'
           + str (dict.replace ('\', ', '\',\n').replace (': ', ':\n').replace ('), ', '),\n' )))

    return dict

    # + str (full_paths ).replace (', ', '\n' ).replace ('\'', '' ).strip ('[' ).strip (']' ).replace
    #  ('\n', ':\n((0.0, 0.0, 0.0), 0.0),\n' ))


def get_icp_data_paths ():
    '''
    Reads reference_transformations.translations to get all transformations currently saved and returns them in a
    dictionary that can be directly used with use_icp_on_dictionary()
    '''
    dict = {}
    for key in reference_transformations.translations:

        reference_path, aligned_path = key

        if (dict.__contains__ (reference_path )):
            dict[reference_path].append (aligned_path )
        else:
            dict.update ({reference_path: [aligned_path]} )

    return dict


if (random.seed != 1337):
    random.seed = 1337
    print ("Random Seed set to: " + str(random.seed ))

if __name__ == '__main__':

    if (random.seed != 1337):
        random.seed = 1337
        print ("Random Seed set to: " + str(random.seed ))

    # # normals / reducing clouds
    if (process_clouds_in_folder ('clouds/Regions/',
                                  permitted_file_extension='.asc',
                                  string_to_ignore='Results',
                                  do_normal_calculation=True,
                                  normals_computation_radius=2.5 )):

        print ("\n\nAll Clouds successfully processed.")
    else:
        print ("Error. Not all clouds could be processed.")

    # # icp
    # print ("\n\nComputing ICP for each cloud pair in reference_transformations.translations returns: "
    #        + str(use_icp_on_dictionary (get_icp_data_paths () )))

    # compare_icp_results (do_icp ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    #                              'clouds/Regions/Everything/ALS16_Cloud _Scan54_reduced_normals.asc' ))

    # # tests

    # print_files_dict_of_folder(folder)

    # print (get_icp_data_paths ())

    # if (use_icp_on_folder ('clouds/Regions/',
    #                        reference_file_tag='ALS16',
    #                        aligned_file_tag='DIM_Cloud',
    #                        permitted_file_extension='.asc' )
    #    and use_icp_on_dictionary ({})
    #    and use_icp_on_dictionary ({})
    #    and use_icp_on_dictionary ({}) ):
    #
    #     print ("\n\nAll Clouds successfully processed.")
    # else:
    #     print ("Error. Not all clouds could be processed.")
