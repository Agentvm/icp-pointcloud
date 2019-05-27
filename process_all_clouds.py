from modules import input_output
from modules import normals
from modules import icp
from data import reference_transformations
from os import listdir, walk
from os.path import isfile, join, splitext
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
    Computes Normals for a Cloud an concatenates the newly computed colums with the Cloud.
    '''

    # build a kdtree
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')

    # set radius for neighbor search
    if ("DSM_Cloud" in file_path):  # DIM clouds are roughly 6 times more dense than ALS clouds
        #query_radius = 0.8  # m
        query_radius = 1.5  # m

    # kdtree radius search
    list_of_point_indices = tree.query_radius(numpy_cloud, r=query_radius )
    additional_values = np.zeros ((numpy_cloud.shape[0], 4 ))

    success = True

    # compute normals for each point
    for index, point_neighbor_indices in enumerate (list_of_point_indices ):

        # check memory usage
        if (psutil.virtual_memory().percent > 95.0):
            print (print ("!!! Memory Usage too high: "
                          + str(psutil.virtual_memory().percent)
                          + "%. Skipping cloud. There still are "
                          + str (len (list_of_point_indices) - index)
                          + " normal vectors left to compute. Reduction process might be lost."))
            success = False
            break

        # you can't estimate a cloud with less than three neighbors
        if (len (point_neighbor_indices) < 3 ):
            continue

        # do a Principal Component Analysis with the plane points obtained by a RANSAC plane estimation
        normal_vector, sigma, mass_center = normals.PCA (
                    normals.ransac_plane_estimation (numpy_cloud[point_neighbor_indices, :],   # point neighbors
                                                     threshold=0.3,  # max point distance from the plane
                                                     w=0.6,         # probability for the point to be an inlier
                                                     z=0.90)        # desired probability that plane is found
                                                     [1] )          # only use the second return value, the points

        # join the normal_vector and sigma value to a 4x1 array and write them to the corresponding position
        additional_values[index, :] = np.append (normal_vector, sigma )

    # delete normals if already computed # refactor
    if ('Nx' in field_labels_list and success ):
        field_labels_list = field_labels_list[:-4]
        numpy_cloud = numpy_cloud[:, :-4]

    # add the newly computed values to the cloud
    numpy_cloud = np.concatenate ((numpy_cloud, additional_values ), axis=1 )
    field_labels_list.append('Nx ' 'Ny ' 'Nz ' 'Sigma' )

    return numpy_cloud, field_labels_list, success


def do_icp (numpy_reference_cloud, numpy_aligned_cloud, full_path ):

    translation, mean_squared_error = icp.icp (numpy_reference_cloud, numpy_aligned_cloud, verbose=True )
    dictionary_line = {full_path: (translation, mean_squared_error)}

    return dictionary_line


def compare_icp_results (icp_results ):

    reference_dict = reference_transformations.translations

    for key in icp_results:

        # find the computed results in the reference data
        if (key in reference_dict ):

            # mash up the string
            folder = str(key.split ('/')[-2])
            list_of_filename_attributes = key.split ('/')[-1].split ('_')[0:3]
            list_of_filename_attributes = ['{0}_'.format(element) for element in list_of_filename_attributes]

            # unpack values
            ref_translation, ref_mse = reference_dict [key]
            icp_translation, icp_mse = icp_results [key]

            # print comparison
            print ('\n' + folder + "/"
                   + ''.join(list_of_filename_attributes)
                   + "\n\treference:\t" + '({: .8f}, '.format(ref_translation[0])
                                        + '{: .8f}, '.format(ref_translation[1])
                                        + '{: .8f}), '.format(ref_translation[2])
                                        + ' {: .8f}, '.format(ref_mse)
                   + "\n\ticp result:\t" + '({: .8f}, '.format(icp_translation[0])
                                         + '{: .8f}, '.format(icp_translation[1])
                                         + '{: .8f}), '.format(icp_translation[2])
                                         + '({: .8f}, '.format(icp_mse[0])
                                         + '{: .8f}, '.format(icp_mse[1])
                                         + '{: .8f}) '.format(icp_mse[2]))


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


def use_icp_on_folder (path_to_folder,
                       reference_file_tag="reference",
                       aligned_file_tag=None,
                       permitted_file_extension=None ):

    # # variables
    # icp_results = {}        # dictionary to hold icp results
    #
    # # crawl path
    # full_paths = get_all_files_in_subfolders (path_to_folder, permitted_file_extension )
    # print ("full_paths: " + str (full_paths ))
    #
    # # before start, check if files exist
    # for file_path in full_paths:
    #     if (input_output.check_for_file (file_path ) is False ):
    #         print ("File " + file_path + " was not found. Aborting.")
    #         return False
    #
    # # only use files containing string_to_validate
    # if (reference_file_tag not in reference_file_path):
    #     continue
    #
    # # # treat clouds folder-specific
    # # find folder name
    # if (len(complete_file_path.split ('/')) == 1):
    #     current_folder = ""     # no folder
    # else:
    #     current_folder = complete_file_path.split ('/')[-2]
    #
    # # check if the folder changed
    # if (current_folder != previous_folder and reduce_clouds):
    #
    #     # all clouds in one folder should get the same trafo
    #     if ("_reference" in file_path): ?
    #         # apply icp to all clouds in a folder, use the cloud marked "_reference" as reference
    #         icp_reference_cloud = numpy_cloud
    #
    #     else:
    #         # the folder is the same -> this is the second file in this folder
    #
    #         # sample DIM Clouds
    #         if ("DSM_Cloud" in file_path):
    #             aligned_cloud = sample_cloud (aligned_cloud, sample_factor )
    #
    #         icp_results.update (do_icp (icp_reference_cloud, numpy_cloud, file_path ))
    #
    # numpy_cloud = input_output.load_ascii_file (file_path )

    return 1


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
                              do_normal_calculation=False ):
    '''
    Loads all .las files in a given folder. At the users choice, this function also reduces their points so they are
    closer to zero, computes normals for all points and then saves them again with a different name, or applies an icp
    algorithm to files in the same folder (on of the files in the folder must have "_reference" in it's name)
    '''

    # crawl path
    full_paths = get_all_files_in_subfolders (path_to_folder, permitted_file_extension )
    print ("full_paths: " + str (full_paths ))

    # # just print paths and quit, if no task was selected
    # if (not reduce_clouds and not do_normal_calculation ):
    #     return True

    # before start, check if files exist
    for file_path in full_paths:
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # set process variables
    previous_folder = ""    # for folder comparison

    # process clouds
    for complete_file_path in full_paths:
        print ("\n\n-------------------------------------------------------")

        # # split path and extension
        #file_name, file_extension = splitext(complete_file_path )

        # skip files containing string_to_ignore
        if (string_to_ignore in complete_file_path):
            continue

        # # load
        numpy_cloud, field_labels_list = conditionalized_load ()

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
                                                                       2.5 )

            # don't change the cloud unless all normals have been computed
            cloud_altered = success

        # save the cloud again
        if (cloud_altered):
            input_output.save_ascii_file (numpy_cloud, field_labels_list, complete_file_path )
            #input_output.save_ascii_file (numpy_cloud, field_labels_list, "clouds/tmp/normals2_test.asc" )

        # set current to previous folder for folder-specific computations
        previous_folder = current_folder

    print ("\n\nDone.")
    return True


if __name__ == '__main__':
    if (process_clouds_in_folder ('clouds/Regions/',
                                  permitted_file_extension='.asc',
                                  string_to_ignore='ALS',
                                  do_normal_calculation=True )):

        print ("\n\nAll Clouds successfully processed.")
    else:
        print ("Error. Not all clouds could be processed.")
