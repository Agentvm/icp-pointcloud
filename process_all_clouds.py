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

    # compute normals for each point
    for index, point_neighbor_indices in enumerate (list_of_point_indices ):

        # check memory usage
        if (psutil.virtual_memory().percent > 95.0):
            print (print ("!!! Memory Usage too high: "
                          + str(psutil.virtual_memory().percent)
                          + "%. Breaking loop. There still are "
                          + str (len (list_of_point_indices) - index)
                          + " normal vectors left to compute."))
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
    if ('Nx' in field_labels_list ):
        field_labels_list = field_labels_list[:-4]
        numpy_cloud = numpy_cloud[:, :-4]

    # add the newly computed values to the cloud
    numpy_cloud = np.concatenate ((numpy_cloud, additional_values ), axis=1 )
    field_labels_list.append('Nx ' 'Ny ' 'Nz ' 'Sigma' )

    return numpy_cloud, field_labels_list


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


def process_clouds (file_extension, reduce_clouds=False, do_normal_calculation=False, apply_icp_algorithm=False):
    '''
    Loads all .las files in a given folder. At the users choice, this function also reduces their points so they are
    closer to zero, computes normals for all points and then saves them again with a different name, or applies an icp
    algorithm to files in the same folder (on of the files in the folder must have "_reference" in it's name)
    '''

    # crawl path
    path = "clouds/Regions/"
    full_paths = get_all_files_in_subfolders (path, file_extension )
    print ("full_paths: " + str (full_paths[17:] ))

    # # just print paths and quit, if no task was selected
    # if (not reduce_clouds and not do_normal_calculation and not apply_icp_algorithm ):
    #     return True

    # logic check
    if (apply_icp_algorithm and (reduce_clouds or do_normal_calculation )):
        print ("Don't reduce clouds or compute their normals while applying icp.\nPlease separate these steps.")
        return True

    # before start, check if files exist
    for file_path in full_paths:
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # set process variables
    previous_folder = ""    # for folder comparison
    icp_results = {}        # dictionary to hold icp results
    icp_reference_cloud = None

    # process clouds
    for file_path in full_paths:
        print ("\n\n-------------------------------------------------------")

        # # split path and extension
        file_name, file_extension = splitext(file_path )
        # check if it's a .las file, else skip it
        # if (file_extension != ".las" ):
        #     continue

        field_labels_list = ['X', 'Y', 'Z']

        # if ("ALS" in file_path):
        #     continue    # normals only on ALS

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

        # # treat clouds folder-specific
        # find folder name
        if (len(file_path.split ('/')) == 1):
            current_folder = file_path
        else:
            current_folder = file_path.split ('/')[-2]

        # check if the folder changed
        if (current_folder != previous_folder):

            if ("_reduced" not in file_path and reduce_clouds):
                # all clouds in one folder should get the same trafo
                min_x, min_y = get_reduction (numpy_cloud )

            if (apply_icp_algorithm and "_reference" in file_path):
                # apply icp to all clouds in a folder, use the cloud marked "_reference" as reference
                icp_reference_cloud = numpy_cloud

        elif (apply_icp_algorithm ):
            # the folder is the same -> this is the second file in this folder

            # sample DIM Clouds
            if ("DSM_Cloud" in file_path):

                sample_factor = 6

                # deterministic sampling
                #numpy_cloud = numpy_cloud[::sample_factor]

                # random sampling
                indices = random.sample(range(0, numpy_cloud.shape[0] ), int (numpy_cloud.shape[0] / sample_factor ))
                numpy_cloud = numpy_cloud[indices, :]

                print ("DIM Cloud sampled, factor: " + str(sample_factor ))

            icp_results.update (do_icp (icp_reference_cloud, numpy_cloud, file_path ))

        previous_folder = current_folder

        # # # alter cloud
        cloud_altered = False

        # # reduce cloud
        # skip files already processed
        if ("_reduced" not in file_path and reduce_clouds):
            numpy_cloud = apply_reduction (numpy_cloud, min_x, min_y )
            cloud_altered = True

        # # compute normals on cloud
        # skip files already processed
        if ("_normals" in file_path and do_normal_calculation):
            numpy_cloud, field_labels_list = compute_normals (numpy_cloud, file_path, field_labels_list, 2.5 )
            cloud_altered = True

        # save the cloud again
        if (cloud_altered):
            input_output.save_ascii_file (numpy_cloud, field_labels_list, file_path )
            #input_output.save_ascii_file (numpy_cloud, field_labels_list, "clouds/tmp/normals2_test.asc" )

    if (apply_icp_algorithm ):
        compare_icp_results (icp_results )

    print ("\n\nDone.")
    return True


if __name__ == '__main__':
    if (process_clouds ('.asc', do_normal_calculation=True )):
        print ("\n\nAll Clouds successfully processed.")
    else:
        print ("Error. Not all clouds could be processed.")
