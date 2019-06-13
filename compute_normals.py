from modules import input_output
from modules import normals
import sklearn.neighbors    # kdtree
import numpy as np
import random
import psutil


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


def compute_normals (numpy_cloud, file_path, field_labels_list, query_radius ):
    '''
    Computes Normals for a Cloud and concatenates the newly computed colums with the Cloud.
    '''

    # build a kdtree
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud[:, 0:3], leaf_size=40, metric='euclidean')

    additional_values = np.zeros ((numpy_cloud.shape[0], 4 ))
    success = True

    # compute normals for each point
    for index, point in enumerate (numpy_cloud[:, 0:3] ):

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

        # make kdtree smaller in DSM clouds to avoid too many matches slowing the process down
        if (len (point_neighbor_indices ) > 500):
            indices = random.sample(range(0, len (point_neighbor_indices ) ), int (len (point_neighbor_indices ) / 5 ))
            point_neighbor_indices = [point_neighbor_indices[i] for i in indices]

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
    field_labels_list = field_labels_list + ['Nx ', 'Ny ', 'Nz ', 'Sigma']

    return numpy_cloud, field_labels_list, success


def process_clouds_in_folder (path_to_folder,
                              permitted_file_extension=None,
                              string_list_to_ignore="",
                              reduce_clouds=False,
                              do_normal_calculation=False,
                              normals_computation_radius=2.5 ):
    '''
    Loads all .las files in a given folder. At the users choice, this function also reduces their points so they are
    closer to zero, computes normals for all points and then saves them again with a different name, or applies an icp
    algorithm to files in the same folder (on of the files in the folder must have "_reference" in it's name)
    '''

    # crawl path
    full_paths = input_output.get_all_files_in_subfolders (path_to_folder, permitted_file_extension )
    #print ("full_paths: " + str (full_paths ))

    # # just print paths and quit, if no task was selected
    # if (not reduce_clouds and not do_normal_calculation ):
    #     return True

    # before start, check if files exist
    print ("The following files will be processed:\n" )
    for file_path in full_paths:

        if (string_list_to_ignore is None
           or not any(ignore_string in file_path for ignore_string in string_list_to_ignore ) ):
            print (str (file_path ))
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # set process variables
    previous_folder = ""    # for folder comparison

    # process clouds
    #for complete_file_path in full_paths[(-6 - steps):(-steps)]:
    for complete_file_path in full_paths:
        print ("\n\n-------------------------------------------------------")

        # # split path and extension
        #file_name, file_extension = splitext(complete_file_path )

        # skip files containing string_list_to_ignore
        if (string_list_to_ignore is not None
           and any(ignore_string in complete_file_path for ignore_string in string_list_to_ignore ) ):
            continue

        # # load
        numpy_cloud, field_labels_list = input_output.conditionalized_load (complete_file_path )

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


if __name__ == '__main__':

    if (random.seed != 1337):
        random.seed = 1337
        print ("Random Seed set to: " + str(random.seed ))

    # # normals / reducing clouds
    if (process_clouds_in_folder ('clouds/Regions/Test Xy/',
                                  permitted_file_extension='.asc',
                                  string_list_to_ignore=['original_clouds'],
                                  do_normal_calculation=True,
                                  normals_computation_radius=2.5 )):
        print ("\n\nAll Clouds successfully processed.")
    else:
        print ("Error. Not all clouds could be processed.")
