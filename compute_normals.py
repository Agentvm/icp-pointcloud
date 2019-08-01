
# local modules
from modules import input_output
from modules import normals

# basic imports
import numpy as np
import random
import os.path
import time

# advanced functionality
import scipy.spatial
import psutil


def get_reduction (numpy_cloud ):
    """Compute the min x and min y coordinate"""

    min_x_coordinate = np.min (numpy_cloud[:, 0] )
    min_y_coordinate = np.min (numpy_cloud[:, 1] )

    return min_x_coordinate, min_y_coordinate


def apply_reduction (numpy_cloud, min_x_coordinate, min_y_coordinate ):
    """Reduce the cloud's point coordinates, so all points are closer to origin """
    numpy_cloud[:, 0] = numpy_cloud[:, 0] - min_x_coordinate
    numpy_cloud[:, 1] = numpy_cloud[:, 1] - min_y_coordinate

    return numpy_cloud


def compute_normals (numpy_pointcloud, file_path, query_radius ):
    """
    Computes Normals for a Cloud and adds the newly computed colums to the Cloud.

    Input:


    Output:

    """

    # outer_iterations = 0
    # inner_iterations = 0
    outer_loop_query_time = 0
    inner_loop_computation_time = 0
    overall_ransac_time = 0
    overall_random_plane_time = 0
    overall_plane_consensus_time = 0
    overall_pca_time = 0
    start_time = time.time ()
    measure = time.time ()

    # build a kdtree
    #tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud[:, 0:3], leaf_size=40, metric='euclidean')
    scipy_kdtree = scipy.spatial.cKDTree (numpy_pointcloud.get_xyz_coordinates () )

    # prepare variables
    additional_values = np.zeros ((numpy_pointcloud.points.shape[0], 4 ))
    success = True

    init_time = time.time () - measure
    measure = time.time ()

    # set radius
    # if ("DSM_Cloud" in file_path):
    #     query_radius = 0.8

    # compute normals for each point
    #for index, point in enumerate (numpy_pointcloud.get_xyz_coordinates () ):
    errors = 0
    step = 10000
    cloud_points_number = numpy_pointcloud.points.shape[0]
    for i in range (0, cloud_points_number, step ):
        interim = time.time ()
        if (i + step > cloud_points_number):
            batch_points = numpy_pointcloud.points[i:, 0:3]
        else:
            batch_points = numpy_pointcloud.points[i:i+step, 0:3]

        # check memory usage
        if (psutil.virtual_memory().percent > 95.0):
            print (print ("!!! Memory Usage too high: "
                          + str(psutil.virtual_memory().percent)
                          + "%. Skipping cloud. There still are "
                          + str (numpy_pointcloud.points.shape[0] - i)
                          + " normal vectors left to compute. Reduction process might be lost."))
            success = False
            break

        #if (numpy_pointcloud.points.shape[0] > 10 and i % int(numpy_pointcloud.points.shape[0] / 10) == 0):
        print ("Progress: " + "{:.1f}".format ((i / numpy_pointcloud.points.shape[0]) * 100.0 ) + " %" )

        # kdtree radius search
        #point_neighbor_indices = tree.query_radius(point.reshape (1, -1), r=query_radius )
        batch_point_indices = scipy_kdtree.query_ball_point (np.ascontiguousarray(batch_points ), r=query_radius )

        # # just get all indices in the point radius
        # # refactor: This is the problem why batches do not work

        # print ("\n--- --- --- --- --- --step: " + str(batch_point_indices.shape[0] ))

        outer_loop_query_time += time.time () - interim

        for t, point_neighbor_indices in enumerate (batch_point_indices, 0 ):

            iterator = i + t

            # if (t == 0):
            #     print ("T stars at 0")

            interim = time.time ()

            # print ("point_indices: " + str(point_neighbor_indices ))

            # diff = numpy_pointcloud.points[point_neighbor_indices, 0:3] - point
            # print ("max_distance = " + str (np.max (np.sqrt (diff[0]**2 + diff[1]**2 + diff[2]**2 ))))

            # # make kdtree smaller in DSM clouds to avoid too many matches slowing the process down
            # if (len (point_neighbor_indices ) > 500):
            #     indices = random.sample(range(0, len (point_neighbor_indices ) ),
            #                             int (len (point_neighbor_indices ) / 5 ))
            #     point_neighbor_indices = [point_neighbor_indices[i] for i in indices]

            # you can't estimate a cloud with less than three neighbors
            if (len (point_neighbor_indices) < 3 ):
                continue

            # # do a Principal Component Analysis with the plane points obtained by a RANSAC plane estimation
            # normal_vector, sigma, mass_center, _ = normals.PCA (
            #             normals.ransac_plane_estimation_new (numpy_pointcloud.points[point_neighbor_indices, :],   # point neighbors
            #                                              threshold=0.3,  # max point distance from the plane
            #                                              fixed_point=numpy_pointcloud.points[iterator, :],
            #                                              w=0.6,         # probability for the point to be an inlier
            #                                              z=0.90)        # desired probability that plane is found
            #                                              [1] )          # only use the second return value, the points

            _, points, random_plane_time, plane_consensus_time, ransac_time = normals.ransac_plane_estimation_new (
                                             numpy_pointcloud.points[point_neighbor_indices, :],
                                             threshold=0.3,  # max point distance from the plane
                                             fixed_point=numpy_pointcloud.points[i, :],
                                             w=0.6,         # probability for the point to be an inlier
                                             z=0.90)        # desired probability that plane is found

            overall_random_plane_time += random_plane_time
            overall_plane_consensus_time += plane_consensus_time
            overall_ransac_time += ransac_time

            # do a Principal Component Analysis with the plane points obtained by a RANSAC plane estimation
            normal_vector, sigma, mass_center, pca_time = normals.PCA (points )
            overall_pca_time += pca_time

            if (normal_vector[0] == 0 and normal_vector[1] == 0 and normal_vector[2] ):
                errors += 1

            # join the normal_vector and sigma value to a 4x1 array and write them to the corresponding position
            additional_values[iterator, :] = np.append (normal_vector, sigma )

            inner_loop_computation_time += time.time () - interim

        #     inner_iterations += 1
        #
        # outer_iterations += 1

    interim = time.time ()

    # add the newly computed values to the cloud
    numpy_pointcloud.add_fields (additional_values, ['Nx', 'Ny', 'Nz', 'Sigma'], replace=True )

    outro_time = time.time () - interim

    loop_overall_time = time.time () - measure
    overall_time = time.time () - start_time

    print ("\nstep:            " + str (step ))
    print ("Init Time:         " + str (init_time ))
    print ("Overall Loop Time: " + str (loop_overall_time ))
    print ("\tOuter Loop Time: " + "{:2f}%".format ((outer_loop_query_time / loop_overall_time ) * 100 ))
    print ("\tRANSAC all Time:  " + "{:2f}%".format ((overall_ransac_time / loop_overall_time ) * 100 ))
    print ("\t\tRANSAC rndm Time: " + "{:2f}%".format ((overall_random_plane_time / overall_ransac_time ) * 100 ))
    print ("\t\tRANSAC cnss Time: " + "{:2f}%".format ((overall_plane_consensus_time / overall_ransac_time ) * 100 ))
    print ("\tPCA Time:         " + "{:2f}%".format ((overall_pca_time / loop_overall_time ) * 100 ))

    print ("Outro Time:        " + str (outro_time ))
    print ("Overall Time:      " + str (overall_time ))

    return numpy_pointcloud, success


def clear_redundand_classes (numpy_pointcloud ):
    """Clears points with a class label > 19"""

    # keep classes < 20
    classes = numpy_pointcloud.get_fields (["Classification"])
    numpy_pointcloud.points = numpy_pointcloud.points[classes < 20, :]

    return numpy_pointcloud


def process_clouds_in_folder (path_to_folder,
                              permitted_file_extension=None,
                              string_list_to_ignore="",
                              reduce_clouds=False,
                              do_normal_calculation=False,
                              clear_classes=False,
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
        np_pointcloud = input_output.conditionalized_load (complete_file_path )

        # # treat clouds folder-specific
        # find folder name
        if (len(complete_file_path.split ('/')) == 1):
            current_folder = ""     # no folder
        else:
            current_folder = complete_file_path.split ('/')[-2]

        # check if the folder changed
        if (current_folder != previous_folder and reduce_clouds):

            # all clouds in one folder should get the same trafo
            min_x, min_y = get_reduction (np_pointcloud.points )

        # # # alter cloud
        cloud_altered = False

        #delete everything that has more or equal to 20 in the 8th row:
        if (clear_classes ):
            np_pointcloud = clear_redundand_classes (np_pointcloud )
            print ("Points with class 20 and above have been removed from this cloud.\n")
            cloud_altered = True

        # # reduce cloud
        if (reduce_clouds ):
            np_pointcloud.points = apply_reduction (np_pointcloud.points, min_x, min_y )
            print ("Cloud has been reduced by x=" + str(min_x ) + ", y=" + str(min_y ) + ".\n")
            cloud_altered = True

        # # compute normals on cloud
        if (do_normal_calculation ):
            np_pointcloud, success = compute_normals (np_pointcloud,
                                                      complete_file_path,
                                                      normals_computation_radius )
            if (success):
                print ("Normals successfully computed.\n")
            # don't change the cloud unless all normals have been computed
            cloud_altered = success

        # save the cloud again
        if (cloud_altered):
            alteration_string = "_reduced" if reduce_clouds else ""
            alteration_string += ("_normals_r_" + str (normals_computation_radius )) if do_normal_calculation else ""
            alteration_string += "_cleared" if clear_classes else ""
            filename, file_extension = os.path.splitext(complete_file_path )
            input_output.save_ascii_file (np_pointcloud.points,
                                          np_pointcloud.field_labels,
                                          filename + alteration_string + ".asc" )

        # set current to previous folder for folder-specific computations
        previous_folder = current_folder

    print ("\n\nDone.")
    return True


if __name__ == '__main__':

    # set the random seed for both the numpy and random module
    random.seed (1337 )
    np.random.seed (1337 )

    # # normals / reducing clouds
    if (process_clouds_in_folder ('clouds/tmp/',
                                  permitted_file_extension='.asc',
                                  string_list_to_ignore=['original_clouds', '_r_', 'fail', '.las'],
                                  do_normal_calculation=True,
                                  reduce_clouds=False,
                                  clear_classes=False,
                                  normals_computation_radius=1 )):
        print ("\n\nAll Clouds successfully processed.")
    else:
        print ("Error. Not all clouds could be processed.")
