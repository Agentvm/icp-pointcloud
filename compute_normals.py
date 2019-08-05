
# local modules
from modules import input_output
from modules import normals

# basic imports
import numpy as np
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


def compute_normals (np_pointcloud, query_radius, batch_size = 10000 ):
    """
    Computes Normals Fields and Sigma Field (noise of normal computation) for each point of a NumpyPointCloud and adds
    the newly computed colums to the Cloud.

    Input:
        np_pointcloud (NumpyPointCloud):    NumpyPointCloud object containing a numpy array and it's data labels
        query_radius (float):               The radius in which to search for neighbors belonging to a plane
        batch_size (integer):               Number of points per kdtree query. Higher numbers mean higer RAM usage.

    Output:
        np_pointcloud (NumpyPointCloud):    The updated NumpyPointCloud
        success (boolean):                  Whether this worked, or rather not
    """

    # build a kdtree
    #tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud[:, 0:3], leaf_size=40, metric='euclidean')
    scipy_kdtree = scipy.spatial.cKDTree (np_pointcloud.get_xyz_coordinates () )

    # prepare variables
    additional_values = np.zeros ((np_pointcloud.points.shape[0], 4 ))  # container for computed normal and sigma values
    cloud_points_number = np_pointcloud.points.shape[0]
    success = True
    errors = 0
    overall_time = time.time ()

    print ("\nStarting normal computation." )
    print ("query_radius = " + str (query_radius ))
    print ("batch_size = " + str (batch_size ))

    # iterate through the cloud in batches of size 'batch_size'
    for i in range (0, cloud_points_number, batch_size ):
        if (i + batch_size > cloud_points_number):
            batch_points = np_pointcloud.points[i:, 0:3]
        else:
            batch_points = np_pointcloud.points[i:i+batch_size, 0:3]

        # check memory usage
        if (psutil.virtual_memory().percent > 95.0):
            print (print ("!!! Memory Usage too high: "
                          + str(psutil.virtual_memory().percent)
                          + "%. Skipping cloud. There still are "
                          + str (cloud_points_number - i)
                          + " normal vectors left to compute. Reduction process might be lost."))
            success = False
            break

        # Show Progress every 'batch_size' points
        print ("Progress: " + "{:.1f}".format ((i / cloud_points_number) * 100.0 ) + " %" )

        # kdtree radius search for each point in this batch
        batch_point_indices = scipy_kdtree.query_ball_point (np.ascontiguousarray(batch_points ), r=query_radius )

        # iterate through the query results of this batch, processing the result of each point individually (slow)
        for t, point_neighbor_indices in enumerate (batch_point_indices, 0 ):
            iterator = i + t    # batch iterator + in-batch iterator

            # you can't estimate a cloud with less than three points
            if (len (point_neighbor_indices) < 3 ):
                errors += 1
                continue

            # start a RANSAC algorithm to find a fitting plane in the detected neighbor points
            normal_vector, points = normals.ransac_plane_estimation (
                                 np_pointcloud.points[point_neighbor_indices, :],   # neighbors
                                 threshold=0.3,                             # max point distance from the plane
                                 fixed_point=np_pointcloud.points[iterator, :],    # this point will be part of the plane
                                 w=0.6,                                     # probability for the point to be an inlier
                                 z=0.90)                                    # desired probability that plane is found

            # do a Principal Component Analysis with the plane points obtained by a RANSAC plane estimation
            # this is an analytic process, hence more precise
            normal_vector, sigma, mass_center = normals.PCA (points )

            # check if normal computation was a success
            if (normal_vector[0] == 0 and normal_vector[1] == 0 and normal_vector[2] ):
                errors += 1

            # join the normal_vector and sigma value to a 4x1 array and write them to the corresponding position
            additional_values[iterator, :] = np.append (normal_vector, sigma )

    # add the newly computed values to the cloud
    np_pointcloud.add_fields (additional_values, ['Nx', 'Ny', 'Nz', 'Sigma'], replace=True )

    print ("Normal computation completed in " + str (time.time () - overall_time ) + " seconds." )
    print (str (cloud_points_number - errors )
           + "/" + str (cloud_points_number )
           + " normal vectors have been computed." )

    return np_pointcloud, success


def clear_redundand_classes (np_pointcloud ):
    """Deletes points with a class label > 19"""

    # keep classes < 20
    classes = np_pointcloud.get_fields (["Classification"])
    np_pointcloud.points = np_pointcloud.points[classes < 20, :]

    return np_pointcloud


def process_clouds_in_folder (path_to_folder,
                              permitted_file_extension=None,
                              string_list_to_ignore="",
                              reduce_clouds=False,
                              do_normal_calculation=False,
                              clear_classes=False,
                              normals_computation_radius=2.5 ):
    """
    Loads all .las files in a given folder. At the users choice, this function also reduces their points so they are
    closer to zero, computes normals for all points, or removes points whose 'Classification' field value is greater
    than 19 and then saves them again with a different name.
    """

    # crawl path
    full_paths = input_output.get_all_files_in_subfolders (path_to_folder, permitted_file_extension )

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
    for complete_file_path in full_paths:

        # skip file paths containing a string from string_list_to_ignore
        if (string_list_to_ignore is not None
           and any(ignore_string in complete_file_path for ignore_string in string_list_to_ignore ) ):
            continue

        print ("\n\n-------------------------------------------------------")

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

        # # alter cloud
        cloud_altered = False

        # delete everything that has a value of more or equal to 20 in "Classification" field
        if (clear_classes ):
            np_pointcloud = clear_redundand_classes (np_pointcloud )
            print ("Points with class 20 and above have been removed from this cloud.\n")
            cloud_altered = True

        # reduce cloud
        if (reduce_clouds ):
            np_pointcloud.points = apply_reduction (np_pointcloud.points, min_x, min_y )
            print ("Cloud has been reduced by x=" + str(min_x ) + ", y=" + str(min_y ) + ".\n")
            cloud_altered = True

        # compute normals on cloud
        if (do_normal_calculation ):
            np_pointcloud, success = compute_normals (np_pointcloud, normals_computation_radius )

            # don't change the cloud unless all normals have been computed
            cloud_altered = success

        # save the cloud again
        if (cloud_altered):
            # add a tag for every action performed
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

    # # normals / reducing clouds / clearing classes
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
