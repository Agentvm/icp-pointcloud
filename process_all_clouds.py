import input_output
from conversions import reduce_cloud
from os import listdir, walk
from os.path import isfile, join, splitext
import sklearn.neighbors    # kdtree
import normals
import numpy as np
import psutil


def get_all_files_in_subfolders (path_to_folder, permitted_file_extension=None ):
    '''
    Finds all files inside the folders below the given folder (1 level below)
    '''

    # find all directories below path_to_folder
    f = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        f.extend(filenames)
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
    if (permitted_file_extension is not None ):
        for path in full_paths:
            filename, file_extension = splitext(path )
            if (file_extension != permitted_file_extension):
                full_paths.remove (path )

    return full_paths


def process_clouds ():
    '''
    Loads all .las files in a given folder, reduces their points so they are closer to zero, cumputes normals for all
    points, then saves them again with a different name
    '''

    # crawl path
    path = "clouds/Regions/"
    full_paths = get_all_files_in_subfolders (path, ".las" )
    print ("full_paths: " + str (full_paths ))

    # check if files exist
    for file_path in full_paths:
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # process las clouds
    previous_folder = ""    # for file folder comparison
    for file_path in full_paths:
        print ("\n\n-------------------------------------------------------")

        # check if it's a .las file, else skip it
        filename, file_extension = splitext(file_path )
        if (file_extension != ".las" ):
            continue

        if ("_reduced_normals" in file_path):
            continue

        field_labels_list = ['X', 'Y', 'Z']

        # # load the file
        if ("DSM_Cloud" in file_path):
            # Load DIM cloud
            numpy_cloud = input_output.load_las_file (file_path, dtype="dim" )
            numpy_cloud[:, 3:6] = numpy_cloud[:, 3:6] / 65535.0  # rgb short int to float
            field_labels_list.append ('Rf ' 'Gf ' 'Bf ' 'Classification ')
        else:
            continue    # only processing DIM clouds
            # Load ALS cloud
            numpy_cloud = input_output.load_las_file (file_path, dtype="als")
            field_labels_list.append('Intensity '
                                     'Number_of_Returns '
                                     'Return_Number '
                                     'Point_Source_ID '
                                     'Classification ')

        # # reduce the cloud, so all points are closer to origin
        # all clouds in one folder should get the same trafo
        if (len(file_path.split ('/')) == 1):
            current_folder = file_path
        else:
            current_folder = file_path.split ('/')[-2]
        if (current_folder != previous_folder):
            min_x_coordinate, min_y_coordinate = reduce_cloud (numpy_cloud, return_transformation=True )[1:]
        previous_folder = current_folder

        # reduce
        numpy_cloud[:, 0] = numpy_cloud[:, 0] - min_x_coordinate
        numpy_cloud[:, 1] = numpy_cloud[:, 1] - min_y_coordinate

        # # compute normals
        # build a kdtree
        tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')

        # set radius for neighbor search
        query_radius = 5.0  # m
        if ("DSM_Cloud" in file_path):  # DIM clouds are roughly 6 times more dense than ALS clouds
            query_radius = query_radius / 6

        # kdtree radius search
        list_of_point_indices = tree.query_radius(numpy_cloud, r=query_radius )     # this floods memory
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
            additional_values[index, :] = np.append (normal_vector, sigma)

        # add the newly computed values to the cloud
        numpy_cloud = np.concatenate ((numpy_cloud, additional_values), axis=1)
        field_labels_list.append('Nx ' 'Ny ' 'Nz ' 'Sigma ' )

        # save the cloud again
        input_output.save_ascii_file (numpy_cloud, field_labels_list, filename + "_reduced_normals.asc" )

    print ("Done.")
    return True


if __name__ == '__main__':
    if (process_clouds () ):
        print ("\n\nAll Clouds successfully processed.")
    else:
        print ("Error. Not all clouds could be processed.")
