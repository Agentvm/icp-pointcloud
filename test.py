import numpy as np
import random


# set random seeds
np.random.seed (1337 )
random.seed (1337 )

# prepare cloud
numpy_cloud = np.array([[1.1, 2.1, 3.1],
                        [1.2, 2.2, 3.2],
                        [1.3, 2.3, 3.3],
                        [1.4, 2.4, 3.4],
                        [1.5, 2.5, 3.5],
                        [1.6, 2.6, 3.6]] )

#

# # speed test accumulator init
# import math
# import time
#
#
# def create_closed_grid (grid_length, step ):
#
#     # grid variables
#     steps_number = math.ceil (grid_length / step + 1 )
#     grid_points_number = steps_number**3
#
#     # make a grid in the style of a pointcloud
#     grid = np.zeros ((grid_points_number, 4 ))
#
#     # in intervals of step, create grid nodes
#     general_iterator = 0
#     minimum = -math.floor (steps_number / 2)
#     maximum = math.ceil (steps_number / 2 )
#     for x_iterator in range (minimum, maximum ):
#         for y_iterator in range (minimum, maximum ):
#             for z_iterator in range (minimum, maximum ):
#
#                 grid[general_iterator, 0:3] = [x_iterator * step,
#                                                y_iterator * step,
#                                                z_iterator * step]
#
#                 general_iterator += 1
#
#     return grid
#
#
# measure = time.time ()
# grid = create_closed_grid (2, 0.1 )
# grid_time = time.time - measure
#
# print ("grid.shape: " + str (grid.shape ))
# # print ("grid:\n" + str (grid ))
#
# measure = time.time ()
# lmax = 1
# lmin = -lmax
# step = 0.1
# xyz = np.transpose(np.reshape(np.mgrid[lmin:lmax+step:step, lmin:lmax+step:step, lmin:lmax+step:step], (-1, 4)))
# print ("\nxyz.shape: " + str (grid.shape ))
# # print ("xyz:\n" + str (grid ))
# xyz = time.time - measure


# # Test np_pointcloud class NumpyPointCloud - This will throw warnings (and expected errors, if # are removed )
# from modules.np_pointcloud import NumpyPointCloud
#
# numpy_cloud = np.concatenate ((numpy_cloud, numpy_cloud), axis=1 )
# my_cloud = NumpyPointCloud (numpy_cloud, ["X", "Y", "Z", "I", "Don't", "Know"] )

# #
# # print ("Test: Wrong Get." + str (my_cloud.get_fields (["X", "Y", "Z", "I", "Dont", "Know"] )))
# # print ("Test: Wrong Get." + str (my_cloud.get_fields (["Dont"] )))
# print ("\nTest: Get." + str (my_cloud.get_fields (["X", "Y", "Z", "I", "Don't", "Know"] )))
#
# # wrong replace
# # my_cloud.add_fields (np.array([1.1, 2.1, 3.1, 1.1, 2.1, 3.1]).reshape (-1, 1), "Know" )
# # my_cloud.add_fields (numpy_cloud[:, 2], "Know" )
# print ("\nTest: Wrong Replace." + str (my_cloud ))
#
# print (my_cloud.shape)
# # replace
# my_cloud.add_fields (np.array ([1.1, 2.1, 1337, 1.1, 2.1, 3.1]).reshape(-1, 1), "aaa", replace=True )
# my_cloud.add_fields (numpy_cloud[:, 1], "I", replace=True )
# print ("\nTest: Replace." + str (my_cloud ))
#
# print (my_cloud.shape)
#
# # add
# my_cloud.add_fields ([1.1, 2.1, 3.1, 1337, 2.1, 3.1], "Test1" )
# print ("\nTest: Add." + str (my_cloud ))
#
# my_cloud.delete_fields (["I", "Dont", "Know"] )
# print ("\nTest: Wrong Delete." + str (my_cloud ))
#
# my_cloud.delete_fields (["Test1"] )
# print ("\nTest: Delete." + str (my_cloud ))


# # basic accumulator
# import math
# import scipy.spatial
# from modules import input_output
#
#
# def create_closed_grid (grid_length, step ):
#
#     # grid variables
#     steps_number = math.ceil (grid_length / step + 1 )
#     grid_points_number = steps_number**3
#
#     # make a grid in the style of a pointcloud
#     grid = np.zeros ((grid_points_number, 4 ))
#
#     # in intervals of step, create grid nodes
#     general_iterator = 0
#     min = -math.floor (steps_number / 2)
#     max = math.ceil (steps_number / 2 )
#     for x_iterator in range (min, max ):
#         for y_iterator in range (min, max ):
#             for z_iterator in range (min, max ):
#
#                 grid[general_iterator, 0:3] = [x_iterator * step,
#                                                y_iterator * step,
#                                                z_iterator * step]
#
#                 general_iterator += 1
#
#     return grid
#
#
# numpy_cloud = np.array([[1, 0, 0],
#                         [1, 0, 0],
#                         [20, 0, 0],
#                         [30, 0, 0],
#                         [40, 0, 0],
#                         [50, 0, 0]], dtype=float )
#
# numpy_cloud += np.random.uniform (-0.1, 0.1, size=(numpy_cloud.shape[0], 3 ))
#
# corresponding_cloud = np.array([[1, 2, 0],
#                            [10, 2, 0],
#                            [20, 2, 0],
#                            [30, 2, 0],
#                            [40, 2, 0],
#                            [50, 0, 2]], dtype=float )
#
# corresponding_cloud += np.random.uniform (-0.1, 0.1, size=(corresponding_cloud.shape[0], 3 ))
#
#
# accumulator_radius = 2
# grid_size = 0.1
#
# # build a grid as a kdtree to discretize the results
# consensus_cube = create_closed_grid (accumulator_radius * 2, grid_size )
# grid_kdtree = scipy.spatial.cKDTree (consensus_cube[:, 0:3] )
# print ("\nconsensus_cube shape: " + str (consensus_cube.shape ))
# #print ("\nconsensus_cube:\n" + str (consensus_cube ))
#
# # build kdtree and query it for points within radius
# scipy_kdtree = scipy.spatial.cKDTree (numpy_cloud[:, 0:3] )
# cloud_indices = scipy_kdtree.query_ball_point (corresponding_cloud[:, 0:3], accumulator_radius )
# #print ("\ncloud_indices: " + str (cloud_indices ))
#
# for i, point_indices in enumerate (cloud_indices ):
#     if (len(point_indices ) > 0):
#
#         # diff all points found near the corresponding point with corresponding point
#         diff_vectors = numpy_cloud[point_indices, 0:3] - corresponding_cloud[i, 0:3]
#         print ("\n-------------------------------------------------------\n\npoint_indices:\n" + str (point_indices ))
#         print ("diff_vectors:\n" + str (diff_vectors ))
#
#         # rasterize
#         dists, point_matches = grid_kdtree.query (diff_vectors, k=1 )
#         print ("dists from gridpoints: " + str (dists.T ))
#         print ("grid point matches: " + str (point_matches.T ))
#
#         # update the cube with the results of this point, ignore multiple hits
#         consensus_cube[np.unique (point_matches ), 3] += 1
#         print ("\nupdated consensus_cube >0:\n" + str (consensus_cube[consensus_cube[:, 3] > 0, :] ))
#
#
#
# best_alignment = consensus_cube[np.argmax (consensus_cube[:, 3] ), 0:3]
# print ("\nbest_alignment: \t" + str (best_alignment ))
# #print ("random_offset: \t\t" + str (random_offset ))


# # Plot an angle histogram of the differences of normal vectors
# from modules import input_output
# from modules.normals import normalize_vector_array, normalize_vector
# import matplotlib.pyplot as plt
# import scipy.spatial
#
#
# def load_example_cloud (folder ):
#
#     # # big cloud
#     numpy_pointcloud = input_output.conditionalized_load(
#         'clouds/Regions/' + folder + '/ALS16_Cloud_reduced_normals_cleared.asc' )
#
#     corresponding_pointcloud = input_output.conditionalized_load (
#         'clouds/Regions/' + folder + '/DSM_Cloud_reduced_normals.asc' )
#
#     return numpy_pointcloud, corresponding_pointcloud
#
#
# def einsum_angle_between (vector_array_1, vector_array_2 ):
#
#     # diagonal of dot product
#     diag = np.clip (np.einsum('ij,ij->i', vector_array_1, vector_array_2 ), -1, 1 )
#
#     return np.arccos (diag )
#
#
# def plot_histogram (data, numer_of_bins, maximum ):
#     # the histogram of the data
#     n, bins, patches = plt.hist(data, numer_of_bins, density=False, range=(0, 180), facecolor='g', alpha=0.75 )
#
#     plt.xlabel('angle' )
#     plt.ylabel('count' )
#     plt.title('Histogram of Angle Differences Yz Houses translated' )
#     #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#     plt.axis([0, numer_of_bins, 0, maximum] )
#     plt.grid(True )
#     plt.show()
#
#
# def get_points_normals_zero (numpy_pointcloud, field_labels_list):
#     normals = numpy_pointcloud.get_normals ()
#     normals = np.absolute (normals )
#
#     sqrt = np.sqrt (normals[:, 0]**2 + normals[:, 1]**2 + normals[:, 2]**2 )
#
#     a = np.where (sqrt > 0.5, True, False )
#
#     return numpy_pointcloud.points[a, :]
#
#
# # load clouds
# # numpy_pointcloud, corresponding_pointcloud = load_example_cloud ("Yz Houses" )
#
# numpy_pointcloud = input_output.load_ascii_file ("clouds/tmp/fail/normals_fixpoint_2.asc" )
# corresponding_pointcloud = input_output.load_ascii_file ("clouds/tmp/fail/normals_fixpoint_3.asc" )
#
# # numpy_pointcloud.points = get_points_normals_zero (numpy_pointcloud.points, numpy_cloud_field_labels )
# # corresponding_pointcloud.points = get_points_normals_zero (
# #                                       corresponding_pointcloud.points, corresponding_cloud_field_labels )
#
# # translate
# corresponding_pointcloud.points[:, 0:3] += (0.314620971680, -0.019294738770, -0.035737037659 )
#
# # extract normals
# normals_numpy_cloud = numpy_pointcloud.get_normals ()
# normals_corresponding_cloud = corresponding_pointcloud.get_normals ()
#
# # normalize
# normals_numpy_cloud = normalize_vector_array (normals_numpy_cloud )
# normals_corresponding_cloud = normalize_vector_array (normals_corresponding_cloud )
#
# # build a kdtree and query it
# kdtree = scipy.spatial.cKDTree (numpy_pointcloud.points[:, 0:3] )
# distances, correspondences = kdtree.query (corresponding_pointcloud.points[:, 0:3], k=1 )
#
# # get the angle differences between the normal vectors
# angle_differences = einsum_angle_between (normals_numpy_cloud[correspondences, :],
#                                           normals_corresponding_cloud ) * (180/np.pi)
#
# # plot
# plot_histogram (angle_differences, 180, 12000 )
#
# # corresponding_cloud = np.concatenate (
# #     (corresponding_cloud, angle_differences.reshape (-1, 1 )), axis=1 )
# # input_output.save_ascii_file (corresponding_cloud,
# #                               corresponding_cloud_field_labels + ["AngleDifferences"],
# #                               "clouds/tmp/yz_houses_dim_angles.asc" )


# # einsum behavior
# numpy_cloud = np.array([[1, 0, 0],
#                         [1, 0, 0],
#                         [1, 0, 0]] )
#
# numpy_cloud_2 = np.array([[0, 0, 0],
#                           [1, 1, 0],
#                           [0, 1, 0]] )
#
# numpy_cloud_2 = normalize_vector_array (numpy_cloud_2 )
#
# print (einsum_angle_between (numpy_cloud, numpy_cloud_2 ))


# # einsum test
# numpy_cloud = np.array([[1, 0, 0],
#                         [1, 0, 0],
#                         [1, 0, 0],
#                         [1, 0, 0]] )
# numpy_cloud_2 = np.array([[1, 0, 0],
#                           [0, 13, 0],
#                           [12, 1, 0],
#                           [0, 1, 1]] )
#
# print (np.dot (numpy_cloud, numpy_cloud_2.T ))
# print (np.einsum('ij,ij->i', numpy_cloud, numpy_cloud_2 ))  # dot product with each row in n and p


# # delete isolated points without neighbors in corresponding cloud
# from modules import conversions
# from modules import icp
# from modules import input_output
#
#
# #load ALS and DSM cloud
# als14_cloud, als14_field_labels = input_output.conditionalized_load (
#     'clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc' )
# dim_cloud, dim_field_labels = input_output.conditionalized_load (
#     'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc' )
#
# radius = 0.5
# als14_cloud, als14_field_labels, dim_cloud, dim_field_labels = conversions.mask_cloudpoints_without_correspondence (
#     als14_cloud, als14_field_labels, dim_cloud, dim_field_labels, radius )
#
# print (icp.icp (als14_cloud, dim_cloud ))
#
# # input_output.save_ascii_file (als14_cloud, als14_field_labels, "clouds/tmp/als14_cloud_" + str(radius ) + ".asc")
# # input_output.save_ascii_file (dim_cloud, dim_field_labels, "clouds/tmp/dim_cloud_" + str(radius ) + ".asc")


# # join saved dictionaries
# from modules import input_output
#
#
# input_output.join_saved_dictionaries (["output_dict_1", "output_dict_2", "output_dict_3"], "output_dict")
# print (str (input_output.load_obj ("output_dict" )).replace (")), ", ")),\n" ))


# # get fields test
# def get_fields (numpy_cloud, field_labels_list, requested_fields ):
#
#     # remove any spaces around the labels
#     field_labels_list = [label.strip () for label in field_labels_list]
#
#     if (requested_fields is not None
#        and all(field in field_labels_list for field in requested_fields ) ):
#         indices = []
#         for field in requested_fields:
#             indices.append (field_labels_list.index(field ))
#     else:
#         raise ValueError ("This Cloud is missing one of the requested fields: "
#                           + str(requested_fields )
#                           + ".\nSupplied Cloud fields are: "
#                           + str(field_labels_list ))
#
#     return numpy_cloud[:, indices]
#
#
# numpy_cloud = np.concatenate ((numpy_cloud, numpy_cloud), axis=1 )
# field_labels_list = ["X", "Y", "Z", "A1", "A2", "A3"]
# requested_fields = ["Z", "A1", "A3"]
#
# print (get_fields (numpy_cloud, field_labels_list, requested_fields ))


# # parallelism test
# import math
# import time
# from multiprocessing import Pool
#
#
# def point_distance_cloud_consensus_parallel_wrapper (input):
#     # translation is received as additional argument
#     (tree_of_numpy_cloud, numpy_cloud, corresponding_cloud, translation, distance_threshold ) = input
#
#     # consensus is started with translated corresponding_cloud
#     (consensus_count, consensus_vector, consensus_time) = point_distance_cloud_consensus (
#         tree_of_numpy_cloud, numpy_cloud, corresponding_cloud+translation, distance_threshold )
#
#     # translation is returned alongside the computed values
#     return (consensus_count, consensus_vector, consensus_time, translation)
#
#
# in loop:
# if (algorithmus == 'distance'):
#
#     arguments_list.append (
#         [scipy_kdtree, numpy_cloud, corresponding_cloud, translation, distance_threshold] )
#
#
# out of loop:
# # go parallel
# with Pool(processes=None) as p:
#     (results_list) = p.map (point_distance_cloud_consensus_parallel_wrapper, arguments_list )
# for (consensus_count, consensus_vector, consensus_time, translation) in results_list:
#
#
# def compute (translation ):
#
#     #print (translation )
#
#     count = 3
#
#     for i in range (5000):
#         isprime = True
#
#         for x in range(2, int(math.sqrt(count ) + 1 )):
#             if count % x == 0:
#                 isprime = False
#                 break
#
#         # if isprime:
#         #     print (count )
#
#         count += 1
#
#     return translation, count, 5
#
#
# step = 0.15
# steps_number = 5
# min = -math.floor (steps_number / 2)
# max = math.ceil (steps_number / 2 )
#
#
# measure = time.time ()
# for x_iterator in range (min, max ):
#     for y_iterator in range (min, max ):
#         for z_iterator in range (min, max ):
#
#             translation = [x_iterator * step,
#                            y_iterator * step,
#                            z_iterator * step]
#
#             compute (translation )
#
# print ("Plain Time: " + str (time.time () - measure ))
#
# measure_whole = time.time ()
# to_do_list = []
# for x_iterator in range (min, max ):
#     for y_iterator in range (min, max ):
#         for z_iterator in range (min, max ):
#
#             translation = [x_iterator * step,
#                            y_iterator * step,
#                            z_iterator * step]
#
#             to_do_list.append (translation )
#
# # go parallel
# with Pool(processes=None) as p:
#     results_list = p.map (compute, to_do_list )
#
# # for (consensus_count, consensus_vector, consensus_time) in results_list:
# #     print (consensus_count )
# #     print (consensus_vector )
# #     print (consensus_time )
#
# print ("Parallel Complete Time: " + str (time.time () - measure_whole ))


# # save current transformations.reference_translations as dict
# from data import transformations
# from modules import input_output
#
#
#
# input_output.save_obj (transformations.reference_translations, "reference_translations_part_3_dict")


# # test data dictionaries
# from queue_alignment_algorithms import get_reference_data_paths, compare_results
# from modules import input_output
# from data import transformations
#
#
# def an_algorithmus (ref_cloud_path, aligned_cloud_path, plot_title ):
#
#     dictionary_line = {(ref_cloud_path, aligned_cloud_path):
#                        ((1337, 0, 0), (1337, 0, 0))}
#
#     return dictionary_line
#
#
# def use_algorithmus_on_dictionary (reference_dictionary_name, algorithmus_function, results_save_name=None ):
#     '''
#     Uses a dictionary of reference cloud file_paths as keys and a list of corresponding aligned cloud file_paths as
#     values
#
#     Input:
# file_paths_dictionary (string):  Dictionary with reference_paths as keys and paths of aligned clouds as values
# algorithmus_function (function): Function that returns dict {(reference path, aligned_path): (translation, mse)}
# results_save_name (string):      Results will be saved as data/results_save_path.pkl. Values may be overwritten.
#     '''
#
#     # parse the reference values saved in a file
#     reference_dictionary = input_output.load_obj (reference_dictionary_name )
#     file_paths_dictionary = get_reference_data_paths (reference_dictionary )
#
#     # before start, check if files exist
#     for key in file_paths_dictionary:
#         if (input_output.check_for_file (key ) is False):
#             print ("File " + key + " was not found. Aborting.")
#             return False
#         for aligned_cloud_path in file_paths_dictionary[key]:
#             if (input_output.check_for_file (aligned_cloud_path ) is False):
#                 print ("File " + aligned_cloud_path + " was not found. Aborting.")
#                 return False
#
#     algorithmus_results = {}    # dictionary
#
#     # create a list of tuples from reference and aligned cloud file paths
#     for ref_cloud_path in file_paths_dictionary:
#         for aligned_cloud_path in file_paths_dictionary[ref_cloud_path]:   # multiple aligned clouds possible
#
#             folder, reference_file_name = input_output.get_folder_and_file_name (ref_cloud_path)
#             folder, aligned_file_name = input_output.get_folder_and_file_name (aligned_cloud_path)
#             plot_title = folder + ' ' + aligned_file_name + ' to ' + reference_file_name
#
#             # call the algorithmus supplied by algorithmus_function
#             algorithmus_results.update (algorithmus_function (ref_cloud_path, aligned_cloud_path, plot_title ))
#
#     if (results_save_name is not None ):
#         input_output.save_obj (algorithmus_results, results_save_name)
#
#     # prints the values computed along with the ground truth in the dictionary
#     compare_results (algorithmus_results, reference_dictionary )
#
#     return True
#
#
# print ("\n\nComputing Consensus for each cloud pair in reference_translations returns: "
#        + str(use_algorithmus_on_dictionary ("reference_translations_dict",
#                                             an_algorithmus,
#                                             "test_results_dict" )))
#
# print (input_output.load_obj ("test_results_dict"))


# # find the column containing the maximum value of a row
#print (numpy_cloud[np.argmax(numpy_cloud[:, 2]), :])


# # find the row containing a certain subset of values and move it to the end of the array
# numpy_cloud = np.array([[1.1, 2.1, 3.1, 0],
#                         [1.2, 2.2, 3.2, 0],
#                         [171.3, 172.3, 3.3, 0],
#                         [1.4, 2.4, 3.4, 0],
#                         [0, 0, 3.5, 0],
#                         [11.6, 2.6, 3.4, 0]] )
#
# best_alignment = [171.3, 172.3, 3.3]
#
# print (numpy_cloud)
# print ()
# best_alignment_index = (numpy_cloud[:, :3] == best_alignment).all(axis=1).nonzero()[0][0]
# best_alignment_row = numpy_cloud[best_alignment_index, :].reshape (1, -1)
# numpy_cloud = np.delete (numpy_cloud, best_alignment_index, axis=0)
# numpy_cloud = np.concatenate ((numpy_cloud, best_alignment_row), axis=0)
#
# print (numpy_cloud)


# # angle speed test for loop and monolith and einsum
# from modules import input_output
# import time
# import numpy.linalg as la
# from scipy.spatial import distance as dist
#
#
# def get_normals (numpy_cloud, field_labels_list ):
#
#     # remove any spaces around the labels
#     field_labels_list = [label.strip () for label in field_labels_list]
#
#     if ('Nx' in field_labels_list
#        and 'Ny' in field_labels_list
#        and 'Nz' in field_labels_list ):
#         indices = []
#         indices.append (field_labels_list.index('Nz' ))
#         indices.append (field_labels_list.index('Ny' ))
#         indices.append (field_labels_list.index('Nx' ))
#     else:
#         raise ValueError ("This Cloud is missing one of the required fields:
#                           'Nx', 'Ny', 'Nz'. Compute Normals first.")
#
#     return numpy_cloud[:, indices]
#
#
# def angle_between (vector_1, vector_2):
#     """ Returns the angle in radians between vectors 'vector_1' and 'vector_2' """
#
#     res = np.arccos(np.clip(np.dot(vector_1, vector_2), -1.0, 1.0))
#
#     return res
#
#
# def alternative_angle_between (vector_array_1, vector_array_2, step=1000 ):
#
#     # prepare results vector with lenght of number of points
#     results = np.zeros ((vector_array_1.shape[0], 1 ))
#
#     # np.dot (vector_array_1[i:], vector_array_2.T) computes a gigantic matrix. In order to save RAM space, it has to
#     # be done in batches
#     for i in range (0, vector_array_1.shape[0], step ):
#         if (i + step > vector_array_1.shape[0]):
#             results[i:] = np.arccos (
#                            np.diagonal (
#                             np.clip (
#                              np.dot (vector_array_1[i:, :],
#                                      vector_array_2[i:, :].T ), -1, 1 ))).reshape (-1, 1)
#         else:
#             results[i:i+step] = np.arccos (
#                                  np.diagonal (
#                                   np.clip (
#                                    np.dot (vector_array_1[i:i+step, :],
#                                            vector_array_2[i:i+step, :].T ), -1, 1 ))).reshape (-1, 1)
#
#     return results
#
#
# def alternative_angle_between_nan (vector_array_1, vector_array_2, step=1000 ):
#
#     # prepare results vector with lenght of number of points
#     results = np.zeros ((vector_array_1.shape[0], 1 ))
#
#     # np.dot (vector_array_1[i:], vector_array_2.T) computes a gigantic matrix. In order to save RAM space, it has to
#     # be done in batches
#     for i in range (0, vector_array_1.shape[0], step ):
#         # the last step, all values until the end of the array
#         if (i + step > vector_array_1.shape[0]):
#             results[i:] = np.arccos (
#                            np.diagonal (np.dot (vector_array_1[i:, :],
#                                      vector_array_2[i:, :].T ))).reshape (-1, 1)
#         # every other step, taking values in the range of step
#         else:
#             results[i:i+step] = np.arccos (
#                                  np.diagonal (
#                                   np.dot (vector_array_1[i:i+step, :],
#                                           vector_array_2[i:i+step, :].T ))).reshape (-1, 1)
#
#     # replace nan values with 90 degrees angle difference
#     return np.where (np.isnan (results), 1.57079632679, results )
#
#
# def alternative_angle_between_noclip (vector_array_1, vector_array_2, step=1000 ):
#
#     # prepare results vector with lenght of number of points
#     results = np.zeros ((vector_array_1.shape[0], 1 ))
#
#     # np.dot (vector_array_1[i:], vector_array_2.T) computes a gigantic matrix. In order to save RAM space, it has to
#     # be done in batches
#     for i in range (0, vector_array_1.shape[0], step ):
#         if (i + step > vector_array_1.shape[0]):
#             results[i:] = np.arccos (
#                            np.diagonal (np.dot (vector_array_1[i:, :],
#                                      vector_array_2[i:, :].T ))).reshape (-1, 1)
#         else:
#             results[i:i+step] = np.arccos (
#                                  np.diagonal (np.dot (vector_array_1[i:i+step, :],
#                                            vector_array_2[i:i+step, :].T ))).reshape (-1, 1)
#
#     return results
#
#
# def simple_loop_angle (vector_array_1, vector_array_2 ):
#
#     results = np.zeros ((vector_array_1.shape[0], 1 ))
#     for i in range (vector_array_1.shape[0]):
#         results[i] = vector_array_1[i, :].dot (vector_array_2[i, :] )
#
#     return np.arccos (np.clip (results, -1, 1 ))
#
#
# def einsum_angle_between (vector_array_1, vector_array_2 ):
#
#     # diagonal of dot product
#     diag = np.clip (np.einsum('ij,ij->i', vector_array_1, vector_array_2 ), -1, 1 )
#
#     return np.arccos (diag )
#
#
# def load_example_cloud ():
#
#     # # big cloud
#     numpy_cloud, numpy_cloud_field_labels = input_output.conditionalized_load(
#         'clouds/Regions/Yz Houses/ALS16_Cloud_reduced_normals_cleared.asc' )
#
#     corresponding_cloud, corresponding_cloud_field_labels = input_output.conditionalized_load (
#         'clouds/Regions/Yz Houses/DSM_Cloud_reduced_normals.asc' )
#
#     return numpy_cloud, numpy_cloud_field_labels, corresponding_cloud, corresponding_cloud_field_labels
#
#
# # ### prepare ####
# numpy_cloud, numpy_cloud_field_labels, corresponding_cloud, corresponding_cloud_field_labels \
#     = load_example_cloud ()
#
# normals_numpy_cloud = get_normals (numpy_cloud, numpy_cloud_field_labels )
# normals_corresponding_cloud = get_normals (corresponding_cloud, corresponding_cloud_field_labels )
#
#
# step = 58
# print ("Step: " + str(step ))
#
# # Step: 40
# # Loop Process Time: 2.312503254413605
# # Monolith Process Time: 0.15936983108520508
# # No Clip Monolith Process Time: 0.1318157744407654
# # NAN Monolith Process Time: 0.1287021803855896
#
# # Step: 50
# # Loop Process Time: 2.491855809688568
# # Monolith Process Time: 0.16239188432693483
# # No Clip Monolith Process Time: 0.1278723359107971
# # NAN Monolith Process Time: 0.12739877462387084
#
# # Step: 56
# # Loop Process Time: 2.857189098993937
# # Monolith Process Time: 0.16701097488403321
# # No Clip Monolith Process Time: 0.13585476875305175
# # NAN Monolith Process Time: 0.14023882548014324
#
# # Step: 58
# # Loop Process Time: 2.310372988382975
# # Monolith Process Time: 0.1322481155395508
# # No Clip Monolith Process Time: 0.10442533493041992
# # NAN Monolith Process Time: 0.10448430379231771
#
# # Step: 60
# # Loop Process Time: 2.739641170501709
# # Monolith Process Time: 0.16630157709121704
# # No Clip Monolith Process Time: 0.13103942155838014
# # NAN Monolith Process Time: 0.1315992569923401
#
# # Step: 62
# # Loop Process Time: 2.4043121496836344
# # Monolith Process Time: 0.1526663939158122
# # No Clip Monolith Process Time: 0.12144707043965658
# # NAN Monolith Process Time: 0.12466743787129721
#
#
# monolith_time = 0
# monolith_nc_time = 0
# monolith_nan_time = 0
# loop_time = 0
# simple_loop_time = 0
# einsum_time = 0
# times = 25
# for i in range (times):
#
#     measure = time.time ()
#     # slow looped process
#     results_loop = normals_numpy_cloud.shape[0] * [None]
#     for index, (vec1, vec2) in enumerate(
#           zip (normals_numpy_cloud, normals_corresponding_cloud[:normals_numpy_cloud.shape[0], :] )):
#         results_loop[index] = (angle_between (vec1, vec2 ) )
#     loop_time += time.time () - measure
#
#     measure = time.time ()
#     results_monolith = alternative_angle_between (
#                   normals_numpy_cloud, normals_corresponding_cloud[:normals_numpy_cloud.shape[0], :], step )
#     monolith_time += time.time () - measure
#
#     measure = time.time ()
#     results_nc_monolith = alternative_angle_between_noclip (
#                   normals_numpy_cloud, normals_corresponding_cloud[:normals_numpy_cloud.shape[0], :], step )
#     monolith_nc_time += time.time () - measure
#
#     measure = time.time ()
#     results_nan_monolith = alternative_angle_between_nan (
#                   normals_numpy_cloud, normals_corresponding_cloud[:normals_numpy_cloud.shape[0], :], step )
#     monolith_nan_time += time.time () - measure
#
#     measure = time.time ()
#     results_simple_loop = simple_loop_angle (
#                   normals_numpy_cloud, normals_corresponding_cloud[:normals_numpy_cloud.shape[0], :] )
#     simple_loop_time += time.time () - measure
#
#     measure = time.time ()
#     results_einsum = einsum_angle_between (#
#                   normals_numpy_cloud, normals_corresponding_cloud[:normals_numpy_cloud.shape[0], :] )
#     einsum_time += time.time () - measure
#
# #
# monolith_time = monolith_time / times
# monolith_nc_time = monolith_nc_time / times
# monolith_nan_time = monolith_nan_time / times
# loop_time = loop_time / times
# simple_loop_time = simple_loop_time / times
# einsum_time = einsum_time / times
#
#
# print ("\nStep: " + str(step ))
# print ("Loop Process Time: " + str(loop_time ))
# print ("Monolith Process Time: " + str(monolith_time ))
# print ("No Clip Monolith Process Time: " + str(monolith_nc_time ))
# print ("NAN Monolith Process Time: " + str(monolith_nan_time ))
# print ("Simple Loop Time: " + str(simple_loop_time ))
# print ("Einsum Time: " + str(einsum_time ))
# #
# print ("\n\nloop:\n" + str(results_loop[:10]))
# print ("monolith:\n" + str(results_monolith[:10].T))
# print ("noclip monolith:\n" + str(results_nc_monolith[:10].T))
# print ("NAN monolith:\n" + str(results_nan_monolith[:10].T))
# print ("Simple Loop:\n" + str(results_simple_loop[:10].T))
# print ("Einsum Result:\n" + str(results_einsum[:10].T))


# # how to append to a list
# list1 = [1, 2, 3]
# list2 = [4, 5, 6]
# list1.append (list2 )
#
# print (list1)
#
# list1 = [1, 2, 3]
# list1 = list1 + list2
#
# print (list1)


# # Speed test of array-wise normla vector angle_between computation
# from modules import normals
# import time
#
#
# def normalize_vector_array (vector_array ):
#     norms = np.apply_along_axis(np.linalg.norm, 1, vector_array )
#     return vector_array / norms.reshape (-1, 1 )
#
#
# def angle_between(vector_1, vector_2):
#     """ Returns the angle in radians between vectors 'vector_1' and 'vector_2' """
#
#     if (vector_1 is None or vector_2 is None or None in vector_1 or None in vector_2 ):
#         return None
#
#     vector_1 = normals.normalize_vector (vector_1 )
#     vector_2 = normals.normalize_vector (vector_2 )
#
#     return np.arccos(np.clip(np.dot(vector_1, vector_2), -1.0, 1.0))
#
#
# vector_array_1 = np.random.uniform (0, 1, size=(10000, 3 ))
# vector_array_2 = np.random.uniform (0, 1, size=(10000, 3 ))
#
# vector_array_1 = normalize_vector_array (vector_array_1 )
# vector_array_2 = normalize_vector_array (vector_array_2 )
#
# # Pure Numpy Process
# start = time.time()
# arccos = np.arccos (vector_array_1.dot (vector_array_2.T)[:, 0] )
# end1 = time.time() - start
#
# # Looped Numpy Process
# start = time.time()
# results = len (vector_array_1 ) * [None]
# for index, (vec1, vec2) in enumerate(zip (vector_array_1, vector_array_2 )):
#     results[index] = (angle_between (vec1, vec2 ) )
# end2 = time.time() - start
#
# print ("Numpy Time = " + str (end1 ))
# print ("Standard Time = " + str (end2 ))
#
# print ("arccos: " + str (arccos ))
# print ("results: " + str (results ))


# # function as argument
# def something ():
#     return "Something"
#
#
# def function_taking_function (some_python_function ):
#     print (str(some_python_function ()))
#
#
# function_taking_function (something )


# # test normal calculation
# from modules import normals
# from modules import input_output
# import sys
# #
# # # ply files
# # # numpy_cloud_1 = input_output.load_ply_file ('clouds/laserscanning/', 'plane1.ply')    # 3806 points
# # #numpy_cloud_2 = input_output.load_ply_file ('clouds/laserscanning/', 'plane2.ply')    # 3806 points
# #
# # # las files
# # #numpy_cloud_1 = input_output.load_las_file ('clouds/laserscanning/plane1.las')    # 3806 points
# # #numpy_cloud_2 = input_output.load_las_file ('clouds/laserscanning/plane2.las')    # 3806 points
# #
# # simple plane
# # numpy_cloud_1 = np.array ([[-1, 0, 0],   # +x
# #                           [2, 0, 0],  # -x
# #                           [0, 2, 0]
# #                           [2, 0, 200]])  # +y
#
# numpy_cloud_1 = np.array ([[-1, 0, 0],   # +x
#                           [2, 0, 0],  # -x
#                           [0, 2, 0],
#                           [0, 3, 0],
#                           [0, 4, 0],
#                           [0, 5, 0],
#                           [-1, 0, 200],   # +x
#                           [2, 0, 200],  # -x
#                           [0, 2, 200],
#                           [0, 3, 200],
#                           [0, 4, 200],
#                           [0, 5, 200]])  # +y
#
# # numpy_cloud_1 = np.random.uniform (-10, 10, (300, 3))
# #
# # 1st cloud
# normal_vector, consensus_points, _, _, _ = \
#     normals.ransac_plane_estimation (numpy_cloud_1, 0.1, fixed_point=numpy_cloud_1[0, :], w=0.8 )
# print ('\nRANSAC, Cloud 1:\nnormal_vector: ' + str(normal_vector ))
# print ('consensus_points:\n' + str(consensus_points ) + '\n')
# #
# normal_vector, sigma, mass_center, _ = normals.PCA (consensus_points )
# print ('\nPCA, Cloud 1:\nnormal_vector: ' + str(normal_vector ))
# print ('sigma: ' + str(sigma ))
# print ('mass_center: ' + str(mass_center ) + '\n')


# corresponding_cloud = np.array([[1.1, 0, 0],
#                                 [2.2, 0, 0],
#                                 [3.3, 0, 0],
#                                 [4.4, 0, 0],
#                                 [5.5, 0, 0],
#                                 [6.6, 0, 0]] )
#
# consensus_count, consensus_vector = cloud_consensus (numpy_cloud, corresponding_cloud, 0.4 )
# print ("consensus_count: " + str(consensus_count ))
# print ("consensus_vector:\n" + str(consensus_vector ))

# print (vector_array_distance (numpy_cloud, corresponding_cloud ))


# # reshape arrays to concat them
#import math
#
#
# an_array = np.array ((1, 2, 3)).reshape (1, 3)
#
# print ("numpy_cloud.shape: " + str(numpy_cloud.shape ))
# print ("an_array.shape: " + str(an_array.shape ))
# print (
#     np.concatenate (  (numpy_cloud, an_array, an_array), axis=0 )
# )

# # sort a list of tuples
# list = [('folder/folder/a', 'folder/folder/b'),
#         ('folder/folder/b', 'folder/folder/a'),
#         ('folder/folder/a', 'folder/folder/c'),
#         ('folder/folder/c', 'folder/folder/a')]
#
# print (sorted (list))


# # clear up the wicked output of
# import sklearn.neighbors.kd_tree
#
#
# numpy_cloud = np.array([[1.1, 0, 0],
#                         [1.2, 0, 0],
#                         [1.3, 0, 0],
#                         [1.4, 0, 0],
#                         [1.5, 0, 0],
#                         [1.6, 0, 0]] )
#
# # build a kdtree
# tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')
# query_radius = 0.3  # m
#
# for index, point in enumerate (numpy_cloud ):
#
#     thing = tree.query_radius(point.reshape (1, -1), r=query_radius )
#     thing = [value2 for value in thing for value2 in value]
#
#     print (thing)


# import random
#
#
# numpy_cloud = np.concatenate ((numpy_cloud, numpy_cloud, numpy_cloud, numpy_cloud), axis=0 )
# sample_factor = 6
#
# # sample deterministic
# print ("a")
# print (numpy_cloud[::sample_factor])
#
# # sample random
# indices = random.sample(range(0, numpy_cloud.shape[0] ), int (numpy_cloud.shape[0] / sample_factor ))
# print ("\nint: " + str(int (numpy_cloud.shape[0] / sample_factor )))
# print (numpy_cloud[indices, :] )


# {: .8f}.format (value)


# # delete everything that has more or equal to 20 in the 8th row _cleared:
# numpy_cloud = numpy_cloud[numpy_cloud[:, 7] < 20]
# cloud_altered = True


# def display_small_cloud (cloud ):
#     fig = pyplot.figure()
#     ax = Axes3D(fig)
#
#     for i in range(0, cloud.size):
#         ax.scatter(cloud[i][0], cloud[i][1], cloud[i][2])
#
#     pyplot.show()


# # reduce and compute normals for a cloud specified by file_path
# import numpy as np
# import sklearn.neighbors    # kdtree
# import normals
# #import math
# import input_output
# from os.path import splitext
# from conversions import reduce_cloud
# import psutil
#
# file_path = "clouds/Regions/Everything/DSM_Cloud_333165_59950 - Cloud.las"
# filename, file_extension = splitext(file_path )
#
# field_labels_list = ['X', 'Y', 'Z']
# previous_folder = ""
#
# # load the file, then reduce it
# if ("DSM_Cloud" in file_path):
#
#     # Load DIM cloud
#     numpy_cloud = input_output.load_las_file (file_path, dtype="dim" )
#     numpy_cloud[:, 3:6] = numpy_cloud[:, 3:6] / 65535.0  # rgb short int to float
#     field_labels_list.append ('Rf ' 'Gf ' 'Bf ' 'Classification ')
# else:
#     # Load ALS cloud
#     numpy_cloud = input_output.load_las_file (file_path, dtype="als")
#     field_labels_list.append('Intensity '
#                              'Number_of_Returns '
#                              'Return_Number '
#                              'Point_Source_ID '
#                              'Classification ')
#
# print ("------------------------------------------------\ncloud successfully loaded!")
#
# # all clouds in one folder should get the same trafo
# if (len(file_path.split ('/')) == 1):
#     current_folder = file_path
# else:
#     current_folder = file_path.split ('/')[-2]
# if (current_folder != previous_folder):
#     min_x_coordinate, min_y_coordinate = reduce_cloud (numpy_cloud, return_transformation=True )[1:]
# previous_folder = current_folder
#
# # reduce
# numpy_cloud[:, 0] = numpy_cloud[:, 0] - min_x_coordinate
# numpy_cloud[:, 1] = numpy_cloud[:, 1] - min_y_coordinate
#
# print ("------------------------------------------------\ncloud successfully reduced!")
# # compute normals
# # build a kdtree
# tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')
#
# # set radius for neighbor search
# query_radius = 5.0  # m
# if ("DSM_Cloud" in file_path):  # DIM clouds are roughly 6 times more dense than ALS clouds
#     query_radius = query_radius / 6
#
# # kdtree radius search
# list_of_point_indices = tree.query_radius(numpy_cloud, r=query_radius )     # this floods memory
# additional_values = np.zeros ((numpy_cloud.shape[0], 4 ))
#
# # compute normals for each point
# for index, point_neighbor_indices in enumerate (list_of_point_indices ):
#
#     if (psutil.virtual_memory().percent > 95.0):
#         print (print ("!!! Memory Usage too high: " + str(psutil.virtual_memory().percent) + "%. Breaking loop."))
#
#     # you can't estimate a cloud with less than three neighbors
#     if (len (point_neighbor_indices) < 3 ):
#         continue
#
#     # do a Principal Component Analysis with the plane points obtained by a RANSAC plane estimation
#     normal_vector, sigma, mass_center = normals.PCA (
#                 normals.ransac_plane_estimation (numpy_cloud[point_neighbor_indices, :],   # point neighbors
#                                                  threshold=0.3,  # max point distance from the plane
#                                                  w=0.6,         # probability for the point to be an inlier
#                                                  z=0.90)        # desired probability that plane is found
#                                                  [1] )          # only use the second return value, the points
#
#     # join the normal_vector and sigma value to a 4x1 array and write them to the corresponding position
#     additional_values[index, :] = np.append (normal_vector, sigma)
#
#
# print ("------------------------------------------------\ncloud successfully norm norm!")
#
# # add the newly computed values to the cloud
# numpy_cloud = np.concatenate ((numpy_cloud, additional_values), axis=1)
# field_labels_list.append('Nx ' 'Ny ' 'Nz ' 'Sigma ' )
#
# print ("------------------------------------------------\nnorm norm added!")
#
# # save the cloud again
# input_output.save_ascii_file (numpy_cloud, field_labels_list, filename + "_reduced_normals.asc" )
#
# print ("Done.")


# # misc tests
# def something ():
#     return (1, 2, 3, 4)
#
#
# print (something ()[1:])
#
# file_path = "This/is/a/very/long/path.las"
# #file_path = "Short/Path.las"
# #file_path = "Path.las"
# #file_path = "This / is //\ a_wierd/\\ path.las"
#
# print (file_path)
# print (len(file_path.split ('/')))
# print (file_path.split ('/')[-2])


# field_names_list = ["x", "y", "z", "i" ]
#
# field_names_list = ['{0} '.format(name) for name in field_names_list]
# str1 = ''.join(field_names_list)
# leading_line = "//" + str1
#
# print (leading_line)


# import numpy as np
# import random
#
#
# numpy_cloud = np.array([[1.1, 2.1, 3.1],
#                         [1.2, 2.2, 3.2],
#                         [1.3, 2.3, 3.3],
#                         [1.4, 2.4, 3.4],
#                         [1.5, 2.5, 3.5],
#                         [1.6, 2.6, 3.6]] )
#
# indices = random.sample(range(0, numpy_cloud.shape[0] ), 6 )
# print ('indices: ' + str (indices ))
#
# for idx in indices:
#     print (numpy_cloud[idx, :] )


# numpy_cloud = np.array([[1.1, 2.1, 3.1],
#                         [1.2, 2.2, 3.2],
#                         [1.3, 2.3, 3.3],
#                         [1.4, 2.4, 3.4],
#                         [1.5, 2.5, 3.5],
#                         [1.6, 2.6, 3.6]] )
#
# cloud = [[1.1, 2.1, 3.1],
#          [1.2, 2.2, 3.2],
#          [1.3, 2.3, 3.3],
#          [1.4, 2.4, 3.4],
#          [1.5, 2.5, 3.5],
#          [1.6, 2.6, 3.6]]
#
# cloud.append (numpy_cloud [2, :].tolist ())
#
#
# win_path = '/some/path/containing/a/Windows Directory'
# linux_path = '/some/path/containing/a/linux_directory'
# print (win_path.replace (' ', '\ ' ))


# import numpy as np
# import input_output
#
# min_x = 1.2
# max_x = 1.5
# min_y = 2.3
# max_y = 2.6
# numpy_cloud = np.array([[1.1, 2.1, 3.1],
#                         [1.2, 2.2, 3.2],
#                         [1.3, 2.3, 3.3],
#                         [1.4, 2.4, 3.4],
#                         [1.5, 2.5, 3.5],
#                         [1.6, 2.6, 3.6]] )
#
# subset_cloud = np.array([0, 0 ,0])
#
# # for point in numpy_cloud:
# #     if (point[0] > min_x
# #         and point[0] < max_x
# #         and point[1] > min_y
# #         and point[1] < max_y):
# #         print ("found point " + str(point ))
# #         subset_cloud.append (point)
# #     print ("point: " + str (point ))
# #     print ("point[0]: " + str (point[0] ))
#
#
# subset_cloud = [point for point in numpy_cloud if (point[0] > min_x
#                                                    and point[0] < max_x
#                                                    and point[1] > min_y
#                                                    and point[1] < max_y)]
#
# print ('subset_cloud.shape: ' + str (len (subset_cloud )))
# print ('subset_cloud: ' + str (subset_cloud ))
#
# input_output.save_ply_file(subset_cloud, '', 'test.ply')
# string = '/this/is/a/long/path/and/a/file.txt'
# print (string.rsplit ('/', 1 ) )
#
# from __future__ import print_function
# import numpy as np
# import pcl
#
# points_1 = np.array([[0, 0, 0],
#                      [1, 0, 0],
#                      [0, 1, 0],
#                      [1, 1, 0]], dtype=np.float32)
# points_2 = np.array([[0, 0, 0.2],
#                      [1, 0, 0],
#                      [0, 1, 0],
#                      [1.1, 1, 0.5]], dtype=np.float32)
#
# pc_1 = pcl.PointCloud(points_1)
# pc_2 = pcl.PointCloud(points_2)
#
# #
# kd = pc_1.make_kdtree_flann()
#
# print('pc_1:')
# print(points_1)
# print('\npc_2:')
# print(points_2)
# print('\n')
#
# # find the single closest points to each point in point cloud 2
# # (and the sqr distances)
# indices, sqr_distances = kd.nearest_k_search_for_cloud(pc_2, 1)
# for i in range(pc_1.size):
#     print('index of the closest point in pc_1 to point %d in pc_2 is %d' % (i, indices[i, 0]))
#     print('the squared distance between these two points is %f' % sqr_distances[i, 0])


# #################################################################
# print ("\nExample 1: Input Cloud")
#
# # define
# input_cloud = np.array([[1.1,  2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# #input_cloud = np.reshape (input_cloud, (-1, 1))
#
#
# # check
# print ('Size: ' + str(input_cloud.size ))
# print ('Shape: ' + str(input_cloud.shape ))
# print ('Cloud:\n' + str(input_cloud ))


# def normalize_vector (vector ):
#     '''
#
#     '''
#     # check if vector is a matrix
#     if (len (vector.shape ) > 1 ):
#         print ("In normalize_vector: Vector is out of shape.")
#         return vector
#
#     vector_magnitude = 0
#     for value in vector:
#         vector_magnitude = vector_magnitude + np.float_power (value, 2 )
#     vector_magnitude = np.sqrt (vector_magnitude )
#
#     return vector / vector_magnitude
#
#
# vector = np.array ([40, 10, 0], float)
#
# print ('Vector: ' + str(vector ))
# print ('Vector Norm: ' + str(normalize_vector (vector )))


# #eigenvalues, eigenvectors = np.linalg.eig(input_cloud )
# eigenvalues = np.zeros (3 )
# eigenvectors = np.zeros ((3, 3 ))
#
# evals = np.array ([2, 3, 1] )
# evecs = np.array (([3,2,1], [6,5,4], [9,8,7] ))
#
# print ('Before:')
# print ('Values: ' + str(evals ))
# print ('Vectors: \n' + str(evecs ))
#
# # sort them
# indices = np.argsort (-evals )
# for loop_count, index in enumerate(indices ):
#     eigenvalues[loop_count] = evals[index]
#     eigenvectors[:, loop_count] = evecs[:, index]
#
# print ('After:')
# print ('Values: ' + str(eigenvalues ))
# print ('Vectors: \n' + str(eigenvectors ))


# # change
# input_cloud = np.concatenate ((input_cloud, input_cloud, input_cloud, input_cloud), axis = -1)
#
# print (type(input_cloud ))
# print (type (pcl.PointCloud ( )))
# print (type (pcl.PointCloud_PointXYZI ( )))
# print ('\n')
#
# input_cloud = np.subtract (input_cloud[:, 0:3], np.array([0.1, 0.5, 2]))
