import numpy as np

numpy_cloud = np.array([[1.1, 2.1, 3.1],
                        [1.2, 2.2, 3.2],
                        [1.3, 2.3, 3.3],
                        [1.4, 2.4, 3.4],
                        [1.5, 2.5, 3.5],
                        [1.6, 2.6, 3.6]] )

#


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


# # delete everything that has more or equal to 20 in the 8th row:
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

# # cloud not mutable :(
#from custom_clouds import CustomCloud
#
#
# a = CustomCloud.initialize_xyz (numpy_cloud )
#
# a.fields.y = 0
#
# print (a.fields.y)
#
# for point in a:
#     point.x = 0
#     print (point)
#
# print (a)


# import input_output
#
#
# big_numpy_cloud = np.array([[1010, 2100, 3.1, 0.5, 1010, 2100, 3.1],
#                             [1620, 1500, 3.2, 0.5, 1010, 2100, 3.1],
#                             [1880, 1470, 3.3, 0.5, 1010, 2100, 3.1]] )
#
# #print (big_numpy_cloud[:, 1] - 1000.0)
# input_output.save_ascii_file (big_numpy_cloud,
#                              ['x', 'y', 'z', 'i', 'normx', 'normy', 'normz'],
#                              "clouds/tmp/input_output_test.asc" )
#
# custom_cloud = input_output.load_ascii_file ("clouds/tmp/input_output_test.asc", return_custom_cloud=True)
#
# print ("Custom:\n" + str(custom_cloud ))


# #time difference between numpy and custom cloud ~ 1/1.5
# from custom_clouds import CustomCloud
# import time
# numpy_cloud = np.random.rand (1000000, 3) * 1000
#
# a = time.time ()
# custom_cloud = CustomCloud.initialize_xyz(numpy_cloud )
# print (time.time () - a )
# a = time.time ()
# input_output.save_ascii_file(custom_cloud.data, custom_cloud.labels )
# print (time.time () - a )


# from custom_clouds import CustomCloud
# import time
# # init both clouds
# numpy_cloud = np.random.rand (10000, 3 ) * 1000
# custom = CustomCloud.initialize_xyz (numpy_cloud )
#
# print ('\nnumpy_cloud:\n' + str(numpy_cloud ))
# print ('\nCustom cloud:\n' + str(custom ))
#
# # test time
# numpy_time = time.time ()
# for point in numpy_cloud:
#     point = point * 3
# print ("numpy time: " + str (time.time () - numpy_time ))
#
# # test time
# custom_time = time.time ()
# for point in custom:
#     point = point * 3
# print ("custom_time time: " + str (time.time () - custom_time ))


# custom = CustomCloud.initialize_xyzi (numpy_cloud )
# happy = np.array((0.0, 1.0, 1.0, 0.0, 1.0, 1.0 ))
#
# # print
# print ('\ncustom.fields: ' + str(custom.labels ))
# print ('\nhas happy: ' + str(custom.has_field ("happy" ) ))
# print ('\nCustom cloud:\n' + str(custom ))
#
# # add some random field
# print ('\n\n------------------------------------------\nadding field happy')
# custom.add_field (happy, "happy" )
#
# # print
# print ('\ncustom.fields: ' + str(custom.labels ))
# print ('\nhas happy: ' + str(custom.has_field ("happy" ) ))
# print ('\nCustom cloud:\n' + str(custom ))
#
# print ('\ncustom.fields.happy: ' + str(custom.fields.happy ) + '\n')
#
# for point in custom:
#     print ('point: ' + str(point ))
#     if (custom.has_field ("happy" )):
#         print ('point.happy: ' + str(point.happy ))


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
