import numpy as np
import input_output
from custom_clouds import CustomCloud

numpy_cloud = np.array([[1.1111, 2.1, 3.1, 0.5],
                        [1.2, 2.2, 3.2, 0.5],
                        [1.3, 2.3, 3.3, 0.5],
                        [1.4, 2.4, 3.4, 0.5],
                        [1.5, 2.5, 3.5, 0.5],
                        [1.6, 2.6, 3.6, 0.5]] )

custom_cloud = CustomCloud.initialize_xyz(numpy_cloud )
input_output.save_ascii_file(custom_cloud.data, custom_cloud.labels )


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
