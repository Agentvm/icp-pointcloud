
import numpy as np
#import pcl


#################################################################
print ("\nExample 1: Input Cloud")

# define
input_cloud = np.array([[1.1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#input_cloud = np.reshape (input_cloud, (-1, 1))


# check
print ('Size: ' + str(input_cloud.size ))
print ('Size: ' + str(input_cloud.shape ))
print ('Cloud:\n' + str(input_cloud ))


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
