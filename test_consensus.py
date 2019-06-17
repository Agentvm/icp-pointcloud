# test consensus
from modules import input_output
from modules import consensus
#from modules import normals
import numpy as np
import random
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def normalize_vector_array (vector_array ):
    norms = np.apply_along_axis(np.linalg.norm, 1, vector_array )
    return vector_array / norms.reshape (-1, 1 )


# small cloud
#numpy_cloud = np.random.uniform (-10, 10, (100, 3 ))
numpy_cloud = np.array([[1, 0, 0],
                        [2, 0, 0],
                        [0, 1, 0],
                        [0, 2, 0],
                        [0, 0, 1],
                        [0, 0, 2]] )

numpy_normals = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0, 1, 1]] )

# random values
numpy_cloud = np.random.uniform (-10, 10, size=(1000, 3 ))
numpy_normals = normalize_vector_array (np.random.uniform (0, 1, size=(1000, 3 )))

# concat
field_labels_list = numpy_cloud_field_labels = corresponding_cloud_field_labels = ['X', 'Y', 'Z', 'Nx ', 'Ny ', 'Nz ']
numpy_cloud = np.concatenate ((numpy_cloud, numpy_normals), axis=1)
print ('numpy_cloud.shape: ' + str (numpy_cloud.shape ))
# print ('numpy_cloud:\n' + str (numpy_cloud ))

# displace
random1 = np.random.uniform ((-1, -1, -1, 0, 0, 0), (1, 1, 1, 0, 0, 0 ))
corresponding_cloud = numpy_cloud + random1

# noise
numpy_cloud = numpy_cloud + np.random.uniform (-0.01, 0.01, size=(1, 6 ))
corresponding_cloud = corresponding_cloud + np.random.uniform (-0.01, 0.01, size=(1, 6 ))

# # big cloud
# numpy_cloud, numpy_cloud_field_labels = input_output.conditionalized_load(
#     'clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc' )
#
# corresponding_cloud, corresponding_cloud_field_labels = input_output.conditionalized_load (
#     'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc' )

# reach consenssu
best_alignment, best_consensus_count, best_alignment_consensus_vector = \
    consensus.cubic_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                           corresponding_cloud, corresponding_cloud_field_labels,
                           threshold=30 * (math.pi/180 ),
                           cubus_length=2,
                           step=.2 )
# best_alignment, best_consensus_count, best_alignment_consensus_vector = \
#     consensus.cubic_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
#                            corresponding_cloud, corresponding_cloud_field_labels,
#                            threshold=0.009,
#                            cubus_length=2,
#                            step=.2 )

print ("Random Offset: " + str(random1 ))
print ("Point Picking Offset: (-0.82777023,  0.16250610,  0.19129372)")
plt.show ()
