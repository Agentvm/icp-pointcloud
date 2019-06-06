import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sklearn.neighbors    # kdtree
import input_output
import time


def vector_array_distance (xyz_array, compared_xyz_array=None ):
    '''
    Computes distances between the vectors of two arrays. Set second array to none to compute magnitudes instead.
    '''

    if (compared_xyz_array is None):
        compared_xyz_array = np.zeros((xyz_array.shape[0], 3 ))

    xyz_array = xyz_array[:, 0:3]
    compared_xyz_array = compared_xyz_array[:, 0:3]

    # The actual process
    output = np.sqrt(np.sum((xyz_array - compared_xyz_array )**2, axis=1))

    return output.reshape ((xyz_array.shape[0], 1 ))


def cloud_consensus (numpy_cloud, corresponding_cloud, threshold ):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        numpy_cloud ([n, 3] np.array):
        corresponding_cloud ([1, 3] np.array):
        threshold (float):

    Output:
        consensus_count (int):                  Number of points with neighbors in suitable range
        consensus_vector ([n, 1] np.array):     Contains 1 if the point had a neighbor in threshold range, else 0
    '''
    start_time = time.time ()

    part_time_1 = time.time ()
    numpy_cloud = numpy_cloud[:, 0:3]
    corresponding_cloud = corresponding_cloud[:, 0:3]
    part_time_1 = time.time () - part_time_1

    part_time_2 = time.time ()
    tree = sklearn.neighbors.kd_tree.KDTree (corresponding_cloud, leaf_size=40, metric='euclidean')
    part_time_2 = time.time () - part_time_2

    part_time_3 = time.time ()
    list_consensus_counts = tree.query_radius (numpy_cloud, threshold, return_distance=False, count_only=True)
    part_time_3 = time.time () - part_time_3

    part_time_4 = time.time ()
    consensus_vector = np.array([1 if count > 0 else 0 for count in list_consensus_counts ])
    part_time_4 = time.time () - part_time_4

    return np.sum(consensus_vector), consensus_vector, (time.time () - start_time, part_time_1, part_time_2, part_time_3, part_time_4)


def cubic_cloud_consensus (numpy_cloud, compared_cloud, threshold, cubus_length, step, return_consensus_cloud=False ):
    '''
    Translates compared_cloud in lenghts of step inside a cubus-shaped space and, for every step, checks how many points
    of cloud numpy_cloud have a neighbor within threshold range in compared_cloud.

    Input:
        numpy_cloud ([n, 3] np.array):
        compared_cloud ([1, 3] np.array):
        threshold (float):                  Threshold that defines the range at which a point is counted as neigbor
        cubus_length (float):               Cubus center is (0, 0, 0). Half of cubus_length is backwards, half forwards.
        step (float):

    Output:
        best_alignment ((x, y, z) tuple ):
        best_alignment_consensus_count (int):
        consensus_cube ((n, 4) numpy array):
    '''

    consensus_round_time = 0
    consensus_part_time_1 = 0
    consensus_part_time_2 = 0
    consensus_part_time_3 = 0
    consensus_part_time_4 = 0
    start_time = time.time ()

    numpy_cloud = numpy_cloud[:, 0:3]
    compared_cloud = compared_cloud[:, 0:3]

    # variables
    steps_number = math.ceil (cubus_length / step + 1 )
    cubus_size = steps_number**3
    #print ("cubus_size: " + str (steps_number**3))
    consensus_cloud = np.zeros ((cubus_size, 4 ))     # empty cloud that will take the shape of the cubus
    best_alignment = (0, 0, 0)
    best_alignment_consensus_vector = np.zeros ((numpy_cloud.shape[0], 1) )    #
    best_consensus_count = 0  #

    iteration_count = 0
    for x_iterator in range (-math.floor (steps_number / 2), math.ceil (steps_number / 2 )):
        for y_iterator in range (-math.floor (steps_number / 2), math.ceil (steps_number / 2 )):
            for z_iterator in range (-math.floor (steps_number / 2), math.ceil (steps_number / 2 )):

                if (iteration_count % int(cubus_size / 10) == 0):
                    print ("Progress: " + "{:.1f}".format ((iteration_count / cubus_size) * 100.0 ) + " %" )

                translation = (x_iterator * step,
                               y_iterator * step,
                               z_iterator * step )

                # find consenting points in the translated compared_cloud
                consensus_count, consensus_vector, consensus_time = cloud_consensus (numpy_cloud,
                                                                     compared_cloud + translation,
                                                                     threshold )

                consensus_round_time = consensus_round_time + consensus_time[0]
                consensus_part_time_1 = consensus_part_time_1 + consensus_time[1]
                consensus_part_time_2 = consensus_part_time_2 + consensus_time[2]
                consensus_part_time_3 = consensus_part_time_3 + consensus_time[3]
                consensus_part_time_4 = consensus_part_time_4 + consensus_time[4]

                if (consensus_count > best_consensus_count ):
                    best_alignment = translation
                    best_consensus_count = consensus_count
                    best_alignment_consensus_vector = consensus_vector

                if (return_consensus_cloud):
                    consensus_cloud[iteration_count, :] = (translation[0],
                                                           translation[1],
                                                           translation[2],
                                                           consensus_count)

                iteration_count = iteration_count + 1

    print ("cloud_consensus Time: " + str (consensus_round_time / cubus_size ))
    print ("cloud_consensus Time Part 1: " + str (consensus_part_time_1 / cubus_size ))
    print ("cloud_consensus Time Part 2: " + str (consensus_part_time_2 / cubus_size ))
    print ("cloud_consensus Time Part 3: " + str (consensus_part_time_3 / cubus_size ))
    print ("cloud_consensus Time Part 4: " + str (consensus_part_time_4 / cubus_size ))
    print ("Overall Time: " + str (time.time () - start_time ))

    if (return_consensus_cloud ):
        return best_alignment, best_consensus_count, best_alignment_consensus_vector, consensus_cloud
    else:
        return best_alignment, best_consensus_count, best_alignment_consensus_vector


def show_consensus_cube (consensus_cube ):

    # save the highest and lowest value to display the whole cubus
    lowest_row = consensus_cube[0, :].reshape (1, 4)
    highest_row = consensus_cube[-1, :].reshape (1, 4)

    # thin out the cloud by removing the
    consensus_cube = consensus_cube[consensus_cube[:, 3] > 0.01].copy ()

    # re-append highest and lowest lowest_row
    consensus_cube = np.concatenate ((lowest_row, consensus_cube, highest_row), axis=0 )

    # normalize consensus row
    normalized_consensus_counts = consensus_cube[:, 3] / np.max(consensus_cube[:, 3])

    #print ("normalized_consensus_counts: " + str (normalized_consensus_counts ))

    # # color the plot
    rgba_colors = np.zeros((normalized_consensus_counts.size, 4 ))

    #print ("consensus_cube:\n" + str (consensus_cube ))

    # fill the colors
    rgba_colors[:, 0] = 0.1
    rgba_colors[:, 1] = normalized_consensus_counts
    rgba_colors[:, 2] = 0.1

    # the fourth column are the alpha values
    rgba_colors[:, 3] = normalized_consensus_counts

    # # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(consensus_cube[:, 0], consensus_cube[:, 1], consensus_cube[:, 2], c=rgba_colors, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


# small cloud
numpy_cloud = np.random.uniform (-1000, 1000, (100, 3 ))

print ('numpy_cloud.shape: ' + str (numpy_cloud.shape ))

random1 = np.random.uniform ((-1, -1, -1), (1, 1, 1))
print ("random1: " + str(random1 ))
corresponding_cloud = numpy_cloud + random1

# # big cloud
# missing_building_als = input_output.load_ascii_file (
#     'clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc' )
#
# missing_building_dim = input_output.load_ascii_file (
#     'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc' )

best_alignment, best_consensus_count, best_alignment_consensus_vector, consensus_cloud = \
    cubic_cloud_consensus (numpy_cloud,
                           corresponding_cloud,
                           threshold=.1,
                           cubus_length=2,
                           step=.05,
                           return_consensus_cloud=True )
print ("best_alignment: " + str(best_alignment ))
print ("best_consensus_count: " + str(best_consensus_count ))
#print ("best_alignment_consensus_vector: " + str(best_alignment_consensus_vector ))
show_consensus_cube (consensus_cloud )


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
