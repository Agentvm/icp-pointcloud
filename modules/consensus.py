from modules import normals
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sklearn.neighbors    # kdtree
import itertools            # speed improvement when making a [list] out of a [list of [lists]]
#import input_output
#import conversions
#from modules import conversions
import time


# def vector_array_distance (xyz_array, compared_xyz_array=None ):
#     '''
#     Computes distances between the vectors of two arrays. Set second array to none to compute magnitudes instead.
#     '''
#
#     if (compared_xyz_array is None):
#         compared_xyz_array = np.zeros((xyz_array.shape[0], 3 ))
#
#     xyz_array = xyz_array[:, 0:3]
#     compared_xyz_array = compared_xyz_array[:, 0:3]
#
#     # The actual process
#     output = np.sqrt(np.sum((xyz_array - compared_xyz_array )**2, axis=1))
#
#     return output.reshape ((xyz_array.shape[0], 1 ))


def angle_between(vector_1, vector_2):
    """ Returns the angle in radians between vectors 'vector_1' and 'vector_2' """

    if (vector_1 is None or vector_2 is None or None in vector_1 or None in vector_2 ):
        return None

    vector_1 = normals.normalize_vector (vector_1 )
    vector_2 = normals.normalize_vector (vector_2 )

    return np.arccos(np.clip(np.dot(vector_1, vector_2), -1.0, 1.0))


def get_normals (numpy_cloud, field_labels_list ):

    # remove any spaces around the labels
    field_labels_list = [label.strip () for label in field_labels_list]

    if ('Nx' in field_labels_list
       and 'Ny' in field_labels_list
       and 'Nz' in field_labels_list ):
        indices = []
        indices.append (field_labels_list.index('Nz' ))
        indices.append (field_labels_list.index('Ny' ))
        indices.append (field_labels_list.index('Nx' ))
    else:
        raise ValueError ("This Cloud is missing one of the required fields: 'Nx', 'Ny', 'Nz'. Compute Normals first.")

    return numpy_cloud[:, indices]


def get_normal_differences (numpy_cloud, numpy_cloud_field_labels,
                            corresponding_cloud, corresponding_cloud_field_labels ):
    '''
    Computes distances between the vectors of two arrays. Set second array to none to compute magnitudes instead.
    '''

    if (numpy_cloud.shape[0] != corresponding_cloud.shape[0]):
        print ("Shapes do not match: " + str(numpy_cloud.shape[0] ) + " / " + str(corresponding_cloud.shape[0] ))

    # select the normals:
    # //X Y Z Number_of_Returns Return_Number Point_Source_ID Classification Nx Ny Nz Sigma
    # //X Y Z Rf Gf Bf Classification Nx Ny Nz Sigma
    normals_numpy_cloud = get_normals (numpy_cloud, numpy_cloud_field_labels )
    normals_corresponding_cloud = get_normals (corresponding_cloud, corresponding_cloud_field_labels )

    results = numpy_cloud.shape[0] * [None]
    for index, (vec1, vec2) in enumerate(zip (normals_numpy_cloud, normals_corresponding_cloud )):
        results[index] = (angle_between (vec1, vec2 ) )

    return results


def combined_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                              corresponding_cloud, corresponding_cloud_field_labels,
                              angle_threshold, distance_threshold):

    # timing DEBUG
    start_time = time.time ()

    # make a tree an get a list of distances to the nearest neigbor and his index
    # but only take the x,y,z fields into consideration (reference_cloud[:, 0:3])
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud[:, 0:3], leaf_size=40, metric='euclidean' )

    part_time_1 = time.time ()
    # query the three, but only take the x,y,z fields into consideration (compared_cloud[:, 0:3])
    output = tree.query (corresponding_cloud[:, 0:3], k=1, return_distance=True )
    part_time_1 = time.time () - part_time_1

    part_time_2 = time.time ()
    # Make a list out of the values of the respective numpy array
    distances = list(itertools.chain(*output[0] ))
    neighbor_indices = list(itertools.chain(*output[1] ))
    part_time_2 = time.time () - part_time_2

    part_time_3 = time.time ()
    angle_differences = get_normal_differences (numpy_cloud[neighbor_indices, :], numpy_cloud_field_labels,
                                                corresponding_cloud, corresponding_cloud_field_labels)
    part_time_3 = time.time () - part_time_3

    part_time_4 = time.time ()
    consensus_vector = [1 if (distance < distance_threshold and angle < angle_threshold) else 0
                        for (distance, angle) in zip (distances, angle_differences)]
    part_time_4 = time.time () - part_time_4

    return np.sum(consensus_vector ), consensus_vector, (time.time () - start_time, part_time_1, part_time_2, part_time_3, part_time_4)


def normal_vector_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                                   corresponding_cloud, corresponding_cloud_field_labels,
                                   threshold ):
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
    part_time_1 = time.time () - part_time_1

    part_time_2 = time.time ()
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')
    part_time_2 = time.time () - part_time_2

    part_time_3 = time.time ()
    # get distances to nearest neighbor
    correlations = list(itertools.chain(*tree.query (corresponding_cloud, k=1, return_distance=False )))
    angle_differences = get_normal_differences (numpy_cloud[correlations, :], numpy_cloud_field_labels,
                                                corresponding_cloud, corresponding_cloud_field_labels)

    # print ("correlations.shape:\n" + str(len(correlations )))

    part_time_3 = time.time () - part_time_3

    part_time_4 = time.time ()
    consensus_vector = np.array([1 if (angle_difference < threshold) else 0 for angle_difference in angle_differences ])
    part_time_4 = time.time () - part_time_4

    return np.sum(consensus_vector), consensus_vector, (time.time () - start_time, part_time_1, part_time_2, part_time_3, part_time_4)


def point_distance_cloud_consensus (numpy_cloud, corresponding_cloud, threshold ):
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
    tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud, leaf_size=40, metric='euclidean')
    part_time_2 = time.time () - part_time_2

    part_time_3 = time.time ()
    list_consensus_counts = tree.query_radius (corresponding_cloud, threshold, return_distance=False, count_only=True)
    part_time_3 = time.time () - part_time_3

    part_time_4 = time.time ()
    consensus_vector = np.array([1 if count > 0 else 0 for count in list_consensus_counts ])
    part_time_4 = time.time () - part_time_4

    return np.sum(consensus_vector), consensus_vector, (time.time () - start_time, part_time_1, part_time_2, part_time_3, part_time_4)


def display_consensus_cube (consensus_cube, corresponding_cloud_size, plot_title="ConsensusCube (TM)",
                            relative_color_scale=True ):

    # normalize consensus row
    consensus_cube[:, 3] = consensus_cube[:, 3] / corresponding_cloud_size

    # save the highest and lowest value to display the whole cubus
    lowest_row = consensus_cube[0, :].reshape (1, 4)
    highest_row = consensus_cube[-1, :].reshape (1, 4)

    # # thin out the cloud by removing the lowest 60% of results
    # sort by best consensus and remove the last
    index = -math.floor (0.1 * consensus_cube.shape[0] )

    consensus_cube.view('i8,i8,i8,i8').sort(order=['f3'], axis=0 )      # .flip (axis=0 )
    # print ("sorted consensus_cube:\n" + str(consensus_cube) )
    # print ("consensus_cube.shape[0]: " + str(consensus_cube.shape[0]))
    # print ("index: " + str(index))
    consensus_cube = consensus_cube[index:-1, :]
    # print ("CUT consensus_cube:\n" + str(consensus_cube) )
    #consensus_cube = consensus_cube[consensus_cube[:, 3] > 0.01]

    # re-append highest and lowest lowest_row
    consensus_cube = np.concatenate ((lowest_row, consensus_cube, highest_row), axis=0 )

    #print ("normalized_consensus_counts: " + str (normalized_consensus_counts ))

    # # color the plot
    rgba_colors = np.zeros((consensus_cube[:, 3].shape[0], 4 ))

    #print ("consensus_cube:\n" + str (consensus_cube ))

    if (relative_color_scale):
        consensus_cube[:, 3] = consensus_cube[:, 3] / np.max (consensus_cube[:, 3] )

    # fill the colors
    rgba_colors[:, 0] = 1 - consensus_cube[:, 3]
    rgba_colors[:, 1] = consensus_cube[:, 3]
    rgba_colors[:, 2] = 0.2

    # the fourth column are the alpha values
    rgba_colors[:, 3] = np.ones ((consensus_cube.shape[0]))     # consensus_cube[:, 3]
    rgba_colors[0, 3] = 0
    rgba_colors[-1, 3] = 0

    # # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(consensus_cube[:, 0], consensus_cube[:, 1], consensus_cube[:, 2], c=rgba_colors, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if (plot_title is None ):
        plot_title = "ConsensusCube (TM)"
    plt.title (plot_title)
    #plt.show()
    #plt.ion ()
    plt.draw ()


def cubic_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                           compared_cloud, compared_cloud_field_labels,
                           cubus_length, step, distance_threshold=0.009, angle_threshold=0.5,
                           algorithmus='distance',
                           display_plot=True, plot_title=None ):
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

    print ("\nStarting Cubic Cloud Consensus" )
    print ("distance_threshold: " + str(distance_threshold ))
    print ("angle_threshold: " + str(angle_threshold ))
    print ("cubus_length: " + str(cubus_length ))
    print ("step: " + str(step ) + '\n' )

    consensus_round_time = 0
    consensus_part_time_1 = 0
    consensus_part_time_2 = 0
    consensus_part_time_3 = 0
    consensus_part_time_4 = 0
    start_time = time.time ()

    numpy_cloud = numpy_cloud
    compared_cloud = compared_cloud

    # variables
    steps_number = math.ceil (cubus_length / step + 1 )
    cubus_size = steps_number**3
    #print ("cubus_size: " + str (steps_number**3))
    consensus_cloud = np.zeros ((cubus_size, 4 ))     # empty cloud that will take the shape of the cubus
    best_alignment = (0, 0, 0)
    best_alignment_consensus_vector = np.zeros ((numpy_cloud.shape[0], 1) )     # field that shows which points consent
    best_consensus_count = 0  #

    # in the complete space of cubus_length * cubus_length * cubus_length, iterate with the interval of step in x, y and
    # z direction
    iteration_count = 0
    min = -math.floor (steps_number / 2)
    max = math.ceil (steps_number / 2 )
    for x_iterator in range (min, max ):
        for y_iterator in range (min, max ):
            for z_iterator in range (min, max ):

                if (iteration_count % int(cubus_size / 10) == 0):
                    print ("Progress: " + "{:.1f}".format ((iteration_count / cubus_size) * 100.0 ) + " %" )

                # create a list that is as long as compared_cloud has fields, so that addition will work and only the
                # first three fields x, y and z are modified
                translation = [x_iterator * step,
                               y_iterator * step,
                               z_iterator * step]
                translation = translation + [0] * (compared_cloud.shape[1] - 3)

                if (algorithmus == 'distance'):

                    consensus_count, consensus_vector, consensus_time = \
                        point_distance_cloud_consensus (numpy_cloud,
                                                       compared_cloud + translation,
                                                       distance_threshold )
                elif (algorithmus == 'vector'):

                    # find consenting points in the translated compared_cloud
                    consensus_count, consensus_vector, consensus_time = \
                        normal_vector_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                                                       compared_cloud + translation, compared_cloud_field_labels,
                                                       angle_threshold )

                else:
                    # combined
                    consensus_count, consensus_vector, consensus_time = \
                        combined_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                                                  compared_cloud + translation, compared_cloud_field_labels,
                                                  angle_threshold=angle_threshold,
                                                  distance_threshold=distance_threshold )

                # timing
                consensus_round_time = consensus_round_time + consensus_time[0]
                consensus_part_time_1 = consensus_part_time_1 + consensus_time[1]
                consensus_part_time_2 = consensus_part_time_2 + consensus_time[2]
                consensus_part_time_3 = consensus_part_time_3 + consensus_time[3]
                consensus_part_time_4 = consensus_part_time_4 + consensus_time[4]

                # check for a new consensus high
                if (consensus_count > best_consensus_count ):
                    best_alignment = [element * -1 for element in translation[0:3]]  # don't know why this is inverted
                    best_consensus_count = consensus_count
                    best_alignment_consensus_vector = consensus_vector

                # prepare the ConsensusCube as a numpy cloud with a fourth field that specifies the consensus count
                if (display_plot):
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
    print ("Overall Time - Consensus Time: " + str (time.time () - start_time - consensus_round_time))
    print ("Overall Time: " + str (time.time () - start_time ))

    print ("\nbest_alignment: " + str(best_alignment ))
    print ("best_consensus_count: " + str(best_consensus_count ))

    if (display_plot):
        display_consensus_cube (consensus_cloud, compared_cloud.shape[0], plot_title )

    return best_alignment, best_consensus_count, best_alignment_consensus_vector
