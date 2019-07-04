from modules import normals
from modules import input_output
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Rectangle    # dummy for legend
import matplotlib.pyplot as plt
import sklearn.neighbors    # kdtree
import itertools            # speed improvement when making a [list] out of a [list of [lists]]
#import input_output
#import conversions
#from modules import conversions
import time
from textwrap import wrap


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


# def angle_between (vector_1, vector_2):
#     """ Returns the angle in radians between vectors 'vector_1' and 'vector_2' """
#
#     if (vector_1 is None or vector_2 is None or None in vector_1 or None in vector_2 ):
#         return None
#
#     # no normalization required
#     # time1 = time.time ()
#     # vector_1 = normals.normalize_vector (vector_1 )
#     # vector_2 = normals.normalize_vector (vector_2 )
#     # print ("time1: " + str(time.time () - time1))
#
#     #time2 = time.time ()
#     res = np.arccos(np.clip(np.dot(vector_1, vector_2), -1.0, 1.0))
#     #print ("time2: " + str(time.time () - time2))
#
#     return res


def alternative_angle_between (vector_array_1, vector_array_2, step=58 ):

    # prepare results vector with lenght of number of points
    results = np.zeros ((vector_array_1.shape[0], 1 ))

    # np.dot (vector_array_1[i:], vector_array_2.T) computes a gigantic matrix. In order to save RAM space, it has to
    # be done in batches
    for i in range (0, vector_array_1.shape[0], step ):
        if (i + step > vector_array_1.shape[0]):
            results[i:] = np.arccos (
                           np.diagonal (
                            np.clip (
                             np.dot (vector_array_1[i:, :],
                                     vector_array_2[i:, :].T ), -1, 1 ))).reshape (-1, 1)
        else:
            results[i:i+step] = np.arccos (
                                 np.diagonal (
                                  np.clip (
                                   np.dot (vector_array_1[i:i+step, :],
                                           vector_array_2[i:i+step, :].T ), -1, 1 ))).reshape (-1, 1)

    return results


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

    # select the normal fiels of the numpy clouds
    normals_numpy_cloud = get_normals (numpy_cloud, numpy_cloud_field_labels )
    normals_corresponding_cloud = get_normals (corresponding_cloud, corresponding_cloud_field_labels )

    # slow looped process: compute angle by angle
    # results_loop = numpy_cloud.shape[0] * [None]
    # for index, (vec1, vec2) in enumerate(zip (normals_numpy_cloud, normals_corresponding_cloud )):
    #     results_loop[index] = (angle_between (vec1, vec2 ) )

    # fast process: compute angles in multiple batches
    results = alternative_angle_between (normals_numpy_cloud, normals_corresponding_cloud )

    return results


def combined_cloud_consensus (tree, numpy_cloud, numpy_cloud_field_labels,
                              corresponding_cloud, corresponding_cloud_field_labels,
                              angle_threshold, distance_threshold):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        tree (sklearn.neighbors.kd_tree): A kd tree of the reference_cloud
        numpy_cloud ([n, 3] np.array):
        corresponding_cloud ([1, 3] np.array):
        angle_threshold (float):
        distance_threshold (float):

    Output:
        consensus_count (int):                  Number of points with neighbors in suitable range
        consensus_vector ([n, 1] np.array):     Contains 1 if the point had a neighbor in threshold range, else 0
    '''

    # timing DEBUG
    start_time = time.time ()

    part_time_1 = time.time ()
    # query the three, but only take the x,y,z fields into consideration (corresponding_cloud[:, 0:3])
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


# refactor: rename tree to tree_of_numpy_cloud
def normal_vector_cloud_consensus (tree, numpy_cloud, numpy_cloud_field_labels,
                                   corresponding_cloud, corresponding_cloud_field_labels,
                                   threshold ):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        tree (sklearn.neighbors.kd_tree): A kd tree of the reference_cloud
        numpy_cloud ([n, 3] np.array):
        corresponding_cloud ([1, 3] np.array):
        threshold (float):

    Output:
        consensus_count (int):                  Number of points with neighbors in suitable range
        consensus_vector ([n, 1] np.array):     Contains 1 if the point had a neighbor in threshold range, else 0
    '''
    start_time = time.time ()

    part_time_1 = time.time ()
    results = tree.query (corresponding_cloud[:, 0:3], k=1, return_distance=False )
    part_time_1 = time.time () - part_time_1

    part_time_2 = time.time ()
    # get distances to nearest neighbor
    correlations = list(itertools.chain(*results))
    part_time_2 = time.time () - part_time_2

    part_time_3 = time.time ()
    angle_differences = get_normal_differences (numpy_cloud[correlations, :], numpy_cloud_field_labels,
                                                corresponding_cloud, corresponding_cloud_field_labels)
    part_time_3 = time.time () - part_time_3

    part_time_4 = time.time ()
    #consensus_vector_loop = np.array([1 if (angle_difference < threshold) else 0 for angle_difference in angle_differences_loop ])
    consensus_vector = np.where(angle_differences < threshold, 1, 0)
    part_time_4 = time.time () - part_time_4

    return np.sum(consensus_vector), consensus_vector, (time.time () - start_time, part_time_1, part_time_2, part_time_3, part_time_4)


def point_distance_cloud_consensus (tree, numpy_cloud, corresponding_cloud, threshold ):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        tree (sklearn.neighbors.kd_tree): A kd tree of the reference_cloud
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
    part_time_2 = time.time () - part_time_2

    part_time_3 = time.time ()
    list_consensus_counts = tree.query_radius (corresponding_cloud, threshold, return_distance=False, count_only=True)
    part_time_3 = time.time () - part_time_3

    part_time_4 = time.time ()
    consensus_vector = np.array([1 if count > 0 else 0 for count in list_consensus_counts ])
    part_time_4 = time.time () - part_time_4

    return np.sum(consensus_vector), consensus_vector, (time.time () - start_time, part_time_1, part_time_2, part_time_3, part_time_4)


def create_line (point1, point2 ):
    xs = (point1[0], point2[0])
    ys = (point1[1], point2[1])
    zs = (point1[2], point2[2])
    line = Line3D(xs, ys, zs, alpha=0.6, c='blue', ls='--')

    return line


def display_consensus_cube (consensus_cube, corresponding_cloud_size, best_alignment, plot_title="ConsensusCube (TM)",
                            relative_color_scale=True ):
    '''
    Output:
        consensus_cube ((n, 4) np.array):   Display-ready consensus_cube
        matplotlib_figure_object (matplotlib.pyplot): Figure object containing the plot for further use
    '''

    # normalize consensus field
    if (relative_color_scale):
        consensus_cube[:, 3] = consensus_cube[:, 3] / np.max (consensus_cube[:, 3] )
    else:
        consensus_cube[:, 3] = consensus_cube[:, 3] / corresponding_cloud_size

    # get x, y, z min and max values
    xmin = np.min (consensus_cube[:, 0] )
    xmax = np.max (consensus_cube[:, 0] )
    ymin = np.min (consensus_cube[:, 1] )
    ymax = np.max (consensus_cube[:, 1] )
    zmin = np.min (consensus_cube[:, 2] )
    zmax = np.max (consensus_cube[:, 2] )

    # # thin out the cloud by removing the lowest 60% of results
    # sort by best consensus and remove the first 60 % of values
    index = -math.floor (0.1 * consensus_cube.shape[0] )

    # sort the 4th row, containing the consensus values, best values last
    consensus_cube.view('i8,i8,i8,i8').sort(order=['f3'], axis=0 )

    # filter the values
    consensus_cube = consensus_cube[index:, :]

    print (consensus_cube)
    print ((consensus_cube[:, :3] == best_alignment).all(axis=1).nonzero())

    # cut out the row containing the best alignment and put it to the end of the cube, so that the best alignment and
    # the last row will be the same
    if (best_alignment != (0, 0, 0 )):
        best_alignment_index = (consensus_cube[:, :3] == best_alignment).all(axis=1).nonzero()[0][0]
        best_alignment_row = consensus_cube[best_alignment_index, :].reshape (1, -1)
        consensus_cube = np.delete (consensus_cube, best_alignment_index, axis=0)
        consensus_cube = np.concatenate ((consensus_cube, best_alignment_row), axis=0)

    # # prepare color values for the plot
    rgba_colors = np.zeros((consensus_cube[:, 3].shape[0], 4 ))

    # fill the colors
    rgba_colors[:, 0] = 1 - consensus_cube[:, 3]
    rgba_colors[:, 1] = consensus_cube[:, 3]
    rgba_colors[:, 2] = 0.2

    # fill the alpha values
    rgba_colors[:, 3] = np.ones ((consensus_cube.shape[0]))     # consensus_cube[:, 3]

    # # create plot objects
    matplotlib_figure_object = plt.figure(num=None, figsize=(6.5, 5.4), dpi=220 )
    axes = matplotlib_figure_object.add_subplot(111, projection='3d')

    # add all values to the scatter plot
    axes.scatter(consensus_cube[:, 0], consensus_cube[:, 1], consensus_cube[:, 2], c=rgba_colors, marker='o')

    # describe the plots axes
    axes.set_xlabel('X Label')
    axes.set_xlim([xmin, xmax])
    axes.set_ylabel('Y Label')
    axes.set_ylim([ymin, ymax])
    axes.set_zlabel('Z Label')
    axes.set_zlim([zmin, zmax])

    # Fill in the plot tile, wrap it if it is too long and change the subplot borders to fit in the legend as well as
    # the title
    plt.subplots_adjust(left=0.2 )
    if (plot_title is None ):
        plot_title = "ConsensusCube (TM)"
    plt.title ("\n".join(wrap(plot_title)))

    # # create lines to mark the translation result with the best consensus
    # line passing through max consensus point in x-direction
    line = create_line ((xmin, best_alignment[1], best_alignment[2] ),
                        (xmax, best_alignment[1], best_alignment[2] ))
    axes.add_line (line)

    # in y-direction
    axes.add_line (create_line ((best_alignment[0], ymin, best_alignment[2] ),
                                (best_alignment[0], ymax, best_alignment[2] )))

    # in z-direction
    axes.add_line (create_line ((best_alignment[0], best_alignment[1], zmin ),
                                (best_alignment[0], best_alignment[1], zmax )))

    # create a dummy Rectangle to add a string to the legend. The string contains the best alignment (found with the
    # maximum consensus)
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    best_alignment_string = ("{:.4f}".format (best_alignment[0])
                            + ", {:.4f}".format (best_alignment[1])
                            + ", {:.4f}".format (best_alignment[2]))

    # add an explanatory legend to the plot. Mention the lines marking the best consensus and the translation that lead
    # to this result
    plt.legend ([line, extra],
                ("Maximum Consensus: " + "{:.2f} %".format(consensus_cube[-1, 3] * 100), best_alignment_string ),
                loc=(-0.28, 0) )

    # a call to show () halts code execution. So if you want to make multiple consensus experiments, better call draw ()
    # here and call show () later, if you need to see the plots
    #plt.show()
    #plt.ion ()
    plt.draw ()

    return consensus_cube, matplotlib_figure_object


def cubic_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                           compared_cloud, compared_cloud_field_labels,
                           cubus_length, step, distance_threshold=0.009, angle_threshold=0.5,
                           algorithmus='distance',
                           display_plot=True, relative_color_scale=False,
                           plot_title="ConsensusCube (TM)", save_plot=False ):
    '''
    Translates compared_cloud in lenghts of step inside a cubus-shaped space and, for every step, checks how many points
    of cloud numpy_cloud have a neighbor within threshold range in compared_cloud.

    Input:
        numpy_cloud ([n, 3] np.array):
        compared_cloud ([1, 3] np.array):
        threshold (float):                  Threshold that defines the range at which a point is counted as neigbor
        cubus_length (float):               Cubus center is (0, 0, 0). Half of cubus_length is backwards, half forwards.
        step (float):
        algorithmus (string):               'distance', 'angle' or 'combined'

    Output:
        best_alignment ((x, y, z) tuple ):
        best_alignment_consensus_count (int):
        consensus_cube ((n, 4) numpy array):
    '''

    print ("\nStarting " + algorithmus + " Cubic Cloud Consensus" )
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

    # build a kd tree
    # but only take the x,y,z fields into consideration (numpy_cloud[:, 0:3])
    sklearn_neighbors_kd_tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud[:, 0:3], leaf_size=40, metric='euclidean')

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
                        point_distance_cloud_consensus (sklearn_neighbors_kd_tree, numpy_cloud,
                                                       compared_cloud + translation,
                                                       distance_threshold )
                elif (algorithmus == 'angle'):

                    # find consenting points in the translated compared_cloud
                    consensus_count, consensus_vector, consensus_time = \
                        normal_vector_cloud_consensus (sklearn_neighbors_kd_tree, numpy_cloud, numpy_cloud_field_labels,
                                                       compared_cloud + translation, compared_cloud_field_labels,
                                                       angle_threshold )

                else:
                    algorithmus = 'combined'
                    consensus_count, consensus_vector, consensus_time = \
                        combined_cloud_consensus (sklearn_neighbors_kd_tree, numpy_cloud, numpy_cloud_field_labels,
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
                    #best_alignment = [element * -1 for element in translation[0:3]]  # don't know why this is inverted
                    best_alignment = translation[0:3]
                    best_consensus_count = consensus_count
                    best_alignment_consensus_vector = consensus_vector

                # prepare the ConsensusCube as a numpy cloud with a fourth field that specifies the consensus count
                if (display_plot):
                    consensus_cloud[iteration_count, :] = (translation[0],
                                                           translation[1],
                                                           translation[2],
                                                           consensus_count)

                iteration_count = iteration_count + 1

    print ("\ncloud_consensus Time: " + str (consensus_round_time / cubus_size ))
    print ("\tPart 1: " + "{:.2f}".format (((consensus_part_time_1 ) / consensus_round_time )*100) + " %")
    print ("\tPart 2: " + "{:.2f}".format (((consensus_part_time_2 ) / consensus_round_time )*100) + " %")
    print ("\tPart 3: " + "{:.2f}".format (((consensus_part_time_3 ) / consensus_round_time )*100) + " %")
    print ("\tPart 4: " + "{:.2f}".format (((consensus_part_time_4 ) / consensus_round_time )*100) + " %")
    print ("\nOverall Time - Consensus Time: " + str (time.time () - start_time - consensus_round_time))
    print ("Overall Time: " + str (time.time () - start_time ))

    print ("\nbest_alignment: " + str(best_alignment ))
    print ("best_consensus_count: " + str(best_consensus_count ))

    if (display_plot):
        # put together the plot tile, including the string given as argument to this function and the other algorithmus
        # parameters
        plot_title = str(plot_title
                + "_" + str(algorithmus ) + "-consensus"
                + "_cubus_length_" + '{:.1f}'.format (cubus_length )
                + "_step_" + '{:.2f}'.format (step ))
        if (algorithmus == 'distance' or algorithmus == 'combined'):
            plot_title = str(plot_title + "_distance_threshold_" + '{:.3f}'.format (distance_threshold ))
        if (algorithmus == 'angle' or algorithmus == 'combined'):
            plot_title = str(plot_title + "_angle_threshold_" + '{:.3f}'.format (angle_threshold))

        # display the plot
        display_cube, figure = display_consensus_cube (
                        consensus_cloud, compared_cloud.shape[0], best_alignment,
                        plot_title, relative_color_scale=relative_color_scale )

        if (save_plot):
            figure.savefig (str("docs/logs/unordered_cube_savefiles/" + plot_title + ".png" ), format='png', dpi=220, bbox_inches='tight')
            np.save (str("docs/logs/unordered_cube_savefiles/" + plot_title ),
                display_cube, allow_pickle=False )
            input_output.save_ascii_file (display_cube, ["X", "Y", "Z", "Consensus"],
                str("docs/logs/unordered_cube_savefiles/" + plot_title + ".asc"))

    return best_alignment, best_consensus_count, best_alignment_consensus_vector
