#from modules import normals
from modules import input_output
from modules import conversions
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Rectangle    # dummy for legend
import matplotlib.pyplot as plt
#import sklearn.neighbors    # kdtree
import scipy.spatial
#import itertools            # speed improvement when making a [list] out of a [list of [lists]]
#import input_output
#import conversions
#from modules import conversions
import time
from textwrap import wrap


# def alternative_angle_between (vector_array_1, vector_array_2, step=58 ):
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


def einsum_angle_between (vector_array_1, vector_array_2 ):

    # diagonal of dot product
    diag = np.clip (np.einsum('ij,ij->i', vector_array_1, vector_array_2 ), -1, 1 )

    return np.arccos (diag )

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
#         raise ValueError ("This Cloud is missing one of the required fields: 'Nx', 'Ny', 'Nz'. Compute Normals first.")
#
#     return numpy_cloud[:, indices]


def get_normal_differences (numpy_cloud, numpy_cloud_field_labels,
                            corresponding_cloud, corresponding_cloud_field_labels ):
    '''
    Computes distances between the vectors of two arrays. Set second array to none to compute magnitudes instead.
    '''

    if (numpy_cloud.shape[0] != corresponding_cloud.shape[0]):
        print ("Shapes do not match: " + str(numpy_cloud.shape[0] ) + " / " + str(corresponding_cloud.shape[0] ))

    # select the normal fiels of the numpy clouds
    normals_numpy_cloud = conversions.get_fields (numpy_cloud, numpy_cloud_field_labels, ['Nx', 'Ny', 'Nz'] )
    normals_corresponding_cloud = \
        conversions.get_fields (corresponding_cloud, corresponding_cloud_field_labels, ['Nx', 'Ny', 'Nz'] )

    # fast process: compute angles in multiple batches
    results = einsum_angle_between (normals_numpy_cloud, normals_corresponding_cloud )

    return results


def all_in_one_cloud_consensus (tree_of_numpy_cloud, numpy_cloud, numpy_cloud_field_labels,
                                corresponding_cloud, corresponding_cloud_field_labels,
                                angle_threshold, distance_threshold):
    '''
    Like combined_cloud_consensus, but returns the results of distance, angle, and combined consensus
    '''

    # query the three, but only take the x,y,z fields into consideration (corresponding_cloud[:, 0:3])
    dists, indices = tree_of_numpy_cloud.query (corresponding_cloud[:, 0:3], k=1 )

    # compare normal vectors
    angle_differences = get_normal_differences (numpy_cloud[indices, :], numpy_cloud_field_labels,
                                                corresponding_cloud, corresponding_cloud_field_labels)

    # # check the consensus for every point
    consensus_vector_distance = np.where (dists < distance_threshold, 1, 0)
    consensus_vector_angle = np.where(angle_differences < angle_threshold, 1, 0)
    consensus_vector_combined = np.where (consensus_vector_distance + consensus_vector_angle == 2, 1, 0 )

    return np.sum(consensus_vector_distance ), np.sum(consensus_vector_angle ), \
        np.sum(consensus_vector_combined ), consensus_vector_combined.reshape (-1, 1 )




def combined_cloud_consensus (tree_of_numpy_cloud, numpy_cloud, numpy_cloud_field_labels,
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

    # query the three, but only take the x,y,z fields into consideration (corresponding_cloud[:, 0:3])
    dists, indices = tree_of_numpy_cloud.query (corresponding_cloud[:, 0:3], k=1 )

    # compare normal vectors
    angle_differences = get_normal_differences (numpy_cloud[indices, :], numpy_cloud_field_labels,
                                                corresponding_cloud, corresponding_cloud_field_labels)

    # check the consensus for every point
    consensus_vector = np.array ([1 if (distance < distance_threshold and angle < angle_threshold) else 0
                        for (distance, angle) in zip (dists, angle_differences)])

    return np.sum(consensus_vector ), consensus_vector.reshape (-1, 1 )


# refactor: rename tree to tree_of_numpy_cloud
def normal_vector_cloud_consensus (tree_of_numpy_cloud, numpy_cloud, numpy_cloud_field_labels,
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
    #results = tree.query (corresponding_cloud[:, 0:3], k=1, return_distance=False )
    dists, indices = tree_of_numpy_cloud.query (corresponding_cloud[:, 0:3], k=1 )
    # get distances to nearest neighbor
    # correlations = list(itertools.chain(*results))
    angle_differences = get_normal_differences (numpy_cloud[indices, :], numpy_cloud_field_labels,
                                                corresponding_cloud, corresponding_cloud_field_labels)
    #consensus_vector_loop = np.array([1 if (angle_difference < threshold) else 0 for angle_difference in angle_differences_loop ])
    consensus_vector = np.where(angle_differences < threshold, 1, 0)

    return np.sum(consensus_vector), consensus_vector.reshape (-1, 1 )


# def point_distance_cloud_consensus_parallel_wrapper (input):
#     # translation is received as additional argument
#     (tree_of_numpy_cloud, numpy_cloud, corresponding_cloud, translation, distance_threshold ) = input
#
#     # consensus is started with translated corresponding_cloud
#     (consensus_count, consensus_vector) = point_distance_cloud_consensus (
#         tree_of_numpy_cloud, numpy_cloud, corresponding_cloud+translation, distance_threshold )
#
#     # translation is returned alongside the computed values
#     return (consensus_count, consensus_vector, translation)


def point_distance_cloud_consensus (tree_of_numpy_cloud, numpy_cloud, corresponding_cloud, threshold ):
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
    numpy_cloud = numpy_cloud[:, 0:3]
    corresponding_cloud = corresponding_cloud[:, 0:3]
    #list_consensus_counts = tree.query_radius (corresponding_cloud, threshold, return_distance=False, count_only=True)
    #consensus_counts = tree.query_ball_point (np.ascontiguousarray (corresponding_cloud ), r=threshold, return_length=True )
    dists, indices = tree_of_numpy_cloud.query (corresponding_cloud[:, 0:3], k=1 )
    #consensus_vector = np.array ([1 if count > 0 else 0 for count in list_consensus_counts ])
    consensus_vector = np.where (dists < threshold, 1, 0)

    return np.sum(consensus_vector), consensus_vector.reshape (-1, 1 )


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

    original_cube = consensus_cube

    # # thin out the cloud by removing the lowest 60% of results
    # sort by best consensus and remove the first 60 % of values
    index = -math.floor (0.1 * consensus_cube.shape[0] )

    # sort the 4th row, containing the consensus values, best values last
    consensus_cube.view('i8,i8,i8,i8').sort(order=['f3'], axis=0 )

    # filter the values
    consensus_cube = consensus_cube[index:, :]

    # print (consensus_cube)
    # print ("\nalign: " + str (best_alignment))
    # print ((consensus_cube[:, :3] == best_alignment).all(axis=1).nonzero())

    # cut out the row containing the best alignment and put it to the end of the cube, so that the best alignment and
    # the last row will be the same
    # ToDo: This crashes if best alignment is at min or max  or zero coordinates. In these cases,
    # best value and corresponding consensus is not appended to the end of the files saved.
    # But even scarier: why are values like (0, 0, 0) and (0.5, 0.5, 0.5) not found in the results? Are they missing?
    # Explanation: (0.5, 0.5, 0.5) is the max value for cubus_length=1 and step=0.10
    best_alignment_query = (consensus_cube[:, :3] == best_alignment).all(axis=1).nonzero()
    if (len (best_alignment_query[0] ) != 0 ):
        best_alignment_index = best_alignment_query[0][0]
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

    return original_cube, matplotlib_figure_object


def create_plot_title (base_title, algorithmus, cubus_length, step, distance_threshold, angle_threshold ):
    plot_title = str(base_title
            + "_" + str(algorithmus ) + "-consensus"
            + "_cubus_length_" + '{:.1f}'.format (cubus_length )
            + "_step_" + '{:.2f}'.format (step ))
    if (algorithmus == 'distance' or algorithmus == 'combined'):
        plot_title = str(plot_title + "_distance_threshold_" + '{:.3f}'.format (distance_threshold ))
    if (algorithmus == 'angle' or algorithmus == 'combined'):
        plot_title = str(plot_title + "_angle_threshold_" + '{:.3f}'.format (angle_threshold))

    return plot_title


def cubic_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                           compared_cloud, compared_cloud_field_labels,
                           cubus_length, step, distance_threshold=0.3, angle_threshold=30,
                           algorithmus='distance',
                           display_plot=True, relative_color_scale=False,
                           plot_title="ConsensusCube (TM)", save_plot=False ):
    '''
    Translates compared_cloud in lenghts of step inside a cubus-shaped space and, for every step, checks how many points
    of cloud numpy_cloud have a neighbor within threshold range in compared_cloud.

    Input:
        numpy_cloud ([n, 3] np.array):
        compared_cloud ([1, 3] np.array):
        distance_threshold (float):         Threshold that defines the range at which a point is counted as neigbor
        angle_threshold (float, degree):    Angle threshold to define maximum deviation of normal vectors
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
    angle_threshold_radians = 0 if angle_threshold is None else angle_threshold * (np.pi/180)

    # special case to compute three results at once
    all_in_one = False
    if (algorithmus == 'all_in_one' ):
        all_in_one = True
        best_alignment_dist = (0, 0, 0)
        best_alignment_angle = (0, 0, 0)
        best_consensus_count_dist = 0
        best_consensus_count_angle = 0
        consensus_cloud_distance = np.zeros ((cubus_size, 4 ))
        consensus_cloud_angle = np.zeros ((cubus_size, 4 ))

    # build a kd tree
    # but only take the x,y,z fields into consideration (numpy_cloud[:, 0:3])
    #sklearn_neighbors_kd_tree = sklearn.neighbors.kd_tree.KDTree (numpy_cloud[:, 0:3], leaf_size=40, metric='euclidean')
    scipy_kdtree = scipy.spatial.cKDTree (numpy_cloud[:, 0:3] )

    # in the complete space of cubus_length * cubus_length * cubus_length, iterate with the interval of step in x, y and
    # z direction
    iteration_count = 0
    min = -math.floor (steps_number / 2)
    max = math.ceil (steps_number / 2 )
    for x_iterator in range (min, max ):
        for y_iterator in range (min, max ):
            for z_iterator in range (min, max ):

                # Progress Prints every 10 %
                if (iteration_count % int(cubus_size / 10) == 0):
                    print ("Progress: " + "{:.1f}".format ((iteration_count / cubus_size) * 100.0 ) + " %" )

                # Create a list that is as long as compared_cloud has fields, so that addition will work and only the
                # first three fields x, y and z are modified
                translation = [x_iterator * step,
                               y_iterator * step,
                               z_iterator * step]
                translation = translation + [0] * (compared_cloud.shape[1] - 3)

                # Start the computation of the consensus for this translation, using the specified algorithm
                if (algorithmus == 'distance'):

                    consensus_count, consensus_vector = \
                        point_distance_cloud_consensus (scipy_kdtree, numpy_cloud,
                                                       compared_cloud + translation,
                                                       distance_threshold )
                elif (algorithmus == 'angle'):

                    consensus_count, consensus_vector = \
                        normal_vector_cloud_consensus (scipy_kdtree, numpy_cloud, numpy_cloud_field_labels,
                                                       compared_cloud + translation, compared_cloud_field_labels,
                                                       angle_threshold_radians )

                # refactor
                elif (all_in_one):

                    # set this for the naming of the plot additional plots will be saved for distance and angle results
                    algorithmus = 'combined'

                    # in addition to the combined_consensus results (consensus_count, consensus_vector), we get
                    # dist and angle consensus counts, to build two additional consensus_cubes
                    consensus_count_dist, consensus_count_angle, consensus_count, consensus_vector = \
                        all_in_one_cloud_consensus (scipy_kdtree, numpy_cloud, numpy_cloud_field_labels,
                                                    compared_cloud + translation, compared_cloud_field_labels,
                                                    angle_threshold=angle_threshold_radians,
                                                    distance_threshold=distance_threshold )

                    # check for a new consensus high
                    if (consensus_count_dist > best_consensus_count_dist ):   # distance
                        best_alignment_dist = translation[0:3]
                        best_consensus_count_dist = consensus_count_dist

                    if (consensus_count_angle > best_consensus_count_angle ):  # angle
                        best_alignment_angle = translation[0:3]
                        best_consensus_count_angle = consensus_count_angle

                    # if (consensus_count > best_consensus_count ):    # combined
                    #     best_alignment = translation[0:3]
                    #     best_consensus_count = consensus_count
                    #     best_alignment_consensus_vector = consensus_vector

                    # prepare the ConsensusCube as a numpy cloud with a fourth field that specifies the consensus count
                    # prepare two extra cubes for angle and distance
                    consensus_cloud_distance[iteration_count, :] = (translation[0], translation[1], translation[2],
                                                           consensus_count_dist)
                    consensus_cloud_angle[iteration_count, :] = (translation[0], translation[1], translation[2],
                                                           consensus_count_angle)
                    # consensus_cloud[iteration_count, :] = (translation[0], translation[1], translation[2],
                                                           # consensus_count)

                else:

                    algorithmus = 'combined'
                    consensus_count, consensus_vector = \
                        combined_cloud_consensus (scipy_kdtree, numpy_cloud, numpy_cloud_field_labels,
                                                  compared_cloud + translation, compared_cloud_field_labels,
                                                  angle_threshold=angle_threshold_radians,
                                                  distance_threshold=distance_threshold )

                #if (not all_in_one):
                # check for a new consensus high
                if (consensus_count > best_consensus_count ):
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

    print ("Overall Time: " + str (time.time () - start_time ))

    print ("\nbest_alignment: " + str(best_alignment ))
    print ("best_consensus_count: " + str(best_consensus_count ))

    if (display_plot or all_in_one):
        # put together the plot tile, including the string given as argument to this function and the other algorithmus
        # parameters
        original_plot_base = plot_title
        plot_title = create_plot_title (
            plot_title, algorithmus, cubus_length, step, distance_threshold, angle_threshold )

        # display the plot
        display_cube, figure = display_consensus_cube (
                        consensus_cloud, compared_cloud.shape[0], best_alignment,
                        plot_title, relative_color_scale=relative_color_scale )

        if (all_in_one):
            plot_title_dist = create_plot_title (
                original_plot_base, 'distance', cubus_length, step, distance_threshold, angle_threshold )
            plot_title_angle = create_plot_title (
                original_plot_base, 'angle', cubus_length, step, distance_threshold, angle_threshold )

            # display the plot
            display_cube_dist, figure_dist = display_consensus_cube (
                            consensus_cloud_distance, compared_cloud.shape[0], best_alignment_dist,
                            plot_title_dist, relative_color_scale=relative_color_scale )

            # display the plot
            display_cube_angle, figure_angle = display_consensus_cube (
                            consensus_cloud_angle, compared_cloud.shape[0], best_alignment_angle,
                            plot_title_angle, relative_color_scale=relative_color_scale )

        if (save_plot):
            figure.savefig (str("docs/logs/unordered_cube_savefiles/" + plot_title + ".png" ), format='png', dpi=220, bbox_inches='tight')
            np.save (str("docs/logs/unordered_cube_savefiles/" + plot_title ),
                display_cube, allow_pickle=False )
            input_output.save_ascii_file (display_cube, ["X", "Y", "Z", "Consensus"],
                str("docs/logs/unordered_cube_savefiles/" + plot_title + ".asc"))

            if (all_in_one):
                figure_dist.savefig (str("docs/logs/unordered_cube_savefiles/" + plot_title_dist + ".png" ), format='png', dpi=220, bbox_inches='tight')
                np.save (str("docs/logs/unordered_cube_savefiles/" + plot_title_dist ),
                    display_cube_dist, allow_pickle=False )
                input_output.save_ascii_file (display_cube_dist, ["X", "Y", "Z", "Consensus"],
                    str("docs/logs/unordered_cube_savefiles/" + plot_title_dist + ".asc"))

                figure_angle.savefig (str("docs/logs/unordered_cube_savefiles/" + plot_title_angle + ".png" ), format='png', dpi=220, bbox_inches='tight')
                np.save (str("docs/logs/unordered_cube_savefiles/" + plot_title_angle ),
                    display_cube_angle, allow_pickle=False )
                input_output.save_ascii_file (display_cube_angle, ["X", "Y", "Z", "Consensus"],
                    str("docs/logs/unordered_cube_savefiles/" + plot_title_angle + ".asc"))

    return best_alignment, best_consensus_count, best_alignment_consensus_vector
