"""
Contains a straightforward implementation of the Consensus Algorithm in three Variants. Distance, normal vector angle
and a combination of these two.
cubic_cloud_consensus algorithm, distance: Translates corresponding_cloud in lenghts of step inside a cubus-shaped space and,
for every step, checks how many points of cloud numpy_cloud have a neighbor within threshold range in corresponding_cloud.
"""

# local modules
from modules import input_output

# basic imports
import numpy as np
import math

# advanced functionality
import scipy.spatial

# plot imports
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Rectangle    # dummy for legend
import matplotlib.pyplot as plt
import textwrap.wrap

# debug
import time


def einsum_angle_between (vector_array_1, vector_array_2 ):

    # diagonal of dot product
    diag = np.clip (np.einsum('ij,ij->i', vector_array_1, vector_array_2 ), -1, 1 )

    return np.arccos (diag )


def get_normal_differences (normals_numpy_cloud, normals_corresponding_cloud ):
    '''
    Computes distances between the vectors of two arrays. Set second array to none to compute magnitudes instead.
    '''

    # check that you got an equal number of normal vectors from each cloud
    if (normals_numpy_cloud.shape[0] != normals_corresponding_cloud.shape[0]):
        raise ValueError ("Shapes do not match: "
                          + str (normals_numpy_cloud.shape[0] )
                          + " / " + str(normals_corresponding_cloud.shape[0] ))

    # fast process: compute angles with einsum
    return einsum_angle_between (normals_numpy_cloud, normals_corresponding_cloud )


def combined_cloud_consensus (tree_of_np_pointcloud, np_pointcloud,
                              corresponding_pointcloud, translation,
                              angle_threshold, distance_threshold):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        tree_of_np_pointcloud (sklearn.neighbors.kd_tree): A kd tree of the reference_cloud
        np_pointcloud (NumpyPointCloud):            NumpyPointCloud object containing a numpy array and it's data labels
        corresponding_pointcloud (NumpyPointCloud): This cloud will translated to match np_pointcloud
        angle_threshold (float):
        distance_threshold (float):

    Output:
        consensus_count (int):                  Number of points with neighbors in suitable range
        consensus_vector ([n, 1] np.array):     Contains 1 if the point had a neighbor in threshold range, else 0
    '''

    # translate the corresponding_pointcloud and query the tree, but only take the x,y,z fields into consideration
    dists, indices = tree_of_np_pointcloud.query (
        corresponding_pointcloud.get_xyz_coordinates () + translation, k=1 )

    # compute the normal vector differences for the matched points
    angle_differences = get_normal_differences (np_pointcloud.get_normals ()[indices, :],
                                                corresponding_pointcloud.get_normals ())
    consensus_vector = np.array ([1 if (distance < distance_threshold and angle < angle_threshold) else 0
                        for (distance, angle) in zip (dists, angle_differences)])

    return np.sum(consensus_vector ), consensus_vector.reshape (-1, 1 )


# refactor: rename tree to tree_of_numpy_cloud
def normal_vector_cloud_consensus (tree_of_np_pointcloud, np_pointcloud,
                                   corresponding_pointcloud, translation,
                                   threshold ):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        tree_of_np_pointcloud (sklearn.neighbors.kd_tree):           A kd tree of np_pointcloud
        np_pointcloud (NumpyPointCloud):            NumpyPointCloud object containing a numpy array and it's data labels
        corresponding_pointcloud (NumpyPointCloud): This cloud will translated to match np_pointcloud
        threshold (float):

    Output:
        consensus_count (int):                  Number of points with neighbors in suitable range
        consensus_vector ([n, 1] np.array):     Contains 1 if the point had a neighbor in threshold range, else 0
    '''

    # translate the corresponding_pointcloud and query the tree, but only take the x,y,z fields into consideration
    dists, indices = tree_of_np_pointcloud.query (
        corresponding_pointcloud.get_xyz_coordinates () + translation, k=1 )

    # compute the normal vector differences for the matched points
    angle_differences = get_normal_differences (np_pointcloud.get_normals ()[indices, :],
                                                corresponding_pointcloud.get_normals ())

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


def point_distance_cloud_consensus (tree_of_reference_cloud, corresponding_pointcloud, translation, threshold ):
    '''
    Counts points of numpy_cloud that have a neighbor of smaller distance than threshold in the corresponding cloud.

    Input:
        tree (sklearn.neighbors.kd_tree): A kd tree of the reference_cloud
        corresponding_pointcloud (NumpyPointCloud): This cloud will translated to match reference_cloud (tree)
        threshold (float):

    Output:
        consensus_count (int):                  Number of points with neighbors in suitable range
        consensus_vector ([n, 1] np.array):     Contains 1 if the point had a neighbor in threshold range, else 0
    '''

    # query the kdtree of the reference cloud
    distances, indices = tree_of_reference_cloud.query (
        corresponding_pointcloud.get_xyz_coordinates () + translation, k=1 )

    # check each result for the distance threshold
    consensus_vector = np.where (distances < threshold, 1, 0)

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

    maximum_consensus = (np.max (consensus_cube[:, 3] / corresponding_cloud_size )) * 100

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

    original_cube = consensus_cube.copy ()

    # # thin out the cloud by keeping the best 500 results
    # sort by best consensus and remove the first values
    index = -math.floor (500 )

    # sort the 4th row, containing the consensus values, best values last
    consensus_cube.view('i8,i8,i8,i8').sort(order=['f3'], axis=0 )

    # filter the values
    consensus_cube = consensus_cube[index:, :]

    # cut out the row containing the best alignment and put it to the end of the cube, so that the best alignment and
    # the last row will be the same
    best_alignment_query = (consensus_cube[:, :3] == best_alignment).all(axis=1).nonzero()
    if (len (best_alignment_query[0] ) > 0 ):
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
    matplotlib_figure_object = plt.figure (num=None, figsize=(7.0, 5.4), dpi=220 )
    axes = matplotlib_figure_object.add_subplot(111, projection=Axes3D.name )

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
    plt.title ("\n".join(textwrap.wrap(plot_title)), loc='right' )

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
                ("Maximum Consensus: " + "{:.2f} %".format(maximum_consensus ), best_alignment_string ),
                loc=(-0.28, 0) )

    # a call to show () halts code execution. So if you want to make multiple consensus experiments, better call draw ()
    # here and call show () later, if you need to see the plots
    #plt.show()
    #plt.ion ()
    #matplotlib_figure_object = plt.show ()
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


def cubic_cloud_consensus (np_pointcloud, corresponding_pointcloud,
                           cubus_length, step, distance_threshold=0.3, angle_threshold=30,
                           algorithmus='distance',
                           display_plot=True, relative_color_scale=False,
                           plot_title="ConsensusCube (TM)", save_plot=False ):
    '''
    Translates corresponding_cloud in lenghts of step inside a cubus-shaped space and, for every step, checks how many points
    of cloud np_pointcloud have a neighbor within threshold range in corresponding_cloud.

    Input:
        np_pointcloud (NumpyPointCloud):    NumpyPointCloud object containing a numpy array and it's data labels
        corresponding_cloud (NumpyPointCloud):   This cloud will be aligned to match np_pointcloud
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

    # variables
    steps_number = math.ceil (cubus_length / step + 1 )
    cubus_size = steps_number**3
    #print ("cubus_size: " + str (steps_number**3))
    consensus_cloud = np.zeros ((cubus_size, 4 ))     # empty cloud that will take the shape of the cubus
    best_alignment = (0, 0, 0)
    best_alignment_consensus_vector = np.zeros ((np_pointcloud.points.shape[0], 1) )     # consens status of points
    best_consensus_count = 0  #
    angle_threshold_radians = 0 if angle_threshold is None else angle_threshold * (np.pi/180)

    # build a kd tree
    # but only take the x,y,z fields into consideration
    scipy_kdtree = scipy.spatial.cKDTree (np_pointcloud.get_xyz_coordinates ())

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

                # Create a list that is as long as corresponding_cloud has fields, so that addition will work and only the
                # first three fields x, y and z are modified
                translation = [x_iterator * step,
                               y_iterator * step,
                               z_iterator * step]
                #translation = translation + [0] * (corresponding_pointcloud.points.shape[1] - 3)

                # Start the computation of the consensus for this translation, using the specified algorithm
                if (algorithmus == 'distance'):

                    consensus_count, consensus_vector = \
                        point_distance_cloud_consensus (scipy_kdtree,
                                                       corresponding_pointcloud,
                                                       translation,
                                                       distance_threshold )
                elif (algorithmus == 'angle'):

                    consensus_count, consensus_vector = \
                        normal_vector_cloud_consensus (scipy_kdtree,
                                                       np_pointcloud,
                                                       corresponding_pointcloud,
                                                       translation,
                                                       angle_threshold_radians )

                else:

                    algorithmus = 'combined'
                    consensus_count, consensus_vector = \
                        combined_cloud_consensus (scipy_kdtree,
                                                  np_pointcloud,
                                                  corresponding_pointcloud,
                                                  translation,
                                                  angle_threshold=angle_threshold_radians,
                                                  distance_threshold=distance_threshold )

                # check for a new consensus high
                if (consensus_count > best_consensus_count ):
                    #best_alignment = [element * -1 for element in translation[0:3]]  # don't know why this is inverted
                    best_alignment = translation
                    best_consensus_count = consensus_count
                    best_alignment_consensus_vector = consensus_vector

                # prepare the ConsensusCube as a numpy cloud with a fourth field that specifies the consensus count
                if (display_plot or save_plot ):
                    consensus_cloud[iteration_count, :] = (translation[0],
                                                           translation[1],
                                                           translation[2],
                                                           consensus_count)

                iteration_count = iteration_count + 1

    print ("Overall Time: " + str (time.time () - start_time ))

    print ("\nbest_alignment: " + str(best_alignment ))
    print ("best_consensus_count: " + str(best_consensus_count ))

    if (display_plot or save_plot ):
        # put together the plot tile, including the string given as argument to this function and the other algorithmus
        # parameters
        plot_title = create_plot_title (
            plot_title, algorithmus, cubus_length, step, distance_threshold, angle_threshold )

        # display the plot
        display_cube, figure = display_consensus_cube (
                        consensus_cloud, corresponding_pointcloud.points.shape[0], best_alignment,
                        plot_title, relative_color_scale=relative_color_scale )

        if (save_plot ):
            # save plot image
            figure.savefig (str("docs/logs/unordered_cube_savefiles/" + plot_title + ".png" ),
                            format='png',
                            dpi=220,
                            bbox_inches='tight')

            # save numpy array of cube
            np.save (str("docs/logs/unordered_cube_savefiles/" + plot_title ),
                display_cube, allow_pickle=False )

            # save ascii cloud of cube
            input_output.save_ascii_file (display_cube, ["X", "Y", "Z", "Consensus"],
                str("docs/logs/unordered_cube_savefiles/" + plot_title + ".asc"))

    return best_alignment, best_consensus_count, best_alignment_consensus_vector
