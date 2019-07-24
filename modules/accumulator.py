#from modules import normals
from modules import input_output
from modules import conversions
# from modules.normals import normalize_vector_array
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Rectangle    # dummy for legend
import matplotlib.pyplot as plt
import scipy.spatial
#import itertools            # speed improvement when making a [list] out of a [list of [lists]]
#import input_output
#import conversions
#from modules import conversions
import time
from textwrap import wrap

# DEBUG
import pylab
import matplotlib._pylab_helpers


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

    original_cube = consensus_cube.copy ()

    # # thin out the cloud by keeping the best 1000 resulst
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
    plt.title ("\n".join(wrap(plot_title)), loc='right' )

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
    #matplotlib_figure_object = plt.show ()
    plt.draw ()

    return original_cube, matplotlib_figure_object


def create_closed_grid (grid_length, step ):

    # grid variables
    steps_number = math.ceil (grid_length / step + 1 )
    grid_points_number = steps_number**3

    # make a grid in the style of a pointcloud
    grid = np.zeros ((grid_points_number, 4 ))
    print ("grid_points_number: " + str (grid_points_number ))
    print ("grid.shape: " + str (grid.shape ))

    # in intervals of step, create grid nodes
    general_iterator = 0
    min = -math.floor (steps_number / 2)
    max = math.ceil (steps_number / 2 )
    for x_iterator in range (min, max ):
        for y_iterator in range (min, max ):
            for z_iterator in range (min, max ):

                grid[general_iterator, 0:3] = [x_iterator * step,
                                               y_iterator * step,
                                               z_iterator * step]

                general_iterator += 1

    return grid


def create_plot_title (base_title, algorithmus, accumulator_radius, grid_size, distance_threshold, angle_threshold ):
    plot_title = str(base_title
            + "_" + str(algorithmus ) + "-consensus"
            + "_sphere_radius_" + '{:.1f}'.format (accumulator_radius )
            + "_step_" + '{:.2f}'.format (grid_size ))
    if (algorithmus == 'distance' or algorithmus == 'combined'):
        plot_title = str(plot_title + "_distance_threshold_" + '{:.3f}'.format (distance_threshold ))
    if (algorithmus == 'angle' or algorithmus == 'combined'):
        plot_title = str(plot_title + "_angle_threshold_" + '{:.3f}'.format (angle_threshold))

    return plot_title


def spheric_cloud_consensus (numpy_cloud, numpy_cloud_field_labels,
                             compared_cloud, compared_cloud_field_labels,
                             accumulator_radius=1.2, grid_size=0.1, distance_threshold=0.2, angle_threshold=30,
                             algorithmus='distance',
                             display_plot=True, save_plot=False,
                             relative_color_scale=False,
                             plot_title="ConsensusCube (TM)"  ):
    '''
    if algorithmus='distance':  Translates compared_cloud in lenghts of grid_size inside a sphere-shaped space and, for
    every step, checks how many points of cloud numpy_cloud have a neighbor within threshold range in compared_cloud.

    Input:
        numpy_cloud ([n, 3] np.array):
        compared_cloud ([1, 3] np.array):
        distance_threshold (float):         Threshold that defines the range at which a point is counted as neigbor
        angle_threshold (float, degree):    Angle threshold to define maximum deviation of normal vectors
        accumulator_radius (float):         Cubus center is (0, 0, 0). Half of cubus_length is backwards, half forwards.
        step (float):
        algorithmus (string):               'distance', 'angle' or 'combined'

    Output:
        best_alignment ((x, y, z) tuple ):
        best_alignment_consensus_count (int):
        consensus_cube ((n, 4) numpy array):
    '''

    print ("\nStarting " + algorithmus + " Accumulator Consensus" )
    print ("distance_threshold: " + str(distance_threshold ))
    print ("angle_threshold: " + str(angle_threshold ))
    print ("accumulator_radius: " + str(accumulator_radius ))
    print ("grid_size: " + str(grid_size ) + '\n' )

    start_time = time.time ()

    # variables
    best_alignment_consensus_vector = np.zeros ((numpy_cloud.shape[0], 1) )     # field that shows which points consent
    # angle_threshold_radians = 0 if angle_threshold is None else angle_threshold * (np.pi/180)

    # build a grid as a kdtree to discretize the results
    consensus_cube = create_closed_grid (accumulator_radius * 2, grid_size )
    grid_kdtree = scipy.spatial.cKDTree (consensus_cube[:, 0:3] )
    print ("\nconsensus_cube shape: " + str (consensus_cube.shape ))
    #print ("\nconsensus_cube:\n" + str (consensus_cube ))

    # build kdtree and query it for points within radius (radius being the maximum translation that can be detected)
    scipy_kdtree = scipy.spatial.cKDTree (numpy_cloud[:, 0:3] )
    #cloud_indices = scipy_kdtree.query_ball_point (compared_cloud[:, 0:3], accumulator_radius )    # memory problem
    # print ("\ncloud_indices: " + str (cloud_indices ))

    for i, point in enumerate (compared_cloud[:, 0:3] ):

        point_indices = scipy_kdtree.query_ball_point (point, accumulator_radius )

        if (len(point_indices ) > 0):

            # diff all points found near the corresponding point with corresponding point
            diff_vectors = numpy_cloud[point_indices, 0:3] - point
            # print ("\n--------------------------------------------------\n\npoint_indices:\n" + str (point_indices ))
            # print ("diff_vectors:\n" + str (diff_vectors ))

            # rasterize
            dists, point_matches = grid_kdtree.query (diff_vectors, k=1 )
            # print ("dists from gridpoints: " + str (dists.T ))
            # print ("grid point matches: " + str (point_matches.T ))

            # update the cube with the results of this point, ignore multiple hits
            consensus_cube[np.unique (point_matches ), 3] += 1
            # print ("\nupdated consensus_cube >0:\n" + str (consensus_cube[consensus_cube[:, 3] > 0, :] ))

    best_alignment = consensus_cube[np.argmax (consensus_cube[:, 3] ), 0:3].copy ()
    best_consensus_count = np.max (consensus_cube[:, 3] ).copy ()

    # put together the plot title, including the string given as argument to this function and the other algorithmus
    # parameters
    original_plot_base = plot_title
    plot_title = create_plot_title (
        original_plot_base, algorithmus, accumulator_radius, grid_size, distance_threshold, angle_threshold )

    # create the plot
    display_cube, figure = display_consensus_cube (consensus_cube, compared_cloud.shape[0], best_alignment,
                                                   plot_title, relative_color_scale )

    # save the plot for later reference
    if (save_plot ):

        # save image
        figure.savefig (str("docs/logs/unordered_cube_savefiles/" + plot_title + ".png" ),
                        format='png', dpi=220, bbox_inches='tight')

        # save numpy array
        np.save (str("docs/logs/unordered_cube_savefiles/" + plot_title ),
            display_cube, allow_pickle=False )

        # save ascii pointcloud
        input_output.save_ascii_file (display_cube, ["X", "Y", "Z", "Consensus"],
            str("docs/logs/unordered_cube_savefiles/" + plot_title + ".asc"))

    # display the plot
    if (display_plot ):
         nothing = plt.show (figure );
    plt.close ()

    print ("\nOverall Time: " + str (time.time () - start_time ))

    return best_alignment, best_consensus_count, best_alignment_consensus_vector
