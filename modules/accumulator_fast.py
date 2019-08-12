"""
An attempt to make a faster version of the accumulator. Problem: KD-Tree Query yields an array of lists that have
varying lenghts. This resulting array can therefore not be used for fast indexing of points. Instead, points must be

Contains the accumulator algorithm to implement a spheric_cloud_consensus. This can robustly find the best translation
between two different clouds that share important features (like two scenes showing the same place at a different time).
Finds a (X, Y, Z) translation if the correct result is within a sphere of 2 meters with a resolution of 0.1 m.
"""

# local modules
from modules import input_output

# basic imports
import numpy as np
import math
import warnings

# advanced functionality
import scipy.spatial

# plot imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Rectangle    # dummy for creating a legend entry
import textwrap    # for wrapping the plot title

# debug
import time


def create_line (point1, point2 ):
    """Returns a mpl_toolkits.mplot3d.art3d.Line3D object that is a line between the two given points."""
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
                ("Maximum Consensus: " + "{:.2f} %".format(consensus_cube[-1, 3] * 100), best_alignment_string ),
                loc=(-0.28, 0) )

    # a call to show () halts code execution. So if you want to make multiple consensus experiments, better call draw ()
    # here and call show () later, if you need to see the plots
    #plt.show()
    #plt.ion ()
    #matplotlib_figure_object = plt.show ()
    plt.draw ()

    return original_cube, matplotlib_figure_object


def create_closed_grid_II (grid_length, step ):

    # borders
    lmax = grid_length / 2
    lmin = -grid_length / 2

    # create a 3-d cubic grid with (grid_length / step + 1)**3 grid values
    # in the space of (-grid_length / 2) to (grid_length / 2)
    grid = np.transpose (np.reshape(np.mgrid[lmin:lmax+step:step, lmin:lmax+step:step, lmin:lmax+step:step], (3, -1) ))

    # add another row for the consensus values
    grid = np.concatenate ((grid, np.zeros (shape=(grid.shape[0], 1))), axis=1 )

    return grid


def create_closed_grid (grid_length, step ):

    # grid variables
    steps_number = math.ceil (grid_length / step + 1 )
    grid_points_number = steps_number**3

    # make a grid in the style of a pointcloud
    grid = np.zeros ((grid_points_number, 4 ))

    # in intervals of step, create grid nodes
    general_iterator = 0
    minimum = -math.floor (steps_number / 2)
    maximum = math.ceil (steps_number / 2 )
    for x_iterator in range (minimum, maximum ):
        for y_iterator in range (minimum, maximum ):
            for z_iterator in range (minimum, maximum ):

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


def spheric_cloud_consensus (np_pointcloud, corresponding_pointcloud,
                             accumulator_radius=1.2, grid_size=0.1, distance_threshold=0.2, angle_threshold=30,
                             algorithmus='distance',
                             display_plot=True, save_plot=False,
                             relative_color_scale=False,
                             plot_title="ConsensusCube (TM)",
                             batch_size=10000 ):
    '''
    if algorithmus='distance':  Counts how many points of cloud np_pointcloud have a neighbor within threshold range in
                                corresponding_cloud.

    Input:
        np_pointcloud (NumpyPointCloud):    NumpyPointCloud object containing a numpy array and it's data labels
        corresponding_cloud (NumpyPointCloud):   This cloud will be aligned to match np_pointcloud
        distance_threshold (float):         Threshold that defines the range at which a point is counted as neigbor
        angle_threshold (float, degree):    Angle threshold to define maximum deviation of normal vectors
        accumulator_radius (float):         Sphere center is translation (0, 0, 0). Translations are possible in sphere
        grid_size (float):                  Rasterization of results. May yield unsatisfying results if too small
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
    print ("batch_size: " + str(batch_size ) + '\n' )

    if (display_plot and save_plot):
        message = ("Displaying the plot halts code execution until the plot is closed. This can be a problem when "
                  + "multiple computations are queued.")
        warnings.warn (message)

    loop_cloud_query_time = 0
    loop_diff_time = 0
    loop_grid_query_time = 0
    rasterization_time = 0
    start_time = time.time ()
    interim = time.time ()

    # variables
    # field that shows which points consent
    best_alignment_consensus_vector = np.zeros ((np_pointcloud.points.shape[0], 1) )
    # angle_threshold_radians = 0 if angle_threshold is None else angle_threshold * (np.pi/180)

    # build a grid as a kdtree to discretize the results
    consensus_cube = create_closed_grid (accumulator_radius * 2, grid_size )
    grid_kdtree = scipy.spatial.cKDTree (consensus_cube[:, 0:3] )

    # build kdtree and query it for points within radius (radius being the maximum translation that can be detected)
    scipy_kdtree = scipy.spatial.cKDTree (np_pointcloud.get_xyz_coordinates ())

    # print ("cloud_indices: " + str (cloud_indices ))

    init_time = time.time () - interim
    interim_2 = time.time ()

    # simplified process:
    #   for each point in corresponding_pointcloud:
    #       Find neighbors within accumulator_radius
    #       Compute the translations to these neighbors
    #       Rasterize the translations by matching them with the grid
    #       When a translation matches a grid cell, increment the consensus counter of that cell
    iterations = 0
    inner_iterations = 0

    # iterate through the cloud in batches of size 'batch_size'
    for i in range (0, corresponding_pointcloud.shape[0], batch_size ):
        if (i + batch_size > corresponding_pointcloud.shape[0]):
            batch_points = np.astype (corresponding_pointcloud.points[i:, 0:3])
        else:
            batch_points = corresponding_pointcloud.points[i:i+batch_size, 0:3]

        interim = time.time ()

        batch_point_indices = scipy_kdtree.query_ball_point (np.ascontiguousarray(batch_points ), r=accumulator_radius )
        batch_point_indices = np.array (batch_point_indices, dtype=np.int )

        loop_cloud_query_time += time.time () - interim
        interim = time.time ()

        print ("batch_point_indices: " + str (batch_point_indices ))

        # diff all points found near the corresponding points with the corresponding points
        diff_vector_array = np_pointcloud.get_xyz_coordinates ()[batch_point_indices, :] - batch_points

        # print ("\n-----------------------------------------------\n\npoint_indices:\n" + str (indices ))
        # print ("diff_vectors:\n" + str (diff_vectors ))
        loop_diff_time += time.time () - interim
        interim = time.time ()

        # rasterize by finding nearest grid node (representing a translation)
        dists, point_matches = grid_kdtree.query (diff_vector_array, k=1 )
        # print ("dists from gridpoints: " + str (dists.T ))
        # print ("grid point matches: " + str (point_matches.T ))
        loop_grid_query_time += time.time () - interim
        interim = time.time ()

        # apply distance filter to results
        if (distance_threshold is not None and distance_threshold > 0 ):
            point_matches = point_matches[dists < distance_threshold]

        # update the cube with the results of this point, ignore multiple hits
        consensus_cube[np.unique (point_matches ), 3] += 1
        # print ("\nupdated consensus_cube >0:\n" + str (consensus_cube[consensus_cube[:, 3] > 0, :] ))
        rasterization_time += time.time () - interim

        # # iterate through the query results of this batch, processing the result of each point individually (slow)
        # for t, point_neighbor_indices in enumerate (batch_point_indices, 0 ):
        #     iterator = i + t    # batch iterator + in-batch iterator
        #
        #     # Progress Prints every 10 %
        #     if (iterator % int(corresponding_pointcloud.points.shape[0] / 10) == 0 ):
        #         print ("Progress: " + "{:.1f}".format (
        #                                     (iterator / corresponding_pointcloud.points.shape[0]) * 100.0 ) + " %" )
        #
        #     #print ("point_neighbor_indices: " + str (point_neighbor_indices ))
        #
        #     if (len(point_neighbor_indices ) > 0):
        #
        #
        #
        #
        #
        #         inner_iterations += 1
        iterations += 1

    overall_loop_time = time.time () - interim_2
    interim = time.time ()

    best_alignment = consensus_cube[np.argmax (consensus_cube[:, 3] ), 0:3].copy ()
    best_consensus_count = np.max (consensus_cube[:, 3] ).copy ()

    # put together the plot title, including the string given as argument to this function and the other algorithmus
    # parameters
    original_plot_base = plot_title
    plot_title = create_plot_title (
        original_plot_base, algorithmus, accumulator_radius, grid_size, distance_threshold, angle_threshold )

    # create the plot
    display_cube, figure = display_consensus_cube (consensus_cube,
                                                   corresponding_pointcloud.points.shape[0],
                                                   best_alignment,
                                                   plot_title,
                                                   relative_color_scale )

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

    outro_time = time.time () - interim

    print ("batch_size: " + str(batch_size ))

    # print (loop_cloud_query_time)
    # print (loop_diff_time)
    # print (loop_grid_query_time)
    # print (rasterization_time)

    # normalizing
    loop_cloud_query_time /= iterations
    loop_diff_time /= iterations
    loop_grid_query_time /= iterations
    rasterization_time /= iterations

    print ("\nInit Time: " + str (init_time ))
    print ("Overall Loop Time: " + str (overall_loop_time ))
    average_loop_time = overall_loop_time / iterations
    #print ("Average Loop Time: " + str (average_loop_time ))
    print ("\tLoop Cloud Query Time: " + "{:.2f}%".format ((loop_cloud_query_time / average_loop_time) * 100 ))
    print ("\tLoop Diff Time: " + "{:.2f}%".format ((loop_diff_time / average_loop_time) * 100 ))
    print ("\tLoop Grid Query Time: " + "{:.2f}%".format ((loop_grid_query_time / average_loop_time) * 100 ))
    print ("\tLoop Rasterization Time: " + "{:.2f}%".format ((rasterization_time / average_loop_time) * 100 ))
    print ("Outro Time: " + str (outro_time ))
    print ("Overall Time: " + str (time.time () - start_time ) + "\n")

    # display the plot
    if (display_plot ):
         _ = plt.show (figure );
    plt.close ()

    return best_alignment, best_consensus_count, best_alignment_consensus_vector
