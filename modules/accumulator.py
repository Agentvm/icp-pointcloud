"""
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
import scipy.ndimage

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
    """
    Displays the results of the accumulator algorithm in a cube shaped plot

    Input:
        consensus_cube: ((n, 4) np.ndarray)             First 3 columns: X, Y, Z translations, last: consensus_counts
        corresponding_cloud_size: (int)                 Point Number of the cloud compared in accumulator algorithm
        best_alignment: (3-tuple)                       Alignment result of the accumulator algorithm
        plot_title: (string)                            The Header of the Plot
        relative_color_scale: (boolean)                 If True, the maximum consensus will mark the top of the scale
    Output:
        consensus_cube: ((n, 4) np.ndarray)             Normalized consensus_cube
        matplotlib_figure_object: (matplotlib.pyplot)   Figure object containing the plot for further use
    """

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

    # sort the 4th column, containing the consensus values, best values last
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
    plt.draw ()

    return original_cube, matplotlib_figure_object


def create_closed_grid (grid_length, grid_size ):
    """
    Returns a (n, 4) numpy.ndarray which contains translations in x, y, z dimensions, spaced in gaps of grid_size and a
    forth, column filled with zeros to contain the consensus counts.
    """

    # grid variables
    steps_number = math.ceil (grid_length / grid_size + 1 )
    grid_points_number = steps_number**3

    # make a grid in the style of a pointcloud
    grid = np.zeros ((grid_points_number, 4 ))

    # in intervals of grid_size, create grid nodes
    general_iterator = 0
    min = -math.floor (steps_number / 2)
    max = math.ceil (steps_number / 2 )
    for x_iterator in range (min, max ):
        for y_iterator in range (min, max ):
            for z_iterator in range (min, max ):

                grid[general_iterator, 0:3] = [x_iterator * grid_size,
                                               y_iterator * grid_size,
                                               z_iterator * grid_size]

                general_iterator += 1

    return grid


def create_plot_title (base_title, accumulator_radius, grid_size, distance_threshold ):
    """Simple concatenation of strings to make a structured accumulator plot title"""
    plot_title = str(base_title
            + "_distance-accumulator"
            + "_sphere_radius_" + '{:.1f}'.format (accumulator_radius )
            + "_step_" + '{:.2f}'.format (grid_size ))
    if (distance_threshold is not None and distance_threshold > 0 ):
        plot_title = str(plot_title + "_distance_threshold_" + '{:.3f}'.format (distance_threshold ))

    return plot_title


def morph_consensus_cube (cube ):

    # add the maximum value to the coordinates, so they are positive
    cube[:, 0:3] += np.max (cube[:, 0])

    # normalize by new maximum, so values are distributed from 0 to 1
    cube[:, 0:3] /= np.max (cube[:, 0])

    # spread the values again by step count (depends on original grid_size of the cube), so the coordinates
    # are now monotonically rising real numbers that can easily be used for array indexing
    steps_count = int (cube.shape[0] ** (1/3 ))
    cube[:, 0:3] *= steps_count - 1

    # create new cube
    new_cube = np.zeros (shape=(steps_count + 1, steps_count + 1, steps_count + 1 ))
    for row in cube:
        new_cube[int (row[0]), int (row[1]), int (row[2])] = row[3]

    return new_cube


def morph_back (morphed_cube, grid_length=2.0 ):

    # get steps and fashion cube container in the style of a pointcloud
    steps_count = int (morphed_cube.shape[0] ** 3 )
    cube = np.zeros (shape=(steps_count, 4 ))

    #
    iterator = 0
    for x_dim in range (morphed_cube.shape[0] ):
        for y_dim in range (morphed_cube.shape[1] ):
            for z_dim in range (morphed_cube.shape[2] ):

                cube[iterator, :] = [x_dim, y_dim, z_dim, morphed_cube[int (x_dim), int (y_dim), int (z_dim)]]
                iterator += 1

    # normalize by steps_count, so values are distributed from 0 to 1
    cube[:, 0:3] /= np.max (cube[:, 0])

    # apply the original grid_length
    cube[:, 0:3] *= grid_length

    # add the maximum value to the coordinates, so they are positive
    cube[:, 0:3] -= np.max (cube[:, 0]) / 2

    return cube


def spheric_cloud_consensus (np_pointcloud, corresponding_pointcloud,
                             accumulator_radius=1.2, grid_size=0.1, distance_threshold=0.2,
                             display_plot=True, save_plot=False,
                             relative_color_scale=False,
                             plot_title="ConsensusCube (TM)"  ):
    """
    Counts how many points of cloud np_pointcloud have a neighbor within threshold range in corresponding_cloud.

    Input:
        np_pointcloud: (NumpyPointCloud)            NumpyPointCloud object containing a numpy array and it's data labels
        corresponding_pointcloud: (NumpyPointCloud) This cloud will be aligned to match np_pointcloud
        accumulator_radius: (float)                 Sphere center is (0,0,0). Determines maximum detectable translation
        grid_size: (float)                          Rasterization of results. May give unsatisfying results if too small
        distance_threshold: (float)                 Defines the range below which a point is counted as neighbor
        save_plot: (boolean)                        Whether to save the plot of the results for later use
        display_plot: (boolean)                     Whether to show the plot of the results
        relative_color_scale: (boolean)             See display_consensus_cube ()
        plot_title: (string)                        How to title the plot of the results

    Output:
        best_alignment: ((x, y, z) tuple )          The resulting alignment of corresponding_pointcloud
        highest_consensus_count: (int)              The maximum consensus count
    """

    print ("\nStarting Distance Accumulator Consensus" )
    print ("distance_threshold: " + str(distance_threshold ))
    print ("accumulator_radius: " + str(accumulator_radius ))
    print ("grid_size: " + str(grid_size ) + '\n' )

    if (display_plot and save_plot):
        message = ("Displaying the plot halts code execution until the plot is closed. This can be a problem when "
                  + "multiple computations are queued.")
        warnings.warn (message)

    # timing variables
    loop_cloud_query_time = 0
    loop_diff_time = 0
    loop_grid_query_time = 0
    rasterization_time = 0
    start_time = time.time ()
    interim = time.time ()

    # build a grid as a kdtree to discretize the results
    consensus_cube = create_closed_grid (accumulator_radius * 2, grid_size )
    grid_kdtree = scipy.spatial.cKDTree (consensus_cube[:, 0:3] )

    # build kdtree and query it for points within radius (radius being the maximum translation that can be detected)
    scipy_kdtree = scipy.spatial.cKDTree (np_pointcloud.get_xyz_coordinates ())
    # cloud_indices = scipy_kdtree.query_ball_point (
    #     np.ascontiguousarray(corresponding_pointcloud.points[:, 0:3]), accumulator_radius )   # memory problem
    init_time = time.time () - interim
    interim_2 = time.time ()

    # simplified process:
    #   for each point in corresponding_pointcloud:
    #       Find neighbors within accumulator_radius
    #       Compute the translations to these neighbors
    #       Rasterize the translations by matching them with the grid
    #       When a translation matches a grid cell, increment the consensus counter of that cell
    iterations = 0
    for i, point in enumerate (corresponding_pointcloud.get_xyz_coordinates ()):
        interim = time.time ()

        point_indices = scipy_kdtree.query_ball_point (point, accumulator_radius )

        loop_cloud_query_time += time.time () - interim

        # Progress Prints every 10 %
        if (i % int(corresponding_pointcloud.points.shape[0] / 10) == 0):
            print ("Progress: " + "{:.1f}".format ((i / corresponding_pointcloud.points.shape[0]) * 100.0 ) + " %" )

        if (len(point_indices ) > 0):
            interim = time.time ()
            # diff all points found near the corresponding point with corresponding point
            diff_vectors = np_pointcloud.points[point_indices, 0:3] - point
            # print ("\n--------------------------------------------------\n\npoint_indices:\n" + str (point_indices ))
            # print ("diff_vectors:\n" + str (diff_vectors ))
            loop_diff_time += time.time () - interim
            interim = time.time ()

            # rasterize by finding nearest grid node (representing a translation)
            dists, point_matches = grid_kdtree.query (diff_vectors, k=1 )
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
            iterations = i

    overall_loop_time = time.time () - interim_2
    interim = time.time ()

    # no gauss
    # consensus_cube = morph_consensus_cube (consensus_cube )
    # sigma = 1
    # consensus_cube = scipy.ndimage.gaussian_filter (consensus_cube, sigma, order=0, truncate=0.5 )
    # consensus_cube = morph_back (consensus_cube )

    # save the results
    best_alignment = consensus_cube[np.argmax (consensus_cube[:, 3] ), 0:3].copy ()
    highest_consensus_count = np.max (consensus_cube[:, 3] ).copy ()

    # sort the 4th column, containing the consensus values, best values last
    consensus_cube.view('i8,i8,i8,i8').sort(order=['f3'], axis=0 )
    np.set_printoptions(precision=6, linewidth=120, suppress=True )

    # Print Results
    print ("best_alignment: " + str (consensus_cube[-1, 0:3] ))
    print ("best_alignment_consensus: " + str (consensus_cube[-1, 3] ))

    print ("2nd best_alignment dist : " + str (np.linalg.norm (consensus_cube[-2, 0:3] - best_alignment )))
    print ("2nd best_alignment_consensus: " + str (consensus_cube[-2, 3] ))

    print ("3rd best_alignment dist: " + str (np.linalg.norm (consensus_cube[-3, 0:3] - best_alignment )))
    print ("3rd best_alignment_consensus: " + str (consensus_cube[-3, 3] ))

    # put together the plot title from the string given as argument to this function and the algorithmus parameters
    plot_title = create_plot_title (plot_title, accumulator_radius, grid_size, distance_threshold )

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

    # normalizing the timing
    loop_cloud_query_time /= iterations
    loop_diff_time /= iterations
    loop_grid_query_time /= iterations
    rasterization_time /= iterations

    # print ("\nInit Time: " + str (init_time ))
    # print ("Overall Loop Time: " + str (overall_loop_time ))
    # average_loop_time = overall_loop_time / iterations
    # print ("\tLoop Cloud Query Time: " + "{:.2f}%".format ((loop_cloud_query_time / average_loop_time) * 100 ))
    # print ("\tLoop Diff Time: " + "{:.2f}%".format ((loop_diff_time / average_loop_time) * 100 ))
    # print ("\tLoop Grid Query Time: " + "{:.2f}%".format ((loop_grid_query_time / average_loop_time) * 100 ))
    # print ("\tLoop Rasterization Time: " + "{:.2f}%".format ((rasterization_time / average_loop_time) * 100 ))
    # print ("Outro Time: " + str (outro_time ))
    # print ("Overall Time: " + str (time.time () - start_time ) + "\n")

    # display the plot
    if (display_plot ):
        _ = plt.show (figure )
    plt.close ()

    return best_alignment, highest_consensus_count
