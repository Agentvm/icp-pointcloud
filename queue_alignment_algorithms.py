"""
Convenient script to compute results of different algorithms (icp, consensus, accumulator) on multiple clouds. Works
the 'data/' folder in the form of loading an saving reference dictionaries that contain the cloud file paths and the
corresponding results in the form of {("",""); ((x,y,z), mse)}
"""


# local modules
from modules import input_output
from modules import icp
from modules import conversions
from modules import consensus
from modules import accumulator

# basic imports
import random


def set_accumulator_arguments (accumulator_radius=1.0, grid_size=0.05 ):
    """Configure the accumulator parameters. Set Search Radius and grid size for accumulator algorithm"""

    global ACCUMULATOR_RADIUS
    global ACCUMULATOR_GRID_SIZE

    ACCUMULATOR_RADIUS = accumulator_radius
    ACCUMULATOR_GRID_SIZE = grid_size


def accumulate (full_path_of_reference_cloud, full_path_of_aligned_cloud, plot_title ):
    """Function that can be passed to use_algorithmus_on_dictionary. Returns a dictionary line of results"""

    if ('ACCUMULATOR_RADIUS' not in globals() or 'ACCUMULATOR_GRID_SIZE' not in globals()):
        raise NameError("Consensus arguments are not defined. Call set_consensus_arguments() first.")

    # load clouds
    reference_pointcloud = input_output.conditionalized_load (full_path_of_reference_cloud )
    aligned_pointcloud = input_output.conditionalized_load (full_path_of_aligned_cloud )

    # reach consensus by accumulation of results
    best_alignment, best_consensus_count = accumulator.spheric_cloud_consensus (reference_pointcloud,
                                                                                aligned_pointcloud,
                                                                                accumulator_radius=ACCUMULATOR_RADIUS,
                                                                                grid_size=ACCUMULATOR_GRID_SIZE,
                                                                                distance_threshold=None,
                                                                                display_plot=False,
                                                                                save_plot=True,
                                                                                relative_color_scale=True,
                                                                                plot_title=plot_title )

    dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud):
                       (best_alignment, (best_consensus_count/aligned_pointcloud.points.shape[0], 0, 0))}

    return dictionary_line


def set_consensus_arguments (distance_threshold=.3, angle_threshold=30,
                             cubus_length=2, step=.15, algorithm="distance" ):
    """Configure the consensus parameters. angle_threshold in degrees"""
    global CONSENSUS_DISTANCE_THRESHOLD
    global CONSENSUS_ANGLE_THRESHOLD
    global CONSENSUS_CUBUS_LENGHT
    global CONSENSUS_STEP
    global CONSENSUS_ALGORITHM

    CONSENSUS_DISTANCE_THRESHOLD = distance_threshold
    CONSENSUS_ANGLE_THRESHOLD = angle_threshold
    CONSENSUS_CUBUS_LENGHT = cubus_length
    CONSENSUS_STEP = step
    CONSENSUS_ALGORITHM = algorithm


def reach_a_consensus (full_path_of_reference_cloud, full_path_of_aligned_cloud, plot_title ):
    """Function that can be passed to use_algorithmus_on_dictionary. Returns a dictionary line of results"""

    if ('CONSENSUS_DISTANCE_THRESHOLD' not in globals()
            or 'CONSENSUS_ANGLE_THRESHOLD' not in globals()
            or 'CONSENSUS_CUBUS_LENGHT' not in globals()
            or 'CONSENSUS_STEP' not in globals()
            or 'CONSENSUS_ALGORITHM' not in globals() ):
        raise NameError("Consensus arguments are not defined. Call set_consensus_arguments() first.")

    # load clouds
    reference_pointcloud = input_output.conditionalized_load (full_path_of_reference_cloud )
    aligned_pointcloud = input_output.conditionalized_load (full_path_of_aligned_cloud )

    best_alignment, best_consensus_count, best_alignment_consensus_vector = \
        consensus.cubic_cloud_consensus (reference_pointcloud,
                                         aligned_pointcloud,
                                         cubus_length=CONSENSUS_CUBUS_LENGHT,
                                         step=CONSENSUS_STEP,
                                         distance_threshold=CONSENSUS_DISTANCE_THRESHOLD,
                                         angle_threshold=CONSENSUS_ANGLE_THRESHOLD,
                                         algorithmus=CONSENSUS_ALGORITHM,
                                         plot_title=plot_title,
                                         save_plot=True)

    dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud):
                       (best_alignment, (best_consensus_count/aligned_pointcloud.points.shape[0], 0, 0))}

    return dictionary_line


def do_icp (full_path_of_reference_cloud, full_path_of_aligned_cloud, dummy_arg = "" ):
    """Function that can be passed to use_algorithmus_on_dictionary. Returns a dictionary line of results"""

    # load reference cloud
    reference_pointcloud = input_output.load_ascii_file (full_path_of_reference_cloud )

    # sample DIM clouds
    if ("DSM_Cloud" in full_path_of_reference_cloud):
        reference_pointcloud.points = conversions.sample_cloud (
                                                    reference_pointcloud.points, 6, deterministic_sampling=False )

    # load aligned clouds
    aligned_pointcloud = input_output.load_ascii_file (full_path_of_aligned_cloud )

    # sample DIM Clouds
    if ("DSM_Cloud" in full_path_of_aligned_cloud):
        aligned_pointcloud.points = conversions.sample_cloud (
                                                    aligned_pointcloud.points, 6, deterministic_sampling=False )

    translation, mean_squared_error = icp.icp (reference_pointcloud.points, aligned_pointcloud.points, verbose=False )

    dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud): (translation, mean_squared_error)}

    return dictionary_line


def print_reference_dict (reference_dictionary_name ):
    """
    Load a reference dictionary in the data/ folder by name (excluding extension) and print it's innards

    Input:
        reference_dictionary_name: (String)   Name of a dict located in 'data/' of shape {("",""); ((x,y,z), mse)}
    """

    # parse the reference values saved in a file
    reference_dictionary = input_output.load_obj (reference_dictionary_name )
    print ("\n" + str (reference_dictionary_name ) + ":" )

    # iterate through the keys (path pairs) of the dictionary
    for path_tuple in sorted(reference_dictionary ):

        # disassemble the key
        reference_path, aligned_path = path_tuple
        results_tuple = reference_dictionary[path_tuple]

        # folder should be the the same
        folder, reference_file_name = input_output.get_folder_and_file_name (reference_path)
        folder, aligned_file_name = input_output.get_folder_and_file_name (aligned_path)

        # unpack values
        ref_translation, ref_mse = results_tuple

        # print comparison
        print ("'" + aligned_file_name
               + "' aligned to '" + reference_file_name + "'"
               + ';{: .8f}'.format(ref_translation[0])
               + '\n;{: .8f}'.format(ref_translation[1])
               + '\n;{: .8f}'.format(ref_translation[2])
               + '\n;{: .8f}'.format(ref_mse[0]))


def compare_results (dictionary, reference_dict, print_csv=True ):
    """
    Given a dictionary of results, this compares the results against a dictionary of reference values

    Input:
        dictionary: (dictionary)   A dictionary of str tuples and translation results {("",""); ((x,y,z), mse)}
        reference_dict: (dictionary)        Contains cloud path tuples and trasnlation results {("",""); ((x,y,z), mse)}
        print_csv: (boolean)                If True, the output is separated by ';' and can be easily processed further
    """

    # # sort the results
    # create a list of tuples from reference and aligned cloud file paths
    unsorted_results = []
    for paths in dictionary:
        unsorted_results.append ((paths, dictionary[paths]) )
    sorted_results = sorted(unsorted_results )

    for paths, translation_value in sorted_results:

        if (paths in reference_dict ):

            # disassemble the key
            reference_path, aligned_path = paths

            # folder should be the the same
            folder, reference_file_name = input_output.get_folder_and_file_name (reference_path)
            folder, aligned_file_name = input_output.get_folder_and_file_name (aligned_path)

            # unpack values
            ref_translation, ref_mse = reference_dict [paths]
            algorithmus_translation, algorithmus_mse = translation_value

            if (print_csv):
                # print comparison
                print ('\n' + folder + "/"
                       + " " + reference_file_name
                       + " " + aligned_file_name
                             + ';{: .8f}'.format(ref_translation[0]) + ';{: .8f}'.format(algorithmus_translation[0])
                            + '\n;{: .8f}'.format(ref_translation[1]) + ';{: .8f}'.format(algorithmus_translation[1])
                            + '\n;{: .8f}'.format(ref_translation[2]) + ';{: .8f}'.format(algorithmus_translation[2])
                            + '\n;{: .8f}'.format(ref_mse) + ';=MAX({: .8f}'.format(algorithmus_mse[0])
                             + ',{: .8f}'.format(algorithmus_mse[1])
                             + ',{: .8f}) '.format(algorithmus_mse[2]))
            else:
                # print comparison
                print ('\n' + folder + "/"
                       + "\nreference cloud:\t" + reference_file_name
                       + "\naligned cloud:\t\t" + aligned_file_name
                       + "\n\tdata alignment:\t\t" + '({: .8f}, '.format(ref_translation[0])
                                                   + '{: .8f}, '.format(ref_translation[1])
                                                   + '{: .8f}), '.format(ref_translation[2])
                                                   + ' {: .8f}, '.format(ref_mse)
                       + "\n\talgorithmus alignment:\t" + '({: .8f}, '.format(algorithmus_translation[0])
                                                + '{: .8f}, '.format(algorithmus_translation[1])
                                                + '{: .8f}), '.format(algorithmus_translation[2])
                                                + '({: .8f}, '.format(algorithmus_mse[0])
                                                + '{: .8f}, '.format(algorithmus_mse[1])
                                                + '{: .8f}) '.format(algorithmus_mse[2]))


# # TODO:  The clouds are then displaced by the translations found in the dictionary.
def use_algorithmus_on_dictionary (reference_dictionary_name, algorithmus_function, results_save_name=None ):
    """
    Uses a dictionary with path tuples (reference cloud file path, aligned cloud file path ) as keys
    and a results tuple (translation, mse) as values to extract the paths and load the clouds. Dictionary structure:
    {(reference path, aligned_path): ((x, y, z), mse)}

    Also uses a Python function that takes three strings (reference_cloud_path, aligned_cloud_path, plot_title) and
    returns a dictionary line in the form of {(reference path, aligned_path): (translation, mse)}

    It then creates a new dictionary of the given form and saves it as 'data/results_save_name.pkl'.

    Input:
        reference_dictionary_name: (string) Dictionary with paths tuple as keys for extraction of file locations
        algorithmus_function: (function)    Function that returns {(reference path, aligned_path); (translation, mse)}
        results_save_name: (string)         Results will be saved in 'data/'. Values may be overwritten.

    Output:
        sucess: (boolean)                   Is true on success
    """

    # parse the reference values saved in a file
    reference_dictionary = input_output.load_obj (reference_dictionary_name )
    file_paths_dictionary = get_reference_data_paths (reference_dictionary )

    # before start, check if files exist
    for key in file_paths_dictionary:
        if (input_output.check_for_file (key ) is False):
            print ("File " + key + " was not found. Aborting.")
            return False
        for aligned_cloud_path in file_paths_dictionary[key]:
            if (input_output.check_for_file (aligned_cloud_path ) is False):
                print ("File " + aligned_cloud_path + " was not found. Aborting.")
                return False

    algorithmus_results = {}    # dictionary

    # create a list of tuples from reference and aligned cloud file paths
    for reference_cloud_path in file_paths_dictionary:
        for aligned_cloud_path in file_paths_dictionary[reference_cloud_path]:   # multiple aligned clouds possible

            folder, reference_file_name = input_output.get_folder_and_file_name (reference_cloud_path)
            folder, aligned_file_name = input_output.get_folder_and_file_name (aligned_cloud_path)
            plot_title = folder + ' ' + aligned_file_name + ' to ' + reference_file_name

            # call the algorithmus supplied by algorithmus_function
            algorithmus_results.update (algorithmus_function (reference_cloud_path, aligned_cloud_path, plot_title ))

    # save the results in a dictionary and print it
    if (results_save_name is not None ):
        input_output.save_obj (algorithmus_results, results_save_name)
        print_reference_dict (results_save_name )
    else:
        # prints the values computed along with the ground truth in the dictionary
        compare_results (algorithmus_results, reference_dictionary )

    return True


def get_reference_data_paths (reference_dict ):
    """
    Reads input_dictionary to get all transformations currently saved and returns them in a
    dictionary that can be directly used with use_algorithmus_on_dictionary()
    """

    dict = {}
    for key in reference_dict:

        reference_path, aligned_path = key

        if (dict.__contains__ (reference_path )):
            dict[reference_path].append (aligned_path )
        else:
            dict.update ({reference_path: [aligned_path]} )

    return dict


if __name__ == '__main__':

    random.seed (1337 )

    # # icp
    # print ("\n\nComputing ICP for each cloud pair in reference_translations returns: "
    #        + str(use_algorithmus_on_dictionary (get_reference_data_paths (), do_icp )))
    #
    # compare_results (do_icp ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    #                          'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc' ), print_csv=True)

    # # # consensus
    # set_consensus_arguments (distance_threshold=0.3,
    #                          angle_threshold=None,
    #                          cubus_length=2,
    #                          step=0.15,
    #                          algorithm='distance' )
    #
    # print ("\n\nComputing Consensus for each cloud pair in reference_translations_dict returns: "
    #        + str(use_algorithmus_on_dictionary (reference_dictionary_name="reference_translations_dict",
    #                                             algorithmus_function=reach_a_consensus,
    #                                             results_save_name="distance_consensus_translations_dict" )))

    # # accumulator
    set_accumulator_arguments ()

    print ("\n\nComputing Accumulator Consensus for each cloud pair in reference_translations_dict returns: "
           + str(use_algorithmus_on_dictionary (reference_dictionary_name="no_translations_dict",
                                                algorithmus_function=accumulate,
                                                results_save_name="accumulator_translations_dict" )))

    # # # icp
    # print ("\n\nComputing ICP for each cloud pair in no_translations_dict returns: "
    #        + str(use_algorithmus_on_dictionary (reference_dictionary_name="no_translations_dict",
    #                                             algorithmus_function=do_icp,
    #                                             results_save_name="icp_translations_dict" )))

    # # print saved dictionaries
    # #print (str(input_output.load_obj ("no_translations_dict" )).replace ("), ('", "),\n('" ))
    # print_reference_dict ("icp_translations_dict" )

    # # get folder structure
    # for path in input_output.get_all_files_in_subfolders("clouds/New Regions/", ".asc" ):
    #     print (path )

    # # Make a dict of test cases (tuple of paths) as keys
    # # and the corresponding results (tuple of translation and mse) as values
    # dict = \
    #     {
    #         ("clouds/New Regions/Color_Houses/Color Houses_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Color_Houses/Color Houses_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/DIM_showcase/DIM showcase_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/DIM_showcase/DIM showcase_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Everything/Everything_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Everything/Everything_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #         ("clouds/New Regions/Everything/Everything_als14_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Everything/Everything_als16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #         ("clouds/New Regions/Everything/Everything_als14_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Everything/Everything_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Field/Field_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Field/Field_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Forest/Forest_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Forest/Forest_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Missing_Building/Missing Building_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Missing_Building/Missing Building_dim16_reduced_normals_r_1_cleared.asc"):
    # ((0, 0, 0), 0),
    #         ("clouds/New Regions/Missing_Building/Missing Building_als14_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Missing_Building/Missing Building_als16_reduced_normals_r_1_cleared.asc"):
    # ((0, 0, 0), 0),
    #         ("clouds/New Regions/Missing_Building/Missing Building_als14_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Missing_Building/Missing Building_dim16_reduced_normals_r_1_cleared.asc"):
    # ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Road/Road_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Road/Road_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Xyz_Square/Xyz Square_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Xyz_Square/Xyz Square_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Xy_Tower/Xy Tower_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Xy_Tower/Xy Tower_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Xz_Hall/Xz Hall_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Xz_Hall/Xz Hall_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Yz_Houses/Yz Houses_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Yz_Houses/Yz Houses_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0),
    #
    #         ("clouds/New Regions/Yz_Street/Yz Street_als16_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Yz_Street/Yz Street_dim16_reduced_normals_r_1_cleared.asc"): ((0, 0, 0), 0)
    #     }
    #
    # input_output.save_obj(dict, "no_translations_dict" )
