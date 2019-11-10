"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

Convenient script to compute results of different algorithms (icp, consensus, accumulator) on multiple clouds. Works
the 'data/' folder in the form of loading an saving reference dictionaries that contain the cloud file paths and the
corresponding results in the form of {("",""); ((x,y,z), mse)}. Saved results can be used as inital alignment basis in
other experiments by choosing them using the parameter "reference_dictionary_name".
"""


# local modules
from modules import input_output
from modules import icp
from modules import conversions
from modules import consensus
from modules import accumulator, diagnosis
from modules.np_pointcloud import NumpyPointCloud

# basic imports
import random

# advanced functionality
import scipy.spatial


def set_pruning_arguments (prune_borders=True, borders_clearance=1.2,
                           prune_water_bodies=True,
                           prune_sigma=True, max_sigma_value=0.05,
                           prune_outliers=True, max_outlier_distance=0.5,
                           prune_normals=True, max_angle_difference=32 ):
    """Configure the pruning of the Cloud"""

    global PRUNE_BORDERS
    global BORDERS_CLEARANCE
    global PRUNE_WATER_BODIES
    global PRUNE_SIGMA
    global MAX_SIGMA_VALUE
    global PRUNE_OUTLIERS
    global MAX_OUTLIER_DISTANCE
    global PRUNE_NORMALS
    global MAX_ANGLE_DIFFERENCE

    PRUNE_BORDERS = prune_borders
    BORDERS_CLEARANCE = borders_clearance
    PRUNE_WATER_BODIES = prune_water_bodies
    PRUNE_SIGMA = prune_sigma
    MAX_SIGMA_VALUE = max_sigma_value
    PRUNE_OUTLIERS = prune_outliers
    MAX_OUTLIER_DISTANCE = max_outlier_distance
    PRUNE_NORMALS = prune_normals
    MAX_ANGLE_DIFFERENCE = max_angle_difference


def check_pruning_arguments ():
    return ('PRUNE_BORDERS' in globals()
            or 'BORDERS_CLEARANCE' in globals()
            or 'PRUNE_WATER_BODIES' in globals()
            or 'PRUNE_SIGMA' in globals()
            or 'MAX_SIGMA_VALUE' in globals()
            or 'PRUNE_OUTLIERS' in globals()
            or 'MAX_OUTLIER_DISTANCE' in globals()
            or 'PRUNE_NORMALS' in globals()
            or 'MAX_ANGLE_DIFFERENCE' in globals())


def set_accumulator_arguments (accumulator_radius=1.0, grid_size=0.05 ):
    """Configure the accumulator parameters. Set Search Radius and grid size for accumulator algorithm"""

    global ACCUMULATOR_RADIUS
    global ACCUMULATOR_GRID_SIZE

    ACCUMULATOR_RADIUS = accumulator_radius
    ACCUMULATOR_GRID_SIZE = grid_size


def accumulate (reference_pointcloud, aligned_pointcloud, plot_title ):
    """
    Function that can be passed to use_algorithm_on_dictionary. Returns a results tuple containing a translation
    and mse values ((x,y,z), (mse_x, mse_y, mse_z))
    """

    if ('ACCUMULATOR_RADIUS' not in globals() or 'ACCUMULATOR_GRID_SIZE' not in globals()):
        raise NameError("Consensus arguments are not defined. Call set_consensus_arguments() first.")

    # sample DIM clouds (which have color fields)
    if (reference_pointcloud.has_fields (["Rf", "Gf", "Bf"] )):
        reference_pointcloud.points = conversions.sample_cloud (
                                                    reference_pointcloud.points, 6, deterministic_sampling=False )

    # sample DIM Clouds (which have color fields)
    if (aligned_pointcloud.has_fields (["Rf", "Gf", "Bf"] )):
        aligned_pointcloud.points = conversions.sample_cloud (
                                                    aligned_pointcloud.points, 6, deterministic_sampling=False )

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

    return (best_alignment, (best_consensus_count/aligned_pointcloud.points.shape[0], 0, 0))


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


def reach_a_consensus (reference_pointcloud, aligned_pointcloud, plot_title ):
    """
    Function that can be passed to use_algorithm_on_dictionary. Returns a results tuple containing a translation
    and mse values ((x,y,z), (mse_x, mse_y, mse_z))
    """

    if ('CONSENSUS_DISTANCE_THRESHOLD' not in globals()
            or 'CONSENSUS_ANGLE_THRESHOLD' not in globals()
            or 'CONSENSUS_CUBUS_LENGHT' not in globals()
            or 'CONSENSUS_STEP' not in globals()
            or 'CONSENSUS_ALGORITHM' not in globals() ):
        raise NameError("Consensus arguments are not defined. Call set_consensus_arguments() first.")

    best_alignment, best_consensus_count, best_alignment_consensus_vector = \
        consensus.cubic_cloud_consensus (reference_pointcloud,
                                         aligned_pointcloud,
                                         cubus_length=CONSENSUS_CUBUS_LENGHT,
                                         step=CONSENSUS_STEP,
                                         distance_threshold=CONSENSUS_DISTANCE_THRESHOLD,
                                         angle_threshold=CONSENSUS_ANGLE_THRESHOLD,
                                         algorithm=CONSENSUS_ALGORITHM,
                                         plot_title=plot_title,
                                         save_plot=True)

    return (best_alignment, (best_consensus_count/aligned_pointcloud.points.shape[0], 0, 0))


def do_icp (reference_pointcloud, aligned_pointcloud, dummy_arg = "" ):
    """
    Function that can be passed to use_algorithm_on_dictionary. Returns a results tuple containing a translation
    and mse values ((x,y,z), (mse_x, mse_y, mse_z))
    """

    # sample DIM clouds (which have color fields)
    if (reference_pointcloud.has_fields (["Rf", "Gf", "Bf"] )):
        reference_pointcloud.points = conversions.sample_cloud (
                                                    reference_pointcloud.points, 6, deterministic_sampling=False )

    # sample DIM Clouds (which have color fields)
    if (aligned_pointcloud.has_fields (["Rf", "Gf", "Bf"] )):
        aligned_pointcloud.points = conversions.sample_cloud (
                                                    aligned_pointcloud.points, 6, deterministic_sampling=False )

    translation, mean_squared_error = icp.icp (reference_pointcloud.points, aligned_pointcloud.points, verbose=False )

    #dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud): (translation, mean_squared_error)}

    return (translation, mean_squared_error)


def rate (reference_pointcloud, aligned_pointcloud, dummy_arg = "" ):
    """
    Function that can be passed to use_algorithm_on_dictionary. Returns a results tuple containing a null-translation
    (0, 0, 0) and mse values (0, 0, 0) ((x,y,z), (mse_x, mse_y, mse_z))
    """

    return ((0, 0, 0 ), (0, 0, 0 ))


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
            algorithm_translation, algorithm_mse = translation_value

            if (print_csv):
                # print comparison
                print ('\n' + folder + "/"
                       + " " + reference_file_name
                       + " " + aligned_file_name
                             + ';{: .8f}'.format(ref_translation[0]) + ';{: .8f}'.format(algorithm_translation[0])
                            + '\n;{: .8f}'.format(ref_translation[1]) + ';{: .8f}'.format(algorithm_translation[1])
                            + '\n;{: .8f}'.format(ref_translation[2]) + ';{: .8f}'.format(algorithm_translation[2])
                            + '\n;{: .8f}'.format(ref_mse) + ';=MAX({: .8f}'.format(algorithm_mse[0])
                             + ',{: .8f}'.format(algorithm_mse[1])
                             + ',{: .8f}) '.format(algorithm_mse[2]))
            else:
                # print comparison
                print ('\n' + folder + "/"
                       + "\nreference cloud:\t" + reference_file_name
                       + "\naligned cloud:\t\t" + aligned_file_name
                       + "\n\tdata alignment:\t\t" + '({: .8f}, '.format(ref_translation[0])
                                                   + '{: .8f}, '.format(ref_translation[1])
                                                   + '{: .8f}), '.format(ref_translation[2])
                                                   + ' {: .8f}, '.format(ref_mse)
                       + "\n\talgorithm alignment:\t" + '({: .8f}, '.format(algorithm_translation[0])
                                                + '{: .8f}, '.format(algorithm_translation[1])
                                                + '{: .8f}), '.format(algorithm_translation[2])
                                                + '({: .8f}, '.format(algorithm_mse[0])
                                                + '{: .8f}, '.format(algorithm_mse[1])
                                                + '{: .8f}) '.format(algorithm_mse[2]))


# DEBUG
SOME_NAME = "rating_test_c2c_PRUNE_dict"


def rate_cloud_alignment (original_reference_pointcloud, original_aligned_pointcloud, translation ):
    """
    Measures cloud alignment quality by comparing the sum of nearest neighbor distances before and after translation
    """

    # prune clouds, so that points that do not match the corresponding other cloud's model are not considered
    original_reference_pointcloud, original_aligned_pointcloud = \
        conversions.prune_cloud_pair (original_reference_pointcloud, original_aligned_pointcloud,
                                      translation=translation,
                                      prune_borders=False,
                                      prune_water_bodies=True,
                                      prune_sigma=False,
                                      prune_outliers=True,
                                      max_outlier_distance=0.5,
                                      prune_normals=False )

    # print ("1")
    #
    # threshold = 0.1
    # copy_of_aligned_pointcloud = NumpyPointCloud (
    #     original_aligned_pointcloud.points.copy (), original_aligned_pointcloud.field_labels )
    # reference_kdtree = scipy.spatial.KDTree (original_reference_pointcloud.get_xyz_coordinates () )
    #
    # print ("2")
    #
    # criterion_before, _ = consensus.point_distance_cloud_consensus(
    #     reference_kdtree, original_aligned_pointcloud, (0, 0, 0), threshold )
    #
    # print ("3")
    #
    # criterion_after, _ = consensus.point_distance_cloud_consensus(
    #     reference_kdtree, copy_of_aligned_pointcloud, translation, threshold )
    #
    # print ("4")

    # Criterion for alignment quality is the sum of all nearest neighbor distances
    criterion_after = diagnosis.cloud2cloud_distance_sum (original_reference_pointcloud, original_aligned_pointcloud )

    # Criterion for alignment quality with original clouds
    criterion_before = diagnosis.cloud2cloud_distance_sum (original_reference_pointcloud,
                                                          original_aligned_pointcloud,
                                                          translation=translation )

    # return criterion_after/criterion_before
    return criterion_before/criterion_after


# # TODO:  The clouds are then displaced by the translations found in the dictionary.
def use_algorithm_on_dictionary (reference_dictionary_name, algorithm_function,
                                   results_save_name=None, prune_clouds=False ):
    """
    Uses a dictionary with path tuples (reference cloud file path, aligned cloud file path ) as keys
    and a results tuple (translation, mse) as values to extract the paths and load the clouds. Dictionary structure:
    {(reference path, aligned_path): ((x, y, z), mse)}

    Also uses a Python function that takes three strings (reference_cloud_path, aligned_cloud_path, plot_title) and
    returns a dictionary line in the form of {(reference path, aligned_path): (translation, mse)}

    It then creates a new dictionary of the given form and saves it as 'data/results_save_name.pkl'.

    Input:
        reference_dictionary_name: (string) Dictionary with paths tuple as keys for extraction of file locations
        algorithm_function: (function)    Function that returns {(reference path, aligned_path); (translation, mse)}
        results_save_name: (string)         Results will be saved in 'data/'. Values may be overwritten.
        prune_clouds: (boolean)             Set True if clouds are reasonably aligned. Removes undesirable points.
                                            Points of both cloud are filtered for bad sigma values from normal calcu-
                                            lation, cloud edged are removed to avoid edge biases and points that have
                                            no corresponding neighbor in the other cloud are removed aswell.
                                            For more info, see modules/conversions.prune_cloud_pair()

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

    algorithm_results = {}    # dictionary

    # create a list of tuples from reference and aligned cloud file paths
    for reference_cloud_path in file_paths_dictionary:
        for aligned_cloud_path in file_paths_dictionary[reference_cloud_path]:   # multiple aligned clouds possible

            folder, reference_file_name = input_output.get_folder_and_file_name (reference_cloud_path)
            folder, aligned_file_name = input_output.get_folder_and_file_name (aligned_cloud_path)
            plot_title = folder + ' ' + aligned_file_name + ' to ' + reference_file_name

            # load clouds
            reference_pointcloud = input_output.conditionalized_load (reference_cloud_path )
            aligned_pointcloud = input_output.conditionalized_load (aligned_cloud_path )

            # create backups for alignment quality criterion
            original_reference_pointcloud = NumpyPointCloud (reference_pointcloud.points.copy (),
                                                             reference_pointcloud.field_labels )
            original_aligned_pointcloud = NumpyPointCloud (aligned_pointcloud.points.copy (),
                                                           aligned_pointcloud.field_labels )

            # displace the aligned cloud with the translation saved in the reference dictionary
            translation = reference_dictionary[(reference_cloud_path, aligned_cloud_path)][0]
            aligned_pointcloud.points[:, 0:3] += translation

            # remove undesireable points of both clouds to allow for a smooth alignment with less outliers
            # or wrong point correspondences (see: set_pruning_arguments ())
            if (prune_clouds ):
                if (check_pruning_arguments ()):
                    reference_pointcloud, aligned_pointcloud = \
                        conversions.prune_cloud_pair (reference_pointcloud, aligned_pointcloud,
                                                      prune_borders=PRUNE_BORDERS,
                                                      borders_clearance=BORDERS_CLEARANCE,
                                                      prune_water_bodies=PRUNE_WATER_BODIES,
                                                      prune_sigma=PRUNE_SIGMA,
                                                      max_sigma_value=MAX_SIGMA_VALUE,
                                                      prune_outliers=PRUNE_OUTLIERS,
                                                      max_outlier_distance=MAX_OUTLIER_DISTANCE,
                                                      prune_normals=PRUNE_NORMALS,
                                                      max_angle_difference=MAX_ANGLE_DIFFERENCE )
                else:
                    raise ValueError ("No Pruning parameters were set, please use set_pruning_arguments().")

            # call the algorithm supplied by algorithm_function and update the results dictionary
            pre_results = algorithm_function (reference_pointcloud, aligned_pointcloud, plot_title )

            # add the previously applied translation to the resulting tranlsation and update the dictionary of results
            # (don't use a tuple if you want to alter it ever again)
            results = ((pre_results[0][0] + translation[0],
                        pre_results[0][1] + translation[1],
                        pre_results[0][2] + translation[2] ), pre_results[1] )

            # find a measure for cloud alignment
            measure = rate_cloud_alignment(original_reference_pointcloud, original_aligned_pointcloud, results[0] )

            # add the previously applied translation to the resulting tranlsation and update the dictionary of results
            # (don't use a tuple if you want to alter it ever again)
            results = ((pre_results[0][0] + translation[0],
                        pre_results[0][1] + translation[1],
                        pre_results[0][2] + translation[2] ), measure )

            algorithm_results.update ({(reference_cloud_path, aligned_cloud_path): results} )

    # save the results in a dictionary and print it
    if (results_save_name is not None ):
        input_output.save_obj (algorithm_results, results_save_name)
        input_output.print_reference_dict (results_save_name )
    else:
        # prints the values computed along with the ground truth in the dictionary
        compare_results (algorithm_results, reference_dictionary )

    return True


def get_reference_data_paths (reference_dict ):
    """
    Reads input_dictionary to get all transformations currently saved and returns them in a
    dictionary that can be directly used with use_algorithm_on_dictionary()
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

    # ### Example 1 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # print saved dictionaries
    input_output.print_reference_dict ("new_regions/combined_consensus_translations_dict" )

    # ### Example 2 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # # # compute a coarse alignment using the consensus algorithm on a dictionary without previous alignments. The
    # # # dictionary does, in this case, only contain the file path data to the point clouds.
    # set_consensus_arguments (distance_threshold=0.2,
    #                          angle_threshold=32,
    #                          cubus_length=1,
    #                          step=0.05,
    #                          algorithm='distance' )
    #
    # print ("\n\nComputing Consensus for each cloud pair in reference dict returns: "
    #        + str(use_algorithm_on_dictionary (reference_dictionary_name="new_regions/no_translations_dict",
    #                                             algorithm_function=reach_a_consensus,
    #                                             results_save_name="distance_consensus_translations" )))

    # ### Example 3 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # # # accumulator
    # set_accumulator_arguments (accumulator_radius=1.0, grid_size=0.05)
    #
    # print ("\n\nComputing Accumulator Consensus for each cloud pair in no_translations_dict returns: "
    #        + str(use_algorithm_on_dictionary (reference_dictionary_name="new_regions/no_translations_dict",
    #                                             algorithm_function=accumulate,
    #                                             results_save_name="accumulator_sampling_translations_dict" )))

    # ### Example 4 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # # # icp and pruning based on the initial alignment of distance consensus.
    # set_pruning_arguments (prune_sigma=False,
    #                        prune_borders=True,
    #                        prune_normals=False, max_angle_difference=32,
    #                        prune_outliers=True, max_outlier_distance=0.5,
    #                        prune_water_bodies=True)
    #
    # print ("\n\nComputing ICP for each cloud pair in reference dict returns: "
    #        + str(use_algorithm_on_dictionary (reference_dictionary_name="new_regions/distance_consensus_translations_dict",
    #                                             algorithm_function=do_icp,
    #                                             results_save_name="prune_onesided_DistCon-distpruning_translations_dict",
    #                                             prune_clouds=True )))

    # ### Example 5 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # # join dictionaries that have their results split up
    # input_output.join_saved_dictionaries (["some_path/combined_consensus_translations_part1_dict",
    #                                        "some_path/combined_consensus_translations_part2_dict",
    #                                        "some_path/combined_consensus_translations_part3_dict",
    #                                        "some_path/combined_consensus_translations_part4_dict",
    #                                        "some_path/combined_consensus_translations_part5_dict"],
    #                                       "combined_consensus_translations_dict")

    # ### Example 6 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # # get folder structure
    # for path in input_output.get_all_files_in_subfolders("clouds/New Regions/", ".asc" ):
    #     print (path )

    # ### Example 7 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # # Create a new dictionary with test cases (tuple of paths) as keys and
    # # the corresponding results (tuple of translation and mse or another value) as values
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
    #         ((0, 0, 0), 0),
    #         ("clouds/New Regions/Missing_Building/Missing Building_als14_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Missing_Building/Missing Building_als16_reduced_normals_r_1_cleared.asc"):
    #         ((0, 0, 0), 0),
    #         ("clouds/New Regions/Missing_Building/Missing Building_als14_reduced_normals_r_1_cleared.asc",
    #         "clouds/New Regions/Missing_Building/Missing Building_dim16_reduced_normals_r_1_cleared.asc"):
    #         ((0, 0, 0), 0),
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
    # input_output.save_obj(dict, "new_no_translations_dict" )
