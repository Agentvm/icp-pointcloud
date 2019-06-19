from modules import input_output
from modules import icp
from modules import conversions
from modules import consensus
from data import transformations
from collections import OrderedDict
import random
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def set_consensus_arguments (distance_threshold=.012, angle_threshold=0.1, cubus_length=2, step=.2):
    """angle_threshold in rad"""
    global CONSENSUS_DISTANCE_THRESHOLD
    global CONSENSUS_ANGLE_THRESHOLD
    global CONSENSUS_CUBUS_LENGHT
    global CONSENSUS_STEP

    CONSENSUS_DISTANCE_THRESHOLD = distance_threshold
    CONSENSUS_ANGLE_THRESHOLD = angle_threshold
    CONSENSUS_CUBUS_LENGHT = cubus_length
    CONSENSUS_STEP = step


def reach_a_consensus (full_path_of_reference_cloud, full_path_of_aligned_cloud, plot_title ):

    if ('CONSENSUS_DISTANCE_THRESHOLD' not in globals()
            or 'CONSENSUS_ANGLE_THRESHOLD' not in globals()
            or 'CONSENSUS_CUBUS_LENGHT' not in globals()
            or 'CONSENSUS_STEP' not in globals() ):
        raise NameError("Consensus arguments are not defined. Call set_consensus_arguments() first.")

    # load clouds
    reference_cloud, reference_cloud_field_labels = input_output.conditionalized_load (full_path_of_reference_cloud )
    aligned_cloud, aligned_cloud_field_labels = input_output.conditionalized_load (full_path_of_aligned_cloud )

    best_alignment, best_consensus_count, best_alignment_consensus_vector = \
        consensus.cubic_cloud_consensus (reference_cloud, reference_cloud_field_labels,
                                         aligned_cloud, aligned_cloud_field_labels,
                                         cubus_length=CONSENSUS_CUBUS_LENGHT,
                                         step=CONSENSUS_STEP,
                                         distance_threshold=CONSENSUS_DISTANCE_THRESHOLD,
                                         angle_threshold=CONSENSUS_ANGLE_THRESHOLD,
                                         algorithmus='angle',
                                         plot_title=plot_title)

    dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud):
                       (best_alignment, (best_consensus_count, 0, 0))}

    return dictionary_line


def do_icp (full_path_of_reference_cloud, full_path_of_aligned_cloud, dummy_arg = "" ):

    # load reference cloud
    reference_cloud = input_output.load_ascii_file (full_path_of_reference_cloud )

    if ("DSM_Cloud" in full_path_of_reference_cloud):
        reference_cloud = conversions.sample_cloud (reference_cloud, 6, deterministic_sampling=False )

    # load aligned clouds
    aligned_cloud = input_output.load_ascii_file (full_path_of_aligned_cloud )

    # sample DIM Clouds
    if ("DSM_Cloud" in full_path_of_aligned_cloud):
        aligned_cloud = conversions.sample_cloud (aligned_cloud, 6, deterministic_sampling=False )

    translation, mean_squared_error = icp.icp (reference_cloud, aligned_cloud, verbose=False )

    dictionary_line = {(full_path_of_reference_cloud, full_path_of_aligned_cloud): (translation, mean_squared_error)}

    return dictionary_line


def compare_results (algorithmus_results, print_csv=True ):

    reference_dict = transformations.reference_translations

    # # sort the results
    # create a list of tuples from reference and aligned cloud file paths
    unsorted_results = []
    for paths in algorithmus_results:
        unsorted_results.append ((paths, algorithmus_results[paths]) )

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


def use_algorithmus_on_dictionary (file_paths_dictionary, algorithmus_function ):
    '''
    Uses a dictionary of reference cloud file_paths as keys
    and a list of corresponding aligned cloud file_paths as values

    Input:
        file_paths_dictionary (string):  Dictionary with reference_paths as keys and paths of aligned clouds as values
        algorithmus_function (function): Function that returns dict {(reference path, aligned_path): (translation, mse)}
    '''

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

    # prints the values computed along with the ground truth in the dictionary
    compare_results (algorithmus_results )

    return True


# refactor
def print_files_dict_of_folder (folder, permitted_file_extension=None ):
    full_paths = input_output.get_all_files_in_subfolders (folder, permitted_file_extension )

    dict = OrderedDict ()
    for path in full_paths:
        dict.update ({('?reference?', path): ((0.0, 0.0, 0.0), 0.0)} )

    print ("Paths in Folder "
           + folder
           + ':\n\n'
           + str (dict.replace ('\', ', '\',\n').replace (': ', ':\n').replace ('), ', '),\n' )))

    return dict

    # + str (full_paths ).replace (', ', '\n' ).replace ('\'', '' ).strip ('[' ).strip (']' ).replace
    #  ('\n', ':\n((0.0, 0.0, 0.0), 0.0),\n' ))


def get_reference_data_paths ():
    '''
    Reads transformations.reference_translations to get all transformations currently saved and returns them in a
    dictionary that can be directly used with use_algorithmus_on_dictionary()
    '''
    dict = {}
    for key in transformations.reference_translations:

        reference_path, aligned_path = key

        if (dict.__contains__ (reference_path )):
            dict[reference_path].append (aligned_path )
        else:
            dict.update ({reference_path: [aligned_path]} )

    return dict


if (random.seed != 1337):
    random.seed = 1337
    print ("Random Seed set to: " + str(random.seed ))

if __name__ == '__main__':

    if (random.seed != 1337):
        random.seed = 1337
        print ("Random Seed set to: " + str(random.seed ))

    # # icp
    # print ("\n\nComputing ICP for each cloud pair in transformations.reference_translations returns: "
    #        + str(use_algorithmus_on_dictionary (get_reference_data_paths (), do_icp )))
    #
    # compare_results (do_icp ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    #                          'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc' ), print_csv=True)

    # # consensus
    set_consensus_arguments (distance_threshold=0.009, cubus_length=2, step=.2 )
    set_consensus_arguments (distance_threshold=None, angle_threshold=5 * (np.pi/180), cubus_length=2, step=0.2 )

    print ("\n\nComputing Consensus for each cloud pair in transformations.reference_translations returns: "
           + str(use_algorithmus_on_dictionary (get_reference_data_paths (), reach_a_consensus )))

    # compare_results (reach_a_consensus ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    #                                     'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc' ), print_csv=False)

    plt.show ()

    # # tests

    # print_files_dict_of_folder(folder)

    # print (get_icp_data_paths ())

    # if (use_icp_on_folder ('clouds/Regions/',
    #                        reference_file_tag='ALS16',
    #                        aligned_file_tag='DIM_Cloud',
    #                        permitted_file_extension='.asc' )
    #    and use_algorithmus_on_dictionary ({})
    #    and use_algorithmus_on_dictionary ({})
    #    and use_algorithmus_on_dictionary ({}) ):
    #
    #     print ("\n\nAll Clouds successfully processed.")
    # else:
    #     print ("Error. Not all clouds could be processed.")
