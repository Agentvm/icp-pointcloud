from modules import input_output
from modules import icp
from modules import conversions
from data import reference_transformations
from collections import OrderedDict
import random


def do_icp (full_path_of_reference_cloud, full_path_of_aligned_cloud ):

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


def compare_icp_results (icp_results, print_csv=False ):

    reference_dict = reference_transformations.translations

    # # sort the results
    # create a list of tuples from reference and aligned cloud file paths
    unssorted_results = []
    for paths in icp_results:
        unssorted_results.append ((paths, icp_results[paths]) )

    sorted_results = sorted(unssorted_results )

    for paths, translation_value in sorted_results:

        if (paths in reference_dict ):

            # disassemble the key
            reference_path, aligned_path = paths

            # folder should be the the same
            folder, reference_file_name = input_output.get_folder_and_file_name (reference_path)
            folder, aligned_file_name = input_output.get_folder_and_file_name (aligned_path)

            # unpack values
            ref_translation, ref_mse = reference_dict [paths]
            icp_translation, icp_mse = translation_value

            if (print_csv):
                # print comparison
                print ('\n' + folder + "/"
                       + " " + reference_file_name
                       + " " + aligned_file_name
                             + ';{: .8f}'.format(ref_translation[0]) + ';{: .8f}'.format(icp_translation[0])
                       + '\n;{: .8f}'.format(ref_translation[1]) + ';{: .8f}'.format(icp_translation[1])
                       + '\n;{: .8f}'.format(ref_translation[2]) + ';{: .8f}'.format(icp_translation[2])
                       + '\n;{: .8f}'.format(ref_mse) + ';=MAX({: .8f}'.format(icp_mse[0])
                            + ',{: .8f}'.format(icp_mse[1])
                            + ',{: .8f}) '.format(icp_mse[2]))
            else:
                # print comparison
                print ('\n' + folder + "/"
                       + "\nreference cloud:\t" + reference_file_name
                       + "\naligned cloud:\t\t" + aligned_file_name
                       + "\n\tdata alignment:\t" + '({: .8f}, '.format(ref_translation[0])
                                                 + '{: .8f}, '.format(ref_translation[1])
                                                 + '{: .8f}), '.format(ref_translation[2])
                                                 + ' {: .8f}, '.format(ref_mse)
                       + "\n\ticp alignment:\t" + '({: .8f}, '.format(icp_translation[0])
                                                + '{: .8f}, '.format(icp_translation[1])
                                                + '{: .8f}), '.format(icp_translation[2])
                                                + '({: .8f}, '.format(icp_mse[0])
                                                + '{: .8f}, '.format(icp_mse[1])
                                                + '{: .8f}) '.format(icp_mse[2]))


def use_icp_on_dictionary (icp_paths_dictionary ):
    '''
    Uses a dictionary of reference cloud file_paths as keys
    and a list of corresponding aligned cloud file_paths as values
    '''

    # before start, check if files exist
    for key in icp_paths_dictionary:
        if (input_output.check_for_file (key ) is False):
            print ("File " + key + " was not found. Aborting.")
            return False
        for aligned_cloud_path in icp_paths_dictionary[key]:
            if (input_output.check_for_file (aligned_cloud_path ) is False):
                print ("File " + aligned_cloud_path + " was not found. Aborting.")
                return False

    icp_results = {}    # dictionary

    # create a list of tuples from reference and aligned cloud file paths
    for reference_cloud_path in icp_paths_dictionary:
        for aligned_cloud_path in icp_paths_dictionary[reference_cloud_path]:   # multiple aligned clouds possible
            # do the icp
            icp_results.update (do_icp (reference_cloud_path, aligned_cloud_path ))

    # prints the values computed along with the ground truth in the dictionary
    compare_icp_results (icp_results )

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


def get_icp_data_paths ():
    '''
    Reads reference_transformations.translations to get all transformations currently saved and returns them in a
    dictionary that can be directly used with use_icp_on_dictionary()
    '''
    dict = {}
    for key in reference_transformations.translations:

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
    # print ("\n\nComputing ICP for each cloud pair in reference_transformations.translations returns: "
    #        + str(use_icp_on_dictionary (get_icp_data_paths () )))

    compare_icp_results (do_icp ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
                                 'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc' ), print_csv=True)

    # # tests

    # print_files_dict_of_folder(folder)

    # print (get_icp_data_paths ())

    # if (use_icp_on_folder ('clouds/Regions/',
    #                        reference_file_tag='ALS16',
    #                        aligned_file_tag='DIM_Cloud',
    #                        permitted_file_extension='.asc' )
    #    and use_icp_on_dictionary ({})
    #    and use_icp_on_dictionary ({})
    #    and use_icp_on_dictionary ({}) ):
    #
    #     print ("\n\nAll Clouds successfully processed.")
    # else:
    #     print ("Error. Not all clouds could be processed.")
