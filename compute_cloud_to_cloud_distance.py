from modules import input_output
# from modules import icp
# from modules import conversions
# from modules import consensus
#from data import transformations
# from collections import OrderedDict
import numpy as np
import os
#import sklearn.neighbors    # kdtree
import scipy.spatial
# import random


def cloud2cloud (reference_cloud, compared_cloud ):
    ''' Computes field C2C_absolute_distances on compared cloud '''

    # make a tree an get a list of distances to the nearest neigbor and his index (which is not needed)
    # but only take the x,y,z fields into consideration (reference_cloud[:, 0:3])
    scipy_kdtree = scipy.spatial.cKDTree (reference_cloud[:, 0:3] )
    #tree = sklearn.neighbors.kd_tree.KDTree (reference_cloud[:, 0:3], leaf_size=40, metric='euclidean' )

    # query the three, but only take the x,y,z fields into consideration (compared_cloud[:, 0:3])
    #output = scipy_kdtree.query (compared_cloud[:, 0:3], k=1, return_distance=True )
    distances, indices = scipy_kdtree.query (compared_cloud[:, 0:3], k=1 )

    # Make a list out of the values of the first numpy array,
    # Take only the distances ([0]), not the neighbor indices ([1])
    #distances = list(itertools.chain(*output[0] ))
    #neighbor_indices = list(itertools.chain(*output[1] ))

    # print ("\nlen(distances_pre): " + str(len(distances_pre)))
    # print (distances_pre)
    #
    # print ("compared_cloud.shape[0]: " + str(compared_cloud.shape[0]))
    # print ("reference_cloud.shape[0]: " + str(reference_cloud.shape[0]))
    # print (distances)

    # add a new field containing the distance to the nearest neighbor of each point to the compared_cloud and return it
    return np.concatenate ((compared_cloud, distances.reshape (-1, 1)), axis=1 )


def use_c2c_on_dictionary (reference_dictionary_name, descriptive_name ):
    '''
    Supply a dictionary of data/transformations.py and produce results with C2C column computed
    '''

    # refactor, iterate through reference_dictionary instead
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

    # create a list of tuples from reference and aligned cloud file paths
    for reference_cloud_path in file_paths_dictionary:
        for aligned_cloud_path in file_paths_dictionary[reference_cloud_path]:   # multiple aligned clouds possible

            # load clouds
            reference_cloud = input_output.load_ascii_file (reference_cloud_path )
            aligned_cloud, field_labels_list = input_output.conditionalized_load(aligned_cloud_path )

            # compute cloud to cloud distance on the translated cloud and update field_labels_list
            # get the translation out of the reference_dictionary
            key = (reference_cloud_path, aligned_cloud_path)    # prepare dictionary key
            alignment = [0] * aligned_cloud.shape[1]            # prepare shape of list for addition
            alignment[0:3], mse = reference_dictionary[key]     # extract alignment

            # apply alignment and update aligned_cloud with C2C_absolute_distances field
            updated_aligned_cloud = cloud2cloud(reference_cloud, aligned_cloud + alignment)
            field_labels_list = field_labels_list + ["C2C_absolute_distances"]

            # make path
            _, reference_file_name = input_output.get_folder_and_file_name (reference_cloud_path)
            _, aligned_file_name = input_output.get_folder_and_file_name (aligned_cloud_path)
            base_path = os.path.dirname(aligned_cloud_path) + "/Results/"
            save_path = base_path + aligned_file_name + "to_" + reference_file_name + descriptive_name + ".asc"

            # save as result
            input_output.save_ascii_file (updated_aligned_cloud, field_labels_list, save_path )

    return True


def get_reference_data_paths (input_dictionary ):
    '''
    Reads transformations.reference_translations to get all transformations currently saved and returns them in a
    dictionary that can be directly used with use_algorithmus_on_dictionary()
    '''
    dict = {}
    for key in input_dictionary:

        reference_path, aligned_path = key

        if (dict.__contains__ (reference_path )):
            dict[reference_path].append (aligned_path )
        else:
            dict.update ({reference_path: [aligned_path]} )

    return dict


if __name__ == '__main__':

    # no translation, original clouds
    print ("\n\nComputing C2C_absolute_distances for each cloud pair in transformations.no_translations returns: "
           + str(use_c2c_on_dictionary ("no_translations_dict", "no_translations" ) ))

    # reference_translations, cloud compare point picking
    print ("\n\nComputing C2C_absolute_distances "
           + "for each cloud pair in transformations.reference_translations returns: "
           + str(use_c2c_on_dictionary ("reference_translations_dict", "point_clicking" )))

    # icp translations
    print ("\n\nComputing C2C_absolute_distances "
           + "for each cloud pair in transformations.icp_translations returns: "
           + str(use_c2c_on_dictionary ("icp_translations_dict", "icp" )))

    # best consensus (point distance version) translations
    # print ("\n\nComputing C2C_absolute_distances "
    #        + "for each cloud pair in transformations.distance_consensus_translations returns: "
    #        + str(use_c2c_on_dictionary ("distance_consensus_translations_dict", "distance_consensus" ) ))
