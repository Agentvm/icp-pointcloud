"""Computes C2C_absolute_distances on a cloud compared to another cloud. Mimics CloudCompare behaviour"""


# local modules
from modules import input_output
from queue_alignment_algorithms import get_reference_data_paths

# basic imports
import os

# advanced functionality
import scipy.spatial


def cloud2cloud (reference_pointcloud, compared_pointcloud ):
    ''' Computes field C2C_absolute_distances on compared cloud '''

    # make a tree an get a list of distances to the nearest neigbor and his index (which is not needed)
    # but only take the x,y,z fields into consideration (reference_cloud[:, 0:3])
    scipy_kdtree = scipy.spatial.cKDTree (reference_pointcloud.get_xyz_coordinates () )

    # query the three, but only take the x,y,z fields into consideration
    c2c_distances, indices = scipy_kdtree.query (compared_pointcloud.get_xyz_coordinates (), k=1 )

    # add a new field containing the distance to the nearest neighbor of each point to the compared_cloud and return it
    return compared_pointcloud.add_fields (c2c_distances.reshape (-1, 1), ["C2C_absolute_distances"] )


def use_c2c_on_dictionary (reference_dictionary_name, descriptive_name ):
    '''
    Takes the name of a dictionary in data/ and returns a numpy cloud with Cloud2Cloud Distance column computed
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
            field_labels_list += ["C2C_absolute_distances"]

            # make path
            _, reference_file_name = input_output.get_folder_and_file_name (reference_cloud_path)
            _, aligned_file_name = input_output.get_folder_and_file_name (aligned_cloud_path)
            base_path = os.path.dirname(aligned_cloud_path) + "/Results/"
            save_path = base_path + aligned_file_name + "to_" + reference_file_name + descriptive_name + ".asc"

            # save as result
            input_output.save_ascii_file (updated_aligned_cloud, field_labels_list, save_path )

    return True


if __name__ == '__main__':

    # no translation, original clouds
    print ("\n\nComputing C2C_absolute_distances for each cloud pair in no_translations returns: "
           + str(use_c2c_on_dictionary ("no_translations_dict", "no_translations" ) ))

    # reference_translations, cloud compare point picking
    print ("\n\nComputing C2C_absolute_distances "
           + "for each cloud pair in reference_translations returns: "
           + str(use_c2c_on_dictionary ("reference_translations_dict", "point_clicking" )))

    # icp translations
    print ("\n\nComputing C2C_absolute_distances "
           + "for each cloud pair in icp_translations returns: "
           + str(use_c2c_on_dictionary ("icp_translations_dict", "icp" )))

    # best consensus (point distance version) translations
    # print ("\n\nComputing C2C_absolute_distances "
    #        + "for each cloud pair in distance_consensus_translations returns: "
    #        + str(use_c2c_on_dictionary ("distance_consensus_translations_dict", "distance_consensus" ) ))
