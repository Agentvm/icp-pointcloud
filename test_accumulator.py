"""Script for quick testing of the accumulator algorithm"""

# local modules
from modules.np_pointcloud import NumpyPointCloud
from modules import input_output
from modules import conversions
from modules import accumulator
from modules.normals import normalize_vector_array

# basic imports
import numpy as np

# plot imports
# import matplotlib.pyplot as plt


def prepare_random_cloud ():
    """Assembles a np.random.uniform (-10, 10) numpy.ndarray of shape (1000, 6) containing normal vectors and points"""

    # random values
    numpy_cloud = np.random.uniform (-10, 10, size=(1000, 3 ))
    numpy_normals = normalize_vector_array (np.random.uniform (0, 1, size=(1000, 3 )))

    # concat points and normals
    numpy_cloud = np.concatenate ((numpy_cloud, numpy_normals), axis=1)
    print ('numpy_cloud.shape: ' + str (numpy_cloud.shape ))

    # displace
    random_displacement = np.random.uniform ((-0.8, -0.8, -0.8, 0, 0, 0), (0.8, 0.8, 0.8, 0, 0, 0 ))
    corresponding_cloud = numpy_cloud + random_displacement

    # noise
    numpy_cloud = numpy_cloud + np.random.uniform (-0.01, 0.01, size=(1, 6 ))
    corresponding_cloud = corresponding_cloud + np.random.uniform (-0.01, 0.01, size=(1, 6 ))

    # change to NumpyPointCloud
    np_pointcloud = NumpyPointCloud (numpy_cloud, ['X', 'Y', 'Z', 'Nx ', 'Ny ', 'Nz '] )
    corresponding_pointcloud = NumpyPointCloud (corresponding_cloud, ['X', 'Y', 'Z', 'Nx ', 'Ny ', 'Nz '] )

    return np_pointcloud, corresponding_pointcloud, random_displacement


def load_example_cloud_old (folder ):
    """Loads ALS16 and DIM16 clouds from a specified folder below 'clouds/Regions'"""

    # # big cloud
    np_pointcloud = input_output.conditionalized_load(
        'clouds/Regions/' + folder + '/ALS16_Cloud_reduced_normals_cleared.asc' )

    corresponding_pointcloud = input_output.conditionalized_load (
        'clouds/Regions/' + folder + '/DSM_Cloud_reduced_normals.asc' )

    return np_pointcloud, corresponding_pointcloud


def load_example_cloud (folder ):
    """Loads ALS16 and DIM16 clouds from a specified folder below 'clouds/New Regions'"""

    # # big cloud
    np_pointcloud = input_output.conditionalized_load(
        'clouds/New Regions/' + folder + '/Yz Houses_als16_reduced_normals_r_1_cleared.asc' )

    corresponding_pointcloud = input_output.conditionalized_load (
        'clouds/New Regions/' + folder + '/Yz Houses_als16_reduced_normals_r_1_cleared.asc' )

    return np_pointcloud, corresponding_pointcloud


if __name__ == '__main__':

    np.random.seed (1337 )
    random_offset = None

    title = "New_Regions_YZ_Houses"
    sigma = False
    filter_sigma_only_on_corresponding_cloud = False
    sigma_value = 0.2
    borders = True

    # prepare example clouds, random or from file
    # np_pointcloud, corresponding_pointcloud, random_offset = prepare_random_cloud ()
    np_pointcloud, corresponding_pointcloud = load_example_cloud_old ("Yz Houses" )
    np_pointcloud, corresponding_pointcloud = load_example_cloud ("Yz_Houses" )

    if (sigma ):
        title += "_sigma_"

        if (filter_sigma_only_on_corresponding_cloud ):
            title += "corr_only_"
        else:
            np_pointcloud.points = np_pointcloud.points[np_pointcloud.get_fields ("Sigma") < sigma_value]
        corresponding_pointcloud.points = corresponding_pointcloud.points[
                                                corresponding_pointcloud.get_fields ("Sigma") < sigma_value]
        title += str(sigma_value )

    if (borders ):
        title += "_borders"
        corresponding_pointcloud.points = conversions.prune_cloud_borders (corresponding_pointcloud.points, 3.0 )

    # reach consensus
    best_alignment, best_consensus_count, = accumulator.spheric_cloud_consensus (np_pointcloud,
                                                                                 corresponding_pointcloud,
                                                                                 accumulator_radius=1.0,
                                                                                 grid_size=0.1,
                                                                                 distance_threshold=None,
                                                                                 display_plot=False,
                                                                                 save_plot=True,
                                                                                 relative_color_scale=True,
                                                                                 plot_title=title )

    print ("best_alignment: \t\t" + str(best_alignment ))
    if (random_offset is not None ):
        print ("Random Offset: \t\t\t" + str(random_offset ))
    else:
        #print ("Point Picking Offset Xy Tower: (-0.82777023,  0.16250610,  0.19129372)")
        print ("Point Picking Offset Yz Houses: [0.31462097, -0.01929474, -0.03573704]")

    # show plot
    #plt.show ()


# # Gute Werte:

# Bsp Random

# # 1 combined
# Starting Cubic Cloud Consensus
# distance_threshold: 0.2
# angle_threshold: 0.6108652381980153
# cubus_length: 2
# step: 0.2

# # 2 combined
# Starting combined Cubic Cloud Consensus
# distance_threshold: 0.3
# angle_threshold: 0.08726646259971647
# cubus_length: 2
# step: 0.15

# Starting Cubic Cloud Consensus
# distance_threshold: 0.2
# angle_threshold: 0.08726646259971647
# cubus_length: 2
# step: 0.2
