"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

Script for quick testing of the consensus algorithm
"""

# local modules
from modules.np_pointcloud import NumpyPointCloud
from modules import input_output
from modules import consensus
from modules.normals import normalize_vector_array

# basic imports
import numpy as np

# plot imports
import matplotlib.pyplot as plt


def prepare_random_cloud ():

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


def load_example_cloud (folder ):

    # # big cloud
    np_pointcloud = input_output.conditionalized_load(
        'clouds/New Regions/' + folder + '/Yz Houses_als16_reduced_normals_r_1_cleared.asc' )

    corresponding_pointcloud = input_output.conditionalized_load (
        'clouds/New Regions/' + folder + '/Yz Houses_dim16_reduced_normals_r_1_cleared.asc' )

    return np_pointcloud, corresponding_pointcloud


if __name__ == '__main__':

    np.random.seed (1337 )

    # prepare example clouds, random or from file
    np_pointcloud, corresponding_pointcloud, random_offset = prepare_random_cloud ()
    # np_pointcloud, corresponding_pointcloud = load_example_cloud ("Yz_Houses" )

    # reach consensus
    best_alignment, best_consensus_count, best_alignment_consensus_vector = \
        consensus.cubic_cloud_consensus (np_pointcloud,
                                         corresponding_pointcloud,
                                         cubus_length=1,
                                         step=.15,
                                         distance_threshold=0.2,
                                         angle_threshold=32,
                                         algorithm='distance',
                                         plot_title="Laufzeit",
                                         relative_color_scale=False,
                                         save_plot=False )

    # show plot
    # plt.show ()
