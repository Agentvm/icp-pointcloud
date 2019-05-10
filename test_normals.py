import normals
import input_output
import sys
import numpy as np


if __name__ == "__main__":
    print ('\nexecuted with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) )

    #pcl_input_cloud = pcl_load ('clouds/simple_plane.vtk')
    #numpy_cloud = pcl_input_cloud
    #pcl_normals = pcl_compute_normals (pcl_input_cloud )
    #normal_vector, noise, sigma = PCA (pcl_input_cloud.to_array () )

    # ply files
    numpy_cloud_1 = input_output.load_ply_file ('clouds/laserscanning/', 'plane1.ply')    # 3806 points
    #numpy_cloud_2 = input_output.load_ply_file ('clouds/laserscanning/', 'plane2.ply')    # 3806 points

    # las files
    #numpy_cloud_1 = input_output.load_las_file ('clouds/laserscanning/', 'plane1.las')    # 3806 points
    #numpy_cloud_2 = input_output.load_las_file ('clouds/laserscanning/', 'plane2.las')    # 3806 points

    # # simple plane
    # numpy_cloud_1 = np.array ([[1, 0, 0],   # +x
    #                           [2, 1, 0],  # -x
    #                           [0, 1, 0],
    #                           [3, 1.5, 0],
    #                           [-2, 1.5, 0],
    #                           [3, 1, 0.51]])  # +y

    # Simple plane OUTPUT
    # executed with python version 3.5
    # PCA completed in 0.00033354759216308594 seconds.
    #
    # PCA, Cloud 1:
    # normal_vector: [-0.0400096  -0.10263703  0.99391392]
    # sigma: 0.17782724826887134
    # mass_center: [ 1.5   0.5  -0.05]
    #
    # index: [2, 3, 1]
    # index: [0, 2, 1]
    # RANSAC completed in 0.0004112720489501953 seconds.
    #
    # RANSAC, Cloud 1:
    # normal_vector: [0.18814417 0.28221626 0.94072087]
    # consensus_points:
    # [[1.0, 0.0, 0.0], [2.0, 0.0, -0.2], [0.0, 1.0, -0.1], [3.0, 1.0, 0.1]]
    # OUTPUT END

    #                                                                 matlab: 7926 points

    # 1st cloud
    normal_vector, sigma, mass_center = normals.PCA (numpy_cloud_1 )
    print ('PCA, Cloud 1:\nnormal_vector: ' + str(normal_vector ))
    print ('sigma: ' + str(sigma ))
    print ('mass_center: ' + str(mass_center ) + '\n')

    normal_vector, consensus_points = normals.ransac_plane_estimation (numpy_cloud_1, 0.5 )
    print ('RANSAC, Cloud 1:\nnormal_vector: ' + str(normal_vector ))
    print ('consensus_points:\n' + str(consensus_points ) + '\n')

    # # 2nd cloud
    # normal_vector, sigma, mass_center = normals.PCA (numpy_cloud_2 )
    # print ('PCA, Cloud 2:\nnormal_vector: ' + str(normal_vector ))
    # print ('sigma: ' + str(sigma ))
    # print ('mass_center: ' + str(mass_center ))
    #
    # normal_vector, consensus_points = normals.ransac_plane_estimation (numpy_cloud_2, 0.5 )
    # print ('RANSAC, Cloud 2:\nnormal_vector: ' + str(normal_vector ))
    # print ('consensus_points:\n' + str(consensus_points ) + '\n')
