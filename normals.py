import time
import sys
import numpy as np
import input_output


def pcl_compute_normals (pcl_cloud):
    start_time = time.time()
    # Search
    searching_neighbour = {'knn_search': 25}
    feat = pcl_cloud.make_NormalEstimation()

    if 'range_search' in searching_neighbour.keys():
        # Range search
        searching_para = searching_neighbour['range_search'] if searching_neighbour['range_search'] > 0 else 0.1
        feat.setRadiusSearch(searching_para)
    elif 'knn_search' in searching_neighbour.keys():
        # kNN search
        searching_para = int(searching_neighbour['knn_search']) if int(searching_neighbour['knn_search']) > 5 else 20
        tree = pcl_cloud.make_kdtree()
        feat.set_SearchMethod(tree)
        feat.set_KSearch(searching_para)
    else:
        print('Define researching method does not support')

    normals = feat.compute()
    print('Computed normal vectors in ' + str(time.time() - start_time) + ' seconds' )

    return normals


def PCA (input_numpy_cloud ):
    start_time = time.time()
    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = input_numpy_cloud.copy ()
    numpy_cloud = numpy_cloud [:, 0:3]

    # build a sum over all points
    sum_xyz = np.array ((0, 0, 0 ))
    for i, point in enumerate (numpy_cloud ):
        sum_xyz[0] = sum_xyz[0] + point[0]
        sum_xyz[1] = sum_xyz[1] + point[1]
        sum_xyz[2] = sum_xyz[2] + point[2]

    print ('sum_xyz: ' + str(sum_xyz ))

    # and normalize it to get center of mass
    sum_xyz = sum_xyz / numpy_cloud.size

    print ('sum_xyz_norm: ' + str(sum_xyz ))

    # reduce point cloud by center of mass
    numpy_cloud_reduced = np.subtract (numpy_cloud[:, 0:3], sum_xyz )

    # build ATA matrix
    a_transposed_a = np.zeros ((3, 3 ))

    for point in numpy_cloud_reduced:
        a_transposed_a[0, 0] = a_transposed_a[0, 0] + np.float_power(point[0], 2 )
        a_transposed_a[0, 1] = a_transposed_a[0, 1] + point[0] * point[1]
        a_transposed_a[0, 2] = a_transposed_a[0, 2] + point[0] * point[2]

        a_transposed_a[1, 0] = a_transposed_a[1, 0] + point[0] * point[1]
        a_transposed_a[1, 1] = a_transposed_a[1, 1] + np.float_power(point[1], 2 )
        a_transposed_a[1, 2] = a_transposed_a[1, 2] + point[1] * point[2]

        a_transposed_a[2, 0] = a_transposed_a[2, 0] + point[0] * point[2]
        a_transposed_a[2, 1] = a_transposed_a[2, 1] + point[2] * point[1]
        a_transposed_a[2, 2] = a_transposed_a[2, 2] + np.float_power(point[2], 2 )

    # get eigenvalues and -vectors from ATA matrix
    eigenvalues = np.zeros (a_transposed_a.shape[0] )
    eigenvectors = np.zeros ((a_transposed_a.shape[0], a_transposed_a.shape[0] ))
    evals, evecs = np.linalg.eig (a_transposed_a )

    # sort them
    indices = np.argsort (-evals )  # reverse sort: greatest numbers first
    for loop_count, index in enumerate(indices ):
        eigenvalues[loop_count] = evals[index]
        eigenvectors[:, loop_count] = evecs[:, index]

    noise = eigenvalues[-1]
    normal_vector = eigenvectors[:, -1]
    sigma = np.sqrt(noise/(numpy_cloud.shape[0] - 3) )
    print('PCA completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return normal_vector, noise, sigma


if __name__ == "__main__":
    print ('\nexecuted with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) )

    #pcl_input_cloud = pcl_load ('clouds/simple_plane.vtk')
    #numpy_cloud = pcl_input_cloud
    #pcl_normals = pcl_compute_normals (pcl_input_cloud )
    #normal_vector, noise, sigma = PCA (pcl_input_cloud.to_array () )

    numpy_cloud_1 = input_output.load_ply_file ('clouds/', 'plane1.ply')    # 23778 points
    #numpy_cloud_2 = input_output.load_ply_file ('clouds/', 'plane2.ply')

    numpy_cloud_1  = input_output.load_las_file ('clouds/', 'plane1.las')  # 31704 points
    #numpy_cloud_2 = input_output.load_las_file ('clouds/', 'plane2.las')

    #                                                                 matlab: 7926 points

    normal_vector, noise, sigma = PCA (numpy_cloud_1 )
    print ('Cloud 1:\nnormal_vector: ' + str(normal_vector ))
    print ('noise: ' + str(noise ))
    print ('sigma: ' + str(sigma ) + '\n')

    # normal_vector, noise, sigma = PCA (numpy_cloud_2 )
    # print ('Cloud 2:\nnormal_vector: ' + str(normal_vector ))
    # print ('noise: ' + str(noise ))
    # print ('sigma: ' + str(sigma ))


# normal_vector: [0.95553649 0.29451123 0.0145996 ]     # as .ply
# normal_vector: [ 0.9582111   0.28454521 -0.02941945]  # as .las

# noise: 2.0739264969012923     # as .ply
# noise: 8.073087439041888      # as .las

# sigma: 0.016179006599999542   # as .ply
# sigma: 0.03192089063208706    # as .las
