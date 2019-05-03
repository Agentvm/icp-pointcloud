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


def normalize_vector (vector ):
    '''

    '''
    # check if vector is a matrix
    if (len (vector.shape ) > 1 ):
        print ("In normalize_vector: Vector is out of shape.")
        return vector

    vector_magnitude = 0
    for value in vector:
        vector_magnitude = vector_magnitude + np.float_power (value, 2 )
    vector_magnitude = np.sqrt (vector_magnitude )

    return vector / vector_magnitude


def PCA (input_numpy_cloud ):
    """
    From the points of the given point cloud, this function derives a plane defined by a normal vector and the noise of
    the given point cloud in respect to this plane.

    Input:
        input_numpy_cloud: numpy array with data points, only the first 3 colums are used

    Output:
        normal_vector:  The normal vector of the computed plane
        sigma:          The noise as given by the smallest eigenvalue, normalized by number of points
        mass_center:        Centre of mass
    """

    start_time = time.time()
    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = input_numpy_cloud[:, 0:3].copy ()     # copying takes roughly 0.000558 seconds per 1000 points
    cloud_size = numpy_cloud.shape[0]

    # build a sum over all points
    sum_xyz = np.array ((0, 0, 0 ), float)
    for i, point in enumerate (numpy_cloud ):
        sum_xyz[0] = sum_xyz[0] + point[0]
        sum_xyz[1] = sum_xyz[1] + point[1]
        sum_xyz[2] = sum_xyz[2] + point[2]

    # and normalize it to get center of mass
    mass_center = sum_xyz / cloud_size

    # reduce point cloud by center of mass
    numpy_cloud_reduced = np.subtract (numpy_cloud[:, 0:3], mass_center )

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

    # get the normal vector, normalize it and if it's turned to the ground, turn it around
    normal_vector = normalize_vector (eigenvectors[:, -1] )     # the last (smallest) vector is the normal vector
    if (normal_vector[2] < 0):
        normal_vector = normal_vector * -1

    # get the noise and normalize it
    noise = eigenvalues[-1]
    sigma = np.sqrt(noise/(cloud_size - 3) )

    print('PCA completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return normal_vector, sigma, mass_center


if __name__ == "__main__":
    print ('\nexecuted with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) )

    #pcl_input_cloud = pcl_load ('clouds/simple_plane.vtk')
    #numpy_cloud = pcl_input_cloud
    #pcl_normals = pcl_compute_normals (pcl_input_cloud )
    #normal_vector, noise, sigma = PCA (pcl_input_cloud.to_array () )

    #numpy_cloud_1 = input_output.load_ply_file ('clouds/laserscanning/', 'plane1.ply')    # 3806 points
    #numpy_cloud_2 = input_output.load_ply_file ('clouds/laserscanning/', 'plane2.ply')    # 3806 points

    numpy_cloud_1 = input_output.load_las_file ('clouds/laserscanning/', 'plane1.las')    # 3806 points
    numpy_cloud_2 = input_output.load_las_file ('clouds/laserscanning/', 'plane2.las')    # 3806 points

    #                                                                 matlab: 7926 points

    #numpy_cloud_1 = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1.1] ))

    normal_vector, sigma, mass_center = PCA (numpy_cloud_1 )
    print ('Cloud 1:\nnormal_vector: ' + str(normal_vector ))
    print ('sigma: ' + str(sigma ))
    print ('mass_center: ' + str(mass_center ) + '\n')

    normal_vector, sigma, mass_center = PCA (numpy_cloud_2 )
    print ('Cloud 2:\nnormal_vector: ' + str(normal_vector ))
    print ('sigma: ' + str(sigma ))
    print ('mass_center: ' + str(mass_center ))


# normal_vector: [0.95553649 0.29451123 0.0145996 ]     # as .ply
# normal_vector: [ 0.9582111   0.28454521 -0.02941945]  # as .las

# noise: 2.0739264969012923     # as .ply
# noise: 8.073087439041888      # as .las

# sigma: 0.016179006599999542   # as .ply
# sigma: 0.03192089063208706    # as .las
