import time
import numpy as np
import random
import pcl
from math import ceil, sqrt


def randomPlaneEstimationII (input_numpy_cloud, draw_radius ):
    '''
    Uses 1 random point and two points in the neighborhood of the first one
    to estimate Plane Parameters. It can therefore find a plane in a scene with multiple planes

    Input:
        input_numpy_cloud (np.array):   numpy array with data points, only the first 3 colums are used
    Output:

    '''

    print ("\n\nrandomPlaneEstimationII is curently not implemented. Aborting. \n\n")
    return 0

    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = input_numpy_cloud[:, 0:3].copy ()     # copying takes roughly 0.000558 seconds per 1000 points
    cloud_size = numpy_cloud.shape[0]

    # convert to pcl cloud to use kdtree
    pcl_cloud = pcl.PointCloud(numpy_cloud)
    tree = pcl_cloud.make_kdtree_flann()

    #indices, sqr_distances = kd.nearest_k_search_for_cloud(pc_2, 1)

    # estimate the points
    # first point is a random draw
    index_1 = random.randint(cloud_size, 1 )
    point_1 = numpy_cloud [index_1, :]

    # search the kdtree and find the 150 nearest neighbors

    # ### PCL-PYTHON CODE: ###############################################
    #[neighbors, distances] = tree.nearest_k_search_for_cloud (? )

    # nearest_k_search_for_cloud(self, BasePointCloud pc, int k=1)
    #
    #     Find the k nearest neighbours and squared distances for all points in the pointcloud.
    #     Results are in ndarrays, size (pc.size, k) Returns: (k_indices, k_sqr_distances)
    #
    # nearest_k_search_for_point(self, BasePointCloud pc, int index, int k=1)
    #
    #     Find the k nearest neighbours and squared distances for the point at pc[index].
    #     Results are in ndarrays, size (k) Returns: (k_indices, k_sqr_distances)

    # ### MATLAB CODE: ##########################################################
    #[neighbors, distances] = knnsearch (tree, p1, 'K', 150 );

    # # Transposing
    # neighbors = neighbors'
    # distances = distances'
    #
    # # --------------------------------------------------------------------
    # # find neighbors with distance greater than specified threshold
    # nearest_acceptable_neighbor = 0
    # farthest_acceptable_neighbor = 0
    # for i = 1:length(distances)
    #    if distances(i) > draw_radius
    #        if nearest_acceptable_neighbor == 0
    #             nearest_acceptable_neighbor = i
    #        end
    #        if distances(i) > draw_radius * 1.7
    #             farthest_acceptable_neighbor = i
    #             break
    #
    # # find random neigbors with distance in threshold
    # if nearest_acceptable_neighbor ~= 0 && farthest_acceptable_neighbor ~= 0:
    #     index2 = randi([nearest_acceptable_neighbor, farthest_acceptable_neighbor], 1, 2)
    # else:
    #     fprintf ('WARNING, randomPlaneEstimationII: No neighbors could ')
    #     fprintf ('be determined in close proximity to the first draw. ')
    #     fprintf ('Using random point.')
    #     index2 = randi(length(neighbors), 1, 2 )
    # end
    #
    # p2 = c(neighbors(index2(1)), : )    # point 2
    #
    # # --------------------------------------------------------------------
    # # search the kdtree and find the 150 nearest neighbors for point 2
    # [neighbors, distances] = knnsearch (tree, p2, 'K', 150 )
    # neighbors = neighbors'
    # distances = distances'
    #
    # # find neighbors with distance greater than specified threshold
    # nearest_acceptable_neighbor = 0
    # farthest_acceptable_neighbor = 0
    # for i = 1:length(distances)
    #    if distances(i) > draw_radius
    #        if nearest_acceptable_neighbor == 0
    #             nearest_acceptable_neighbor = i
    #        end
    #        if distances(i) > draw_radius * 1.7
    #             farthest_acceptable_neighbor = i
    #             break
    #
    # # find random neigbors with distance in threshold
    # if nearest_acceptable_neighbor ~= 0 && farthest_acceptable_neighbor ~= 0
    #     index3 = randi([nearest_acceptable_neighbor, farthest_acceptable_neighbor ], 1, 2)
    # else
    #     fprintf ('WARNING, randomPlaneEstimationII: A neighbor could not ')
    #     fprintf ('be determined in close proximity to the second draw. ')
    #     fprintf ('Using random point.')
    #     index3 = randi(length(neighbors), 1, 2)
    # end
    #
    # p3 = c(neighbors(index3(2)), : )
    #
    # # DEBUG show the points
    # #scatter3(p1(1,1), p1(1,2), p1(1,3), 20, 'o');
    # #scatter3(p2(1,1), p2(1,2), p2(1,3), 20, '+');
    # #scatter3(p3(1,1), p3(1,2), p3(1,3), 20, '*');
    #
    # # compute the plane parameters
    # n = transpose (cross((p2 - p1 ), (p3 - p1 )))
    # n = n / norm(n )
    # d = -(n(1 )*p1(1 ) + n(2 )*p1(2 ) + n(3 )*p1(3 ))
    #
    return 0


def pcl_compute_normals (pcl_cloud):
    '''
    Computes normals for a pcl cloud

    Input:
        pcl_cloud (pcl.PointCloud):  Any pcl cloud

    Output:
        normals (?):    ..?
    '''

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
    Takes a vector and returns it's unit vector
    '''

    # check if vector is a matrix
    if (len (vector.shape ) > 1 ):
        print ("In normalize_vector: Vector is out of shape. Returning input vector.")
        return vector

    if (np.sum (vector ) == 0):
        print ("In normalize_vector: Vector is 0. Returning input vector.")
        return vector

    # vector_magnitude = 0
    # for value in vector:
    #     vector_magnitude = vector_magnitude + np.float_power (value, 2 )
    # vector_magnitude = sqrt (vector_magnitude )
    #
    # return vector / vector_magnitude

    return vector / np.linalg.norm(vector)


# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::
#
#             >>> angle_between((1, 0, 0), (0, 1, 0))
#             1.5707963267948966
#             >>> angle_between((1, 0, 0), (1, 0, 0))
#             0.0
#             >>> angle_between((1, 0, 0), (-1, 0, 0))
#             3.141592653589793
#     """
#     v1_u = normalize_vector (v1 )
#     v2_u = normalize_vector (v2 )
#     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def eigenvalue_decomposition (a_t_a_matrix ):
    '''
    Uses np.linalg.eig () to decompose a 3x3 matrix.
    Returns normal vector and smallest eigenvalue.
    '''
    # get eigenvalues and -vectors from ATA matrix
    eigenvalues = np.zeros (a_t_a_matrix.shape[0] )
    eigenvectors = np.zeros ((a_t_a_matrix.shape[0], a_t_a_matrix.shape[0] ))
    evals, evecs = np.linalg.eig (a_t_a_matrix )

    # sort them
    indices = np.argsort (-evals )  # reverse sort: greatest numbers first
    for loop_count, index in enumerate(indices ):
        eigenvalues[loop_count] = evals[index]
        eigenvectors[:, loop_count] = evecs[:, index]

    # get the normal vector, normalize it and if it's turned to the ground, turn it around
    normal_vector = normalize_vector (eigenvectors[:, -1] )     # the last (smallest) vector is the normal vector
    if (normal_vector[2] < 0):
        normal_vector = normal_vector * -1

    return normal_vector, eigenvalues[-1]


def build_covariance_matrix (input_numpy_cloud, reduce_by_center_of_mass=True ):

    # build a sum over all points
    sum_xyz = np.sum (input_numpy_cloud, axis=0 )

    # and normalize it to get center of mass
    mass_center = sum_xyz / input_numpy_cloud.shape[0]

    # reduce point cloud by center of mass
    if (reduce_by_center_of_mass ):
        numpy_cloud_reduced = np.subtract (input_numpy_cloud[:, 0:3], mass_center )
    else:
        numpy_cloud_reduced = input_numpy_cloud

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

    return a_transposed_a, mass_center


def PCA (input_numpy_cloud ):
    """
    From the points of the given point cloud, this function derives a plane defined by a normal vector and the noise of
    the given point cloud in respect to this plane.

    Input:
        input_numpy_cloud (np.array):   numpy array with data points, only the first 3 colums are used

    Output:
        normal_vector ([1x3] np.array): The normal vector of the computed plane
        sigma (float):                  The noise as given by the smallest eigenvalue, normalized by number of points
        mass_center ([1x3] np.array):   Centre of mass
    """

    start_time = time.time()
    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = input_numpy_cloud[:, 0:3].copy ()     # copying takes roughly 0.000558 seconds per 1000 points
    cloud_size = numpy_cloud.shape[0]

    # get covariance matrix
    a_transposed_a, mass_center = build_covariance_matrix (numpy_cloud )

    # get normal vector and smallest eigenvector
    normal_vector, smallest_eigenvalue = eigenvalue_decomposition (a_transposed_a )

    # get the noise and normalize it
    noise = smallest_eigenvalue
    sigma = sqrt(noise/(cloud_size - 3) )

    print ('PCA completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return normal_vector, sigma, mass_center


def random_plane_estimation (numpy_cloud ):
    '''
    Uses 3 random points to estimate Plane Parameters

    Output:
        normal_vector :
        plane_parameter_d :
    '''

    # get 3 random indices
    idx_1, idx_2, idx_3 = random.sample(range(0, numpy_cloud.shape[0] ), 3 )

    point_1 = numpy_cloud [idx_1, :]
    point_2 = numpy_cloud [idx_2, :]
    point_3 = numpy_cloud [idx_3, :]

    # get the normal vector, normalize it and if it's turned to the ground, turn it around
    normal_vector = normalize_vector (np.cross((point_2 - point_1), (point_3 - point_1 )))
    plane_parameter_d = -(normal_vector[0] * point_1[0]
                          + normal_vector[1] * point_1[1]
                          + normal_vector[2] * point_1[2] )
    if (normal_vector[2] < 0):      # z component
        normal_vector = normal_vector * -1

    return normal_vector, plane_parameter_d


def ransac_plane_estimation (input_numpy_cloud, threshold, w = .9, z = 0.95 ):
    '''
    Uses Ransac with the probability parameters w and z to estimate a valid plane in given cloud.
    Uses distance from plane compared to given threshold to determine the consensus set.
    Returns points and point indices of the detected plane.

    Input:
        input_numpy_cloud (np.array):   Input cloud
        treshold (float, in m):         Points closer to the plane than this value are counted as inliers
        w (float between 0 and 1):      probability that any observation belongs to the model
        z (float between 0 and 1):      desired probability that the model is found
    Output:
        not sure yet
    '''

    # measure time
    start_time = time.time ()

    # variables
    consensus_points = []  # points matching the cloud
    normal_vector_list = []
    current_consensus = 0
    best_consensus = 0

    # determine probabilities and number of draws
    b = np.float_power(w, 3 )   # probability that all three observations belong to the model
    k = ceil(np.log(1-z ) / np.log(1-b ))   # number of draws

    # copy cloud
    numpy_cloud = input_numpy_cloud[:, 0:3].copy ()

    # iterate: draw 3 points k times
    for i in range (1, k):

        # reset
        current_consensus = 0
        points = []

        # estimate a plane with 3 random points
        [normal_vector, d] = random_plane_estimation (numpy_cloud )
        normal_vector_list.append (normal_vector )

        # plane paramters are elements of the normal vector
        a = normal_vector[0]
        b = normal_vector[1]
        c = normal_vector[2]

        # computing distances of every point from plane for consensus set
        for point in numpy_cloud:
            dist = ((a * point[0]
                   + b * point[1]
                   + c * point[2]
                     + d )
                    / (sqrt (np.float_power (a, 2 ) + np.float_power (b, 2 ) + np.float_power (c, 2 ))))

            # threshold match?
            if (dist < threshold ):
                current_consensus = current_consensus + 1     # counting consensus
                points.append (point.tolist ())     # this might be slowing the code down

        # is the current consensus match higher than the previous ones?
        if (current_consensus > best_consensus ):
            consensus_points = points
            best_consensus = current_consensus    # keep best consensus set

    # print time
    print('RANSAC completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return normal_vector, consensus_points
