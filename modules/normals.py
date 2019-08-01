"""
Offers Principal Component Analysis and RANSAC algorithms for normal vector computation.
"""


# basic imports
import time
import numpy as np
import random
from math import ceil, sqrt


def normalize_vector (vector ):
    '''
    Takes a vector and returns it's unit vector
    '''

    if (np.sum (vector ) == 0):
        #print ("In normalize_vector: Vector is 0. Returning input vector.")
        return vector

    return vector / np.linalg.norm(vector)


def normalize_vector_array (vector_array ):
    norms = np.apply_along_axis(np.linalg.norm, 1, vector_array )
    norms = np.where (norms == 0, 1, norms )    # these filtered values belong to arrays that already are normalized

    return vector_array / norms.reshape (-1, 1 )


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


def build_covariance_matrix (numpy_cloud, reduce_by_center_of_mass=True ):

    # build a sum over all points
    sum_xyz = np.sum (numpy_cloud, axis=0 )

    # and normalize it to get center of mass
    mass_center = sum_xyz / numpy_cloud.shape[0]

    # reduce point cloud by center of mass
    if (reduce_by_center_of_mass ):
        numpy_cloud_reduced = np.subtract (numpy_cloud[:, 0:3], mass_center )
    else:
        numpy_cloud_reduced = numpy_cloud.copy ()

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


def PCA (numpy_cloud ):
    """
    From the points of the given point cloud, this function derives a plane defined by a normal vector and the noise of
    the given point cloud in respect to this plane.

    Input:
        numpy_cloud (np.ndarray):   numpy array with data points, only the first 3 colums are used

    Output:
        normal_vector ([1, 3] np.array): The normal vector of the computed plane
        sigma (float):                  The noise as given by the smallest eigenvalue, normalized by number of points
        mass_center ([1, 3] np.array):   Centre of mass
    """

    start_time = time.time()

    # abort, if there are no points
    if (numpy_cloud.shape[0] == 0):
        #print ("In normals.py, in PCA: The input array is empty. Returning a null vector and high sigma")
        return np.array ((0, 0, 0)), 100.0, np.array ((0, 0, 0)), time.time() - start_time

    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = numpy_cloud[:, :3].copy ()     # copying takes roughly 0.000558 seconds per 1000 points
    cloud_size = numpy_cloud.shape[0]

    # get covariance matrix
    a_transposed_a, mass_center = build_covariance_matrix (numpy_cloud )

    # get normal vector and smallest eigenvector
    normal_vector, smallest_eigenvalue = eigenvalue_decomposition (a_transposed_a )

    # get the noise and normalize it
    noise = smallest_eigenvalue
    if (cloud_size <= 3 or noise < 1 * 10 ** -10):
        sigma = noise   # no noise with 3 points
    else:
        sigma = sqrt(noise/(cloud_size - 3) )

    #print ('PCA completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return normal_vector, sigma, mass_center, time.time() - start_time


def random_plane_estimation (numpy_cloud, fixed_point=None ):
    '''
    Uses 3 random points to estimate Plane Parameters. Used for RANSAC.

    Input:
        numpy_cloud (np.ndarray): The Point Cloud in which to find a random plane

    Output:
        normal_vector ([1, 3] np.array):
        plane_parameter_d (float):
    '''
    start_time = time.time()

    measure = time.time ()

    # get 3 random indices
    idx_1, idx_2, idx_3 = random.sample(range(0, numpy_cloud.shape[0] ), 3 )

    part_1_time = time.time () - measure
    measure = time.time ()

    if (fixed_point is None):
        point_1 = numpy_cloud [idx_1, :].copy ()
    else:
        point_1 = fixed_point[:3].copy ()
    point_2 = numpy_cloud [idx_2, :].copy ()
    point_3 = numpy_cloud [idx_3, :].copy ()

    part_2_time = time.time () - measure
    measure = time.time ()

    # get the normal vector, normalize it and if it's turned to the ground, turn it around
    normal_vector = normalize_vector (np.cross((point_2 - point_1), (point_3 - point_1 )))

    part_3_time = time.time () - measure
    measure = time.time ()

    plane_parameter_d = -(normal_vector[0] * point_1[0]
                          + normal_vector[1] * point_1[1]
                          + normal_vector[2] * point_1[2] )
    if (normal_vector[2] < 0):      # z component
        normal_vector = normal_vector * -1

    part_4_time = time.time () - measure
    measure = time.time ()

    all = part_1_time + part_2_time + part_3_time + part_4_time

    # print ("\npart 1 : " + "{:2f}%".format ((part_1_time / all ) * 100 ))
    # print ("part 2 : " + "{:2f}%".format ((part_2_time / all ) * 100 ))
    # print ("part 3 : " + "{:2f}%".format ((part_3_time / all ) * 100 ))
    # print ("part 4 : " + "{:2f}%".format ((part_4_time / all ) * 100 ))

    return normal_vector, plane_parameter_d, time.time() - start_time


def plane_consensus_II (points, normal_vector, d, threshold ):
    '''
    Counts points that have a smaller distance than threshold from a given plane

    Input:
        points ([n, 3] np.ndarray):
        normal_vector ([1, 3] np.ndarray):
        d (float):                          Plane parameter d
        threshold (float):

    Output:
        consensus_count (int):
        consensus_points ([[x,y,z], ...] list)
    '''
    start_time = time.time()

    # create array for fast computation
    # normal_d = np.append (normal_vector, d)
    # array = np.concatenate ((np.tile (normal_d, (points.shape[0], 1) ), points ), axis=1 )

    #dists = np.absolute (np.apply_along_axis (distance_from_plane, 1, array ))
    dists = (normal_vector[0] * points[:, 0]
            + normal_vector[1] * points[:, 1]
            + normal_vector[2] * points[:, 2]
            + d )
    consensus_vector = np.where (dists < threshold, True, False )

    return np.sum (consensus_vector), points[consensus_vector, :], time.time() - start_time


def plane_consensus (numpy_cloud, normal_vector, d, threshold ):
    '''
    Counts points that have a smaller distance than threshold from a given plane

    Input:
        numpy_cloud ([n, 3] np.array):
        normal_vector ([1, 3] np.array):
        d
        threshold

    Output:
        consensus_count (int):
        consensus_points ([[x,y,z], ...] list)
    '''
    start_time = time.time()

    # plane paramters are elements of the normal vector
    a = normal_vector[0]
    b = normal_vector[1]
    c = normal_vector[2]

    # computing distances of every point from plane for consensus set
    consensus_count = 0
    consensus_points = []

    divisor = np.float_power (a, 2 ) + np.float_power (b, 2 ) + np.float_power (c, 2 )

    # refactor: apply along axis, or similar speed improvement
    # refactor: return np.ndarray, so no conversion is necessary in RANSAC
    for point in numpy_cloud:
        dist = (a * point[0] + b * point[1] + c * point[2] + d ) / sqrt (divisor)

        # threshold match?
        if (dist < threshold ):
            consensus_count = consensus_count + 1     # counting consensus
            consensus_points.append (point.tolist ())     # this might be slowing the code down

    return consensus_count, consensus_points, time.time() - start_time


def ransac_plane_estimation (input_numpy_cloud, threshold, fixed_point=None, w = .9, z = 0.95 ):
    """
    Uses Ransac with the probability parameters w and z to estimate a valid plane in given cloud.
    Uses distance from plane compared to given threshold to determine the consensus set.
    Returns points and point indices of the detected plane.
    Input:
        input_numpy_cloud (np.array):   Input cloud
        threshold (float, in m):        Points closer to the plane than this value are counted as inliers
        fixed_point (int):              This point will be used as one of three points for every plane estimation
        w (float between 0 and 1):      probability that any observation belongs to the model
        z (float between 0 and 1):      desired probability that the model is found
    Output:
        consensus_normal_vector ([1, 3] np.array):  The normal_vector computed
        consensus_points (np.array):                All points used for plane estimation
    """

    # measure time
    start_time = time.time ()

    # variables
    current_consensus = 0
    best_consensus = 0
    consensus_points = []  # points matching the cloud
    consensus_normal_vector = []

    # determine probabilities and number of draws
    b = np.float_power(w, 3 )   # probability that all three observations belong to the model
    k = ceil(np.log(1-z ) / np.log(1-b ))   # number of draws

    # copy cloud
    numpy_cloud = input_numpy_cloud[:, 0:3].copy ()

    # iterate: draw 3 points k times
    for i in range (1, k):

        # estimate a plane with 3 random points
        normal_vector, d, _ = random_plane_estimation (numpy_cloud, fixed_point )

        # this happens if three points are the same or on a line
        if (np.sum (normal_vector ) == 0 ):
            #print ("this happens if three points are the same or on a line")
            continue

        # count all points that consent with the plane
        current_consensus, current_consensus_points, _ = plane_consensus (numpy_cloud, normal_vector, d, threshold )

        # is the current consensus match higher than the previous ones?
        if (current_consensus > best_consensus ):
            consensus_points = current_consensus_points
            best_consensus = current_consensus    # keep best consensus set
            consensus_normal_vector = normal_vector

    # print time
    #print('RANSAC completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return np.array (consensus_normal_vector), np.array (consensus_points).copy (), 1, 1, 1


def ransac_plane_estimation_new (numpy_cloud, threshold, fixed_point=None, w = .9, z = 0.95 ):
    """
    Uses Ransac with the probability parameters w and z to estimate a valid plane in given cloud.
    Uses distance from plane compared to given threshold to determine the consensus set.
    Returns points and point indices of the detected plane.

    Input:
        numpy_cloud (np.ndarray):   Input cloud
        threshold (float, in m):        Points closer to the plane than this value are counted as inliers
        fixed_point (int):              This point will be used as one of three points for every plane estimation
        w (float between 0 and 1):      probability that any observation belongs to the model
        z (float between 0 and 1):      desired probability that the model is found
    Output:
        consensus_normal_vector ([1, 3] np.array):  The normal_vector computed
        consensus_points (np.ndarray):                All points used for plane estimation
    """

    # measure time
    overall_random_plane_time = 0
    overall_plane_consensus_time = 0
    start_time = time.time ()

    # variables
    current_consensus = 0
    best_consensus = 0
    consensus_points = np.array([])  # points matching the cloud
    consensus_normal_vector = []

    # determine probabilities and number of draws
    b = np.float_power(w, 3 )   # probability that all three observations belong to the model
    k = ceil(np.log(1-z ) / np.log(1-b ))   # number of draws

    # # # copy cloud
    numpy_cloud = numpy_cloud[:, 0:3].copy ()

    # # iterate: draw 3 points k times
    # for i in range (1, k):
    #
    #     # estimate a plane with 3 random points
    #     normal_vector, d, random_plane_time = random_plane_estimation (numpy_cloud, fixed_point )
    #     overall_random_plane_time += random_plane_time

    measure = time.time ()

    # get 3 * k random indices
    indices = np.random.random_integers (0, numpy_cloud.shape[0] - 1, 3 * k )    # .reshape (-1, 3)

    # print ("\nindices, cloud")
    # print (indices.shape)
    # print (numpy_cloud.shape)

    points = numpy_cloud[indices, 0:3].copy ()

    points = points.reshape (-1, 9)
    points_1 = points[:, 0:3]
    points_2 = points[:, 3:6]
    points_3 = points[:, 6:9]

    # fixed point
    if (fixed_point is not None ):
        points_1 = fixed_point[0:3].reshape (-1, 3)  # np.tile (fixed_point, (points_2.shape[0], 1 ))

    # print ("Poins")
    # print (points.shape)
    # print (points_1.shape)
    # print (points_2.shape)
    # print (points_3.shape)

    # get the normal vector, normalize it
    normal_vectors = normalize_vector_array (np.cross((points_2 - points_1), (points_3 - points_1 )))

    # get plane parameter d, distance from origin
    plane_parameters_d = -(normal_vectors[:, 0] * points_1[:, 0]
                          + normal_vectors[:, 1] * points_1[:, 1]
                          + normal_vectors[:, 2] * points_1[:, 2] )

    # if it's turned to the ground, turn it around
    normal_vectors = np.where (normal_vectors[2] < 0, normal_vectors * -1, normal_vectors )

    overall_random_plane_time = time.time () - measure

    for (normal_vector, d) in zip (normal_vectors, plane_parameters_d ):

        # print ("\nnormal_vector array: " + str (normal_vector ))
        # print ("normal_vector for:   " + str (n_2 ))
        # print ("d array: " + str (d ))
        # print ("d for:   " + str (d_2 ))

        # count all points that consent with the plane
        current_consensus, current_consensus_points, plane_consensus_time = \
            plane_consensus_II (numpy_cloud, normal_vector, d, threshold )
        overall_plane_consensus_time += plane_consensus_time

        # is the current consensus match higher than the previous ones?
        if (current_consensus > best_consensus ):
            consensus_points = current_consensus_points
            best_consensus = current_consensus    # keep best consensus set
            consensus_normal_vector = normal_vector

    # print time
    #print('RANSAC completed in ' + str(time.time() - start_time) + ' seconds.\n' )

    return np.array (consensus_normal_vector), consensus_points, overall_random_plane_time, overall_plane_consensus_time, time.time() - start_time


# set the random seed for both the numpy and random module, if it is not already set.
random.seed (1337 )
np.random.seed (1337 )
