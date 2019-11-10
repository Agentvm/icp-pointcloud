"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

Offers Principal Component Analysis and RANSAC algorithms for normal vector computation.
"""


# basic imports
import numpy as np
from math import ceil, sqrt


def normalize_vector (vector ):
    """Takes a vector and returns it's unit vector"""

    if (np.sum (vector ) == 0):
        #print ("In normalize_vector: Vector is 0. Returning input vector.")
        return vector

    return vector / np.linalg.norm(vector)


def normalize_vector_array (vector_array ):
    """Normalizes Vectors of an numpy.ndarray of shape (-1, n)"""
    norms = np.linalg.norm (vector_array, axis=1 )
    norms = np.where (norms == 0, 1, norms )    # these filtered values belong to arrays that already are normalized

    return vector_array / norms.reshape (-1, 1 )


def einsum_angle_between (vector_array_1, vector_array_2 ):
    """Works on (n, 3) numpy.ndarrays of vectors and returns the angle difference in rad between each pair of vectors"""

    # diagonal of dot product
    diag = np.clip (np.einsum ('ij,ij->i', vector_array_1, vector_array_2 ), -1, 1 )

    return np.arccos (diag )


def eigenvalue_decomposition (a_t_a_matrix ):
    """Uses np.linalg.eig () to decompose a 3x3 matrix. Returns normal vector and smallest eigenvalue."""
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
    """Given a pointcloud, this computes the corrsponding covariance matrix for eigenvalue decomposition"""

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
        numpy_cloud: (np.ndarray)           Numpy array with data points, only the first 3 columns are used

    Output:
        normal_vector: ([1, 3] np.array)    The normal vector of the computed plane
        sigma: (float)                      The noise as given by the smallest eigenvalue, normalized by point number
        mass_center: ([1, 3] np.array)      Centre of mass
    """

    # abort, if there are no points
    if (numpy_cloud.shape[0] == 0):
        #print ("In normals.py, in PCA: The input array is empty. Returning a null vector and high sigma")
        return np.array ((0, 0, 0)), 1.0, np.array ((0, 0, 0))

    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = numpy_cloud[:, :3].copy ()     # copying takes roughly 0.000558 seconds per 1000 points
    cloud_size = numpy_cloud.shape[0]

    # get covariance matrix
    a_transposed_a, mass_center = build_covariance_matrix (numpy_cloud )

    # get normal vector and smallest eigenvalue
    normal_vector, smallest_eigenvalue = eigenvalue_decomposition (a_transposed_a )

    # the noise is based on the smallest eigenvalue and normalized by number of points in cloud
    noise = smallest_eigenvalue
    if (cloud_size <= 3 or noise < 1 * 10 ** -10):
        sigma = noise   # no noise with 3 points
    else:
        sigma = sqrt(noise/(cloud_size - 3) )

    return normal_vector, sigma, mass_center


def random_plane_estimation (numpy_cloud, number_of_planes, fixed_point=None ):
    '''
    Generates a number of planes from randomly chosen triples of points from numpy_cloud.

    Input:
        numpy_cloud: (np.ndarray)       The Point Cloud in which to find random planes
        number_of_planes: (int)         Sets how many planes are to be determined. Must be a multiple of 3.
        fixed_point: ([1, 3]np.ndarray) It is possible to set one angle point that is part of every determined plane

    Output:
        normal_vectors: ([number_of_planes, 3] np.ndarray)      The computed normal vectors, one for each plane
        plane_parameter_d: ([number_of_planes, ] np.ndarray)    The plane paramters d, one for each plane
    '''

    # get random indices and extract the corresponding point from the cloud (casting to int as a safety measure)
    indices = np.random.random_integers (0, numpy_cloud.shape[0] - 1, int (number_of_planes / 3) * 3 )
    points = numpy_cloud[indices, 0:3].copy ()

    # reshape the results, so that pairs of three points can be formed
    points = points.reshape (-1, 9)
    points_1 = points[:, 0:3]
    points_2 = points[:, 3:6]
    points_3 = points[:, 6:9]

    # introduce fixed point
    if (fixed_point is not None ):
        points_1 = fixed_point[0:3].reshape (-1, 3)

    # get the normal vectors, normalize them
    normal_vectors = normalize_vector_array (np.cross((points_2 - points_1), (points_3 - points_1 )))

    # get plane parameters d, distance from origin
    plane_parameters_d = -(normal_vectors[:, 0] * points_1[:, 0]
                          + normal_vectors[:, 1] * points_1[:, 1]
                          + normal_vectors[:, 2] * points_1[:, 2] )

    # normal vector: if it's turned to the ground, turn it around
    normal_vectors = np.where (normal_vectors[2] < 0, normal_vectors * -1, normal_vectors )

    return normal_vectors, plane_parameters_d


def plane_consensus (points, normal_vector, d, threshold ):
    '''
    Counts points that have a smaller distance than threshold from a given plane

    Input:
        points: ([n, 3] np.ndarray)             The points which to test
        normal_vector: ([1, 3] np.ndarray)      Normal vector of plane
        d: (float)                              Plane parameter d
        threshold: (float)                      Distance at which a point is no longer part of the plane

    Output:
        consensus_count: (int)                  Count of consenting points (points that are part of the plane)
        consensus_points: ([x, 3] np.ndarray)   Plane points
    '''

    distances = (normal_vector[0] * points[:, 0]
                + normal_vector[1] * points[:, 1]
                + normal_vector[2] * points[:, 2]
                + d )

    consensus_vector = np.where (distances < threshold, True, False )

    return np.sum (consensus_vector), points[consensus_vector, :]


def ransac_plane_estimation (numpy_cloud, threshold, fixed_point=None, w = .9, z = 0.95 ):
    """
    Uses Ransac with the probability parameters w and z to estimate a valid plane in given cloud.
    Uses distance from plane compared to given threshold to determine the consensus set.
    Returns points and normal vector of the detected plane.

    Input:
        numpy_cloud: (np.ndarray)               Input cloud
        threshold: (float, in m)                Points closer to the plane than this value are counted as inliers
        fixed_point: ([1, 3]np.ndarray)         This point will be used for every plane estimation
        w: (float between 0 and 1)              Probability that any observation belongs to the model
        z: (float between 0 and 1)              Desired probability that the model is found

    Output:
        best_normal_vector: ([1, 3] np.array)   The resulting normal vector
        consensus_points: (np.ndarray)          Points that are part of the estimated plane
    """

    # variables
    current_consensus = 0               # keeps track of how many points match the current plane
    best_consensus = 0                  # shows how many points matched the best plane yet
    consensus_points = np.array([])     # np.ndarray of points matching the cloud
    best_normal_vector = np.array ([])  # current best normal vector

    # determine probabilities and number of draws
    b = np.float_power(w, 3 )               # probability that all three observations belong to the model
    k = ceil(np.log(1-z ) / np.log(1-b ))   # estimated number of draws

    # copy cloud
    numpy_cloud = numpy_cloud[:, 0:3].copy ()

    # estimate k * 3 random planes, defined through one normal vector and one plane parameter d, respectively
    normal_vectors, plane_parameters_d = random_plane_estimation (numpy_cloud, k * 3, fixed_point )

    # iterate through all planes found to see which one performs best
    for (normal_vector, d) in zip (normal_vectors, plane_parameters_d ):

        # count all points that consent with the plane
        current_consensus, current_consensus_points = plane_consensus (numpy_cloud, normal_vector, d, threshold )

        # is the current consensus match higher than the previous ones?
        if (current_consensus > best_consensus ):

            # keep best consensus set
            consensus_points = current_consensus_points
            best_normal_vector = normal_vector
            best_consensus = current_consensus

    return best_normal_vector, consensus_points
