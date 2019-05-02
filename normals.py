import time
import pcl
import sys
import numpy as np
from laspy.file import File

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


def load_las(dir_in, filename, dtype = None):
    """
    Loads .las data as numpy array

    Inputs:
        dir_in: string; directory in
        filename: String; name of the .las tile (incl. .las)
        dtype: String;
        if dtype = 'als', then the function will return points as [x, y, z, intensity, class]
        if dtype = 'dim', then the function will return points as [x, y, z, r, g, b, class]
        if dtype = None, then the function will return points as [x, y, z, class]
        default: dtype = None

    Outputs:
        points: np array; contains n points with different columns depending on dtype
    """

    # load tile
    with File(dir_in + filename, mode = 'r') as inFile:
        x = np.reshape(inFile.x.copy(), (-1, 1)) # create colums
        y = np.reshape(inFile.y.copy(), (-1, 1))
        z = np.reshape(inFile.z.copy(), (-1, 1))
        raw_class = np.reshape(inFile.raw_classification.copy(), (-1, 1))


        if dtype == 'dim':
            red = np.reshape(inFile.red.copy(), (-1, 1))    # add rgb
            green = np.reshape(inFile.green.copy(), (-1, 1))
            blue = np.reshape(inFile.blue.copy(), (-1, 1))
            points = np.concatenate((x, y, z, red, green, blue, raw_class), axis = -1)
        elif dtype == 'als':
            intensity = np.reshape(inFile.intensity.copy(), (-1, 1))    # add intensity
            #num_returns = inFile.num_returns    # number of returns
            #return_num = inFile.return_num      # this points return number
            points = np.concatenate((x, y, z, intensity, raw_class), axis = -1)
        else:
            points = np.concatenate((x, y, z, raw_class), axis = -1)

    return points

def pcl_load (fileName):
    # Loading
    start_time = time.time()
    fileName = 'clouds/laserscanning_plane_cc.obj'
    print('Load file ... ')
    #inputPointCloud = pcl.PointCloud
    inputPointCloud = pcl.load(fileName)
    print('Cloud loaded in' + str(time.time() - start_time) + ' seconds.\nNumber of points: ' + str(inputPointCloud.size ) + '\n')

    return inputPointCloud

def numpy_to_pcl (numpy_cloud ):
    return pcl.PointCloud_XYZI(np.array(numpy_cloud, dtype=np.float32))


def pcl_to_numpy (pcl_cloud ):
    return pcl_cloud.to_array (pcl_cloud)


def PCA (input_numpy_cloud ):
    start_time = time.time()
    # we only need three colums [X, Y, Z, I] -> [X, Y, Z]
    numpy_cloud = input_numpy_cloud.copy ()
    numpy_cloud = numpy_cloud [:, 0:3]

    # build a sum over all points
    sum_xyz = np.array ((0,0,0 ))
    for point in numpy_cloud:
        sum_xyz[0] = sum_xyz[0] + point[0]
        sum_xyz[1] = sum_xyz[1] + point[1]
        sum_xyz[2] = sum_xyz[2] + point[2]

    # and normalize it to get center of mass
    sum_xyz = sum_xyz / numpy_cloud.size

    # reduce point cloud by center of mass
    numpy_cloud_reduced = np.subtract (numpy_cloud[:, 0:3], sum_xyz )


    # build ATA matrix
    a_transposed_a = np.zeros ((3,3 ))

    for point in numpy_cloud_reduced:
       a_transposed_a[0,0] = a_transposed_a[0,0] + np.float_power(point[0],2 );
       a_transposed_a[0,1] = a_transposed_a[0,1] + point[0] * point[1];
       a_transposed_a[0,2] = a_transposed_a[0,2] + point[0] * point[2];

       a_transposed_a[1,0] = a_transposed_a[1,0] + point[0] * point[1];
       a_transposed_a[1,1] = a_transposed_a[1,1] + np.float_power(point[1], 2 );
       a_transposed_a[1,2] = a_transposed_a[1,2] + point[1] * point[2];

       a_transposed_a[2,0] = a_transposed_a[2,0] + point[0] * point[2];
       a_transposed_a[2,1] = a_transposed_a[2,1] + point[2] * point[1];
       a_transposed_a[2,2] = a_transposed_a[2,2] + np.float_power(point[2], 2 );

    # get eigenvalues and -vectors from ATA matrix
    eigenvalues = np.zeros (a_transposed_a.shape[0] )
    eigenvectors = np.zeros ((a_transposed_a.shape[0], a_transposed_a.shape[0] ))
    evals, evecs = np.linalg.eig (a_transposed_a )

    # sort them
    indices = np.argsort (-evals ) # reverse sort: greatest numbers first
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

    #pcl_input_cloud = pcl_load ('clouds/laserscanning_plane.ply')
    #pcl_normals = pcl_compute_normals (pcl_input_cloud )
    #normal_vector, noise, sigma = PCA (pcl_input_cloud.to_array () )

    np_cloud = load_las ('clouds/', 'laserscanning_plane_cc.las')

    normal_vector, noise, sigma = PCA (np_cloud )
    print ('normal_vector: ' + str(normal_vector ))
    print ('noise: ' + str(noise ))
    print ('sigma: ' + str(sigma ))
