import sys
#import os
import numpy as np
from laspy.file import File
import matplotlib
import pcl


# useful stuff
#   num_returns = inFile.num_returns    # number of returns
#   return_num = inFile.return_num      # this points return number
#   ground_points = inFile.points[num_returns == return_num]
#
#   inFile.intensity
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





print ('executed with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) + "\n" )

# Loading
clouds_folder = 'clouds/ALS2014/'
file_name = '33314059950_05_2014-01-03.las'

print('Loading file ' + file_name + ' ...')
input_cloud = load_las (clouds_folder, file_name )
#input_cloud = pcl.load(file_name)
print ('Cloud imported.\nNumber of points: ' + str(input_cloud.size ))

print ('\n Make pcl cloud')
cloud = pcl.PointCloud_PointXYZRGB()





#########################################
# try this
np_cloud = np.empty([pcl_cloud.width, 4], dtype=np.float32)
np_cloud = np.asarray(pcl_cloud)
np_cloud = pcl_cloud.to_array()



#########################################
# or this
p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
seg = p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
indices, model = seg.segment()

# print (input_cloud.shape)
# print (type (input_cloud[0,0]) )
# for value in input_cloud:
#    value = np.float32 (value )
#
# #input_cloud.astype (float)
# print (type (input_cloud[0,0]) )
#
# cloud.from_array(input_cloud )
#
# print ('\nDone.')




# def load_cloud (path_to_file, file_name ):
#     """
#     Wrapping the pcl load and custom .lsa load function
#     """
#
#     print('Loading file ' + file_name + '.')
#     input_cloud = None;
#
#     # check for file extension to determine the function used for loading
#     file_extension = os.path.splitext (file_name )[1]
#     if (file_extension == '.las'):
#         input_cloud = load_las ()
#     else:
#         input_cloud = pcl.load(file_name)
#
#     if (input_cloud != None ):
#         print('Cloud imported.\nNumber of points: ' + str(input_cloud.size ))
#     else:
#         print ('Cloud import error');
#
#     return input_cloud
