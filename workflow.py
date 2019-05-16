# for testing and demonstration of point cloud workflows

import sys

# custom modules
import conversions
import input_output

# visualization
#from matplotlib import pyplot
#from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':

    print ('executed with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) + "\n" )

    # Files
    #clouds_folder = 'clouds/ALS2014/'
    #file_name = '33314059950_05_2014-01-03.las'

    clouds_folder = 'clouds/DIM2016/'
    file_name = 'DSM_Cloud_333140_59950.las'

    #clouds_folder = 'clouds/laserscanning/'
    #file_name = 'laserscanning.ply'

    # Loading
    numpy_cloud = input_output.load_las_file (clouds_folder + file_name )

    print ('Numpy:\n' + str(numpy_cloud ) + "\n\n" )

    pcl_cloud = conversions.numpy_to_pcl (numpy_cloud )
    #pcl_cloud = pcl.load(str(clouds_folder + file_name ))

    print ('np(PCL):\n' + str(pcl_cloud.to_array () ) + "\n\n" )  # how to print a pcl cloud??

    #pcl.save (pcl_cloud, str(clouds_folder) + 'output.ply')
