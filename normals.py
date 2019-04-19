import time
import pcl
import sys

# Loading
fileName = 'clouds/laserscanning.ply'
print('Load file ... ')
inputPointCloud = pcl.load(fileName)
print('Number of points: ' + str(inputPointCloud.size ))

# Search
searching_neighbour = {'knn_search': 25}
feat = inputPointCloud.make_NormalEstimation()

if 'range_search' in searching_neighbour.keys():
    # Range search
    searching_para = searching_neighbour['range_search'] if searching_neighbour['range_search'] > 0 else 0.1
    feat.setRadiusSearch(searching_para)
elif 'knn_search' in searching_neighbour.keys():
    # kNN search
    searching_para = int(searching_neighbour['knn_search']) if int(searching_neighbour['knn_search']) > 5 else 20
    tree = inputPointCloud.make_kdtree()
    feat.set_SearchMethod(tree)
    feat.set_KSearch(searching_para)
else:
    print('Define researching method does not support')


start_time = time.time()
normals = feat.compute()

print ('\nexecuted with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) )
print("Computed normal vectors in " + str(time.time() - start_time) + " seconds.")
