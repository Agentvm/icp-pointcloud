# snippets


# def display_small_cloud (cloud ):
#     fig = pyplot.figure()
#     ax = Axes3D(fig)
#
#     print (type (cloud))
#     if isinstance(cloud, pcl._pcl.PointCloud):
#         cloud = cloud.to_array ()
#
#     for i in range(0, cloud.size):
#         ax.scatter(cloud[i][0], cloud[i][1], cloud[i][2])
#
#     pyplot.show()


# seg = p.make_segmenter()
# seg.set_model_type(pcl.SACMODEL_PLANE)
# seg.set_method_type(pcl.SAC_RANSAC)
# indices, model = seg.segment()


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
