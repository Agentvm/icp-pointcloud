import input_output
from conversions import reduce_cloud
from os import listdir, walk
from os.path import isfile, join, splitext


def get_all_files_in_subfolders (path_to_folder, permitted_file_extension=None ):
    '''
    Finds all files inside the folders below the given folder (1 level below)
    '''

    # find all directories below path_to_folder
    f = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        f.extend(filenames)
        break

    # append the directories to the input directory
    # add the input directory itself, so files in there will be found
    full_directories = [path_to_folder.strip ('/')]
    for dir in dirnames:
        dir = path_to_folder + dir
        full_directories.append (dir)

    #for every directory found, find all files inside and append the resulting path to each file to full_paths
    full_paths = []
    for dir_count, directory in enumerate (full_directories ):
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        for file in onlyfiles:
            full_paths.append (full_directories[dir_count] + '/' + file )

    # if specified, remove all file extensions that do not match the specified extension
    if (permitted_file_extension is not None ):
        for path in full_paths:
            filename, file_extension = splitext(path )
            if (file_extension != permitted_file_extension):
                full_paths.remove (path )

    return full_paths


def reduce_clouds ():

    # load all .las files in a given folder, reduce their points, so they are closer to zero,
    # then save them again with a different name

    # crawl path
    path = "clouds/Regions/"    # there are obly clouds below this folder
    full_paths = get_all_files_in_subfolders (path, ".las" )
    print ("full_paths: " + str (full_paths ))

    # check if files exist
    for file_path in full_paths:
        if (input_output.check_for_file (file_path ) is False ):
            print ("File " + file_path + " was not found. Aborting.")
            return False

    # load las clouds
    for file_path in full_paths[3:4]:
        print ("\n\n-------------------------------------------------------")

        # check if it's a .las file, else skip it
        filename, file_extension = splitext(file_path )
        if (file_extension != ".las" ):
            continue

        field_labels_list = ['X', 'Y', 'Z']

        # load the file, then reduce it
        if ("DSM_Cloud" in file_path):
            # Load DIM cloud
            numpy_cloud = input_output.load_las_file (file_path, dtype="dim" )
            numpy_cloud[:, 3:6] = numpy_cloud[:, 3:6] / 65535.0  # rgb short int to float
            field_labels_list.append ('Rf ' 'Gf ' 'Bf ' 'Classification')
        else:
            # Load ALS cloud
            numpy_cloud = input_output.load_las_file (file_path, dtype="als")
            field_labels_list.append('Intensity '
                                     'Number_of_Returns '
                                     'Return_Number '
                                     'Point_Source_ID '
                                     'Classification')

        # reduce
        numpy_cloud = reduce_cloud (numpy_cloud, copy=False )

        # save them again
        input_output.save_ascii_file (numpy_cloud, field_labels_list, "clouds/tmp/reduce_test.asc" )

    print ("Done.")
    return True


if __name__ == '__main__':
    if (reduce_clouds () ):
        print ("\n\nAll Clouds successfully reduced.")
    else:
        print ("Error. Not all clouds could be reduced.")
