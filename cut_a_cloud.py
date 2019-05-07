import numpy as np
import input_output
import argparse


def init_parser ():
    '''
    initalize commandline parser options (module argparse)
    '''

    parser = argparse.ArgumentParser (description='Process some integers.' )
    parser.add_argument ('las_source_file', type=str, help='The .las file to load' )
    parser.add_argument ('min_x', type=float, help='The lower x coordinate, where the cut begins' )
    parser.add_argument ('max_x', type=float, help='The higher x coordinate, where the cut ends' )
    parser.add_argument ('min_y', type=float, help='The lower y coordinate, where the cut begins' )
    parser.add_argument ('max_y', type=float, help='The higher y coordinate, where the cut ends' )
    parser.add_argument ('-d', '--is_dim_cloud',  action='store_true',
                         help='set, if you are loading a dense image matching cloud,'
                              + ' omit if you are loading a aerial laserscanning cloud' )
    parser.add_argument ('--ply_output_file', default='',
                         help='The name of the ply file to save output in, including .ply' )

    return parser


if __name__ == "__main__":
    '''
    This script loads a given .las file, cuts out a given part of it and saves it with a given name.
    Open it via console by typing 'python cut_a_cloud.py -h'.

    example usage: 'python cut_a_cloud.py name_of_your_file 1 10 1 10'
    '''

    # parse the arguments
    parser = init_parser ()
    args = parser.parse_args()

    # handle the path names
    if ('/' in args.las_source_file ):
        [path, input_file] = args.las_source_file.rsplit ('/', 1 )
        path = path + '/'
        if (args.ply_output_file == ''):
            args.ply_output_file = path + 'output_' + input_file
    else:
        path = ''
        input_file = args.las_source_file
        if (args.ply_output_file == ''):
            args.ply_output_file = 'output_' + input_file

    # load the file
    dtype = 'als'
    if (args.is_dim_cloud ):
        dtype = 'dim'
    numpy_cloud = input_output.load_las_file(path, input_file, dtype )

    # cut a cloud via list comprehension
    print ('Selecting a subset of points\n'
           + 'x = { > ' + str (args.min_x ) + ', < ' + str (args.max_x ) + '}\n'
           + 'y = { > ' + str (args.min_y ) + ', < ' + str (args.max_y ) + '}\n')

    # new_list = [expression(i) for i in old_list if filter(i)]
    subset_cloud = np.array ([point for point in numpy_cloud if (point[0] > args.min_x
                                                                 and point[0] < args.max_x
                                                                 and point[1] > args.min_y
                                                                 and point[1] < args.max_y)])

    print ('Subset size: ' + str (subset_cloud.shape[0]) )
    print ('Subset:\n' + str (subset_cloud ) + '\n')

    # save cloud
    # rsplit ('.', 1)[0] splits the string at the last occurence of '.' and returns the first part
    input_output.save_ply_file(subset_cloud, args.ply_output_file.rsplit ('.', 1)[0] + '.ply' )
