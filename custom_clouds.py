#import itertools
import numpy as np
from collections import namedtuple


class CustomCloud(object ):
    """CustomCloud docstring"""

    # basic init function
    def __init__(self, numpy_array, labels_list ):
        '''
        This function is called when an object of this class is instantiated.
        It fills the object with the values provided
        '''
        super(CustomCloud, self).__init__()

        # instance variables
        # data
        self.data = numpy_array

        #labels
        self.labels = labels_list
        labels = namedtuple("Colums", self.labels )
        self.fields = labels._make (column for column in self.data.T )
        self.rows = self.__iter__ ()

        # numpy attributes
        self.shape = self.data.shape

    # numpy init function
    @classmethod
    def initialize_xyz (cls, input_cloud_numpy ):  # cls = the class
        '''
        Initialize a Pointcloud from a numpy array with three colums: x, y and z
        This wraps __init__ and hands over the according numpy colums and labels
        '''

        if (input_cloud_numpy.shape[1] < 3 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        return cls (input_cloud_numpy[:, 0:3], ["x", "y", "z"] )   # this calls __init__

    # numpy init function
    @classmethod
    def initialize_xyzi (cls, input_cloud_numpy ):  # cls = the class
        '''
        Initialize a Pointcloud from a numpy array with four colums: x, y, z and intensity
        This wraps __init__ and hands over the according numpy colums and labels
        '''

        if (input_cloud_numpy.shape[1] < 4 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        return cls (input_cloud_numpy[:, 0:4], ["x", "y", "z", "i"] )   # this calls __init__

    # numpy init function
    @classmethod
    def initialize_xyzrgb (cls, input_cloud_numpy ):  # cls = the class
        '''
        Initialize a Pointcloud from a numpy array with six colums: x, y, z, red, green, blue
        This wraps __init__ and hands over the according numpy colums and labels
        '''

        if (input_cloud_numpy.shape[1] < 6 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        return cls (input_cloud_numpy[:, 0:6], ["x", "y", "z", "r", "g", "b"] )   # this calls __init__

    # iteration
    def __iter__(self ):
        '''
        This function is called when an object of this class is iterated over.
        Returns an iteratable generator made of namedtuples to make iterating over this class possible
        '''

        Point = namedtuple("Point", self.labels )

        #returning a generator that is used like an iterator using list comprehension
        return ((Point._make (row) for (row) in self.data ))

    # printing
    def __str__(self ):
        '''
        This Method is called when an object of this class is printed.
        It returns a nicely formatted string containing the points
        '''

        cloud_size = self.data.shape[0]

        # general info including name of the class and it's number of points
        string = str (type(self).__name__  # + ',\nwith labels: ' + str (self.labels )
                     + ', number of points: ' + str (cloud_size ) + '\n' + str (self.labels ).replace (',', '') )
        margin = 2  # how many points are printed before output is clipped

        # add points pairs to the string, but exlcude those that are out of the margin
        for counter, row in enumerate (self.data ):
            if (counter >= margin and counter < cloud_size - margin):
                if (counter == margin + 1):
                    string = string + "\n\n...\n"
                pass
            else:
                string = string + '\n' + str(row )

        return str (string )

    # def __repr__(self):
    #     return "x, y, z"

    def add_field (self, input_cloud_numpy, field_name):
        '''
        This function adds a field (colum) to the cloud.
        '''
        if (len (input_cloud_numpy.shape ) != 1):   # if no shape is one
            raise ValueError("The given numpy array is misshaped. Please pass an array which has shape (n, 1). "
                             + "Aborting.")

        # if (input_cloud_numpy.shape[1] != 1):
        #     input_cloud_numpy = input_cloud_numpy.T     # transpose

        if (input_cloud_numpy.shape[0] != self.shape[0] ):
            raise ValueError("The given numpy array contains too few points, compared to this cloud: ("
                             + str (input_cloud_numpy.shape[0] ) + "points/" + str (self.shape[0] )
                             + " points). Aborting.")

        # add data
        self.data = np.concatenate ((self.data, input_cloud_numpy[:, None]), 1 )

        # update labels
        self.labels.append (field_name )
        labels = namedtuple("Colums", self.labels )
        self.fields = labels._make (column for column in self.data.T )

    def add_normals (self, input_cloud_numpy ):
        '''
        This function adds three fields (colums) to the cloud. These are called "norm_x", "norm_y" and "norm_z"
        '''

        if (input_cloud_numpy.shape[1] < 3 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        if (input_cloud_numpy.shape[0] != self.shape[0] ):
            raise ValueError("The given numpy array contains too few points, compared to this cloud: ("
                             + str (input_cloud_numpy.shape[0] ) + "points/" + str (self.shape[0] )
                             + " points). Aborting.")

        # add data
        self.data = np.concatenate ((self.data, input_cloud_numpy[:, 0:3]), 1 )

        # update labels
        self.labels.append (["norm_x", "norm_y", "norm_z"] )
        labels = namedtuple("Colums", self.labels )
        self.fields = labels._make (column for column in self.data.T )

    def has_field (self, field_name ):
        return any(field_name in s for s in self.labels)
