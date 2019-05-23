#import itertools
import numpy as np
from collections import namedtuple


class CustomCloud(object ):
    """
    CustomCloud:

    Attributes:
        data (np.array):
        labels (list of str):
        fields (container of numpy colums):
        rows (container of numpy rows):

    """

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
        self.size = self.data.size
        self.itemsize = self.data.itemsize
        self.nbytes = self.data.nbytes

    # numpy init function
    @classmethod
    def initialize_xyz (cls, input_cloud_numpy ):  # cls = the class
        '''
        Initialize a Pointcloud from a numpy array with three colums: X, Y, and Z
        This wraps __init__ and hands over the according numpy colums and labels
        '''

        if (input_cloud_numpy.shape[1] < 3 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        return cls (input_cloud_numpy[:, 0:3], ["X", "Y", "Z"] )   # this calls __init__

    # numpy init function
    @classmethod
    def initialize_xyzi (cls, input_cloud_numpy ):  # cls = the class
        '''
        Initialize a Pointcloud from a numpy array with four colums: X, Y, Z and Intensity
        This wraps __init__ and hands over the according numpy colums and labels
        '''

        if (input_cloud_numpy.shape[1] < 4 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        return cls (input_cloud_numpy[:, 0:4], ["X", "Y", "Z", "Intensity"] )   # this calls __init__

    # numpy init function
    @classmethod
    def initialize_xyzrgb (cls, input_cloud_numpy ):  # cls = the class
        '''
        Initialize a Pointcloud from a numpy array with six colums: X, Y, Z, red, green and blue (Rf, Gf, Bf)
        This wraps __init__ and hands over the according numpy colums and labels

        Please use 0-1 float color
        '''

        if (input_cloud_numpy.shape[1] < 6 ):
            raise ValueError("The given numpy array only has " + str (input_cloud_numpy.shape[1] ) + " rows. Aborting.")

        return cls (input_cloud_numpy[:, 0:6], ["X", "Y", "Z", "Rf", "GF", "Bf"] )   # this calls __init__

    # iteration
    def __iter__(self ):
        '''
        This function is called when an object of this class is iterated over.
        Returns an iteratable generator made of namedtuples to make iterating over this class possible
        '''

        Point = namedtuple("Point", self.labels )

        #returning a generator that is used like an iterator using list comprehension
        return ((Point._make (row) for (row) in self.data ))

    def __getitem__(self, input):
        return self.data.__getitem__ (input)

    # makes CustomCloud[:, "x":"z"] possible
    # def __getitem__(self, input):
        # length, fields = input
        # print ("length.start: " + str(length.start ))
        # print ("length.stop: " + str(length.stop ))
        # print ("length.step: " + str(length.step ))
        #
        # print ("fields.start: " + str(fields.start ))
        # print ("fields.stop: " + str(fields.stop ))
        # print ("fields.step: " + str(fields.step ))

        # index = start
        # if stop is None:
        #     end = start + 1
        # else:
        #     end = stop
        # if step is None:
        #     stride = 1
        # else:
        #     stride = step
        # return self.__data[index:end:stride]

    # printing
    def __str__(self ):
        '''
        This Method is called when an object of this class is printed.
        It returns a nicely formatted string containing the first few and last few points.
        '''

        cloud_size = self.data.shape[0]
        np.set_printoptions (precision=1)  # print to give a rough outline of the cloud

        # general info including name of the class and it's number of points
        string = str (type(self).__name__  # + ',\nwith labels: ' + str (self.labels )
                     + ', number of points: ' + str (cloud_size ) + '\n' + str (self.labels ).replace (',', '') )
        margin = 3  # how many points are printed before output is clipped

        # add points pairs to the string, but exlcude those that are out of the margin
        for counter, row in enumerate (self.data ):
            if (counter >= margin and counter < cloud_size - margin):
                if (counter == margin + 1):
                    string = string + "\n\n...\n"
                pass
            else:
                string = string + '\n' + str(row )

        np.set_printoptions (precision=None)
        return str (string )

    def update_attributes (self ):
        '''
        This is called after changing the structure of the cloud
        '''
        # instance variables
        labels = namedtuple("Colums", self.labels )
        self.fields = labels._make (column for column in self.data.T )
        self.rows = self.__iter__ ()

        # numpy attributes
        self.shape = self.data.shape
        self.size = self.data.size
        self.itemsize = self.data.itemsize
        self.nbytes = self.data.nbytes

    # def __repr__(self):
    #     return "x, y, z"

    def add_field (self, input_cloud_numpy, field_name):
        '''
        This function adds a field (colum) to the cloud.

        Naming Convention examples:
            'Intensity '
            'Number_of_Returns '
            'Return_Number '
            'Point_Source_ID '
            'Classification'
        '''
        if (len (input_cloud_numpy.shape ) != 1):   # if no shape is one
            raise ValueError("The given numpy array is misshaped. Expected an array of shape (1, n). Aborting.")

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
        self.update_attributes ()

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
        self.update_attributes ()

    def has_field (self, field_name ):
        return any(field_name in s for s in self.labels)
