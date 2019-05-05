#import itertools
from collections import namedtuple
Point = namedtuple ("Point", ['x', 'y', 'z'])


class XYZCloud(object):
    """docstring for XYZCloud."""

    # variables
    x = []
    y = []
    z = []
    idx = 0

    # basic init function
    def __init__(self, x, y, z):
        '''
        This function is called when an object of this class is instantiated.
        It fills the object with the values provided
        '''
        super(XYZCloud, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.idx = 0

    # numpy init function
    @classmethod
    def from_numpy_array (cls, input_cloud_numpy):  # cls = the class
        '''
        This wraps __init__ and
        '''

        # iterating over a list is slightly faster than iterating over a numpy array
        x = input_cloud_numpy[:, 0].tolist ()
        y = input_cloud_numpy[:, 1].tolist ()
        z = input_cloud_numpy[:, 2].tolist ()

        return cls (x, y, z )   # this calls __init__

    # iteration
    def __iter__(self):
        '''
        This function is called when an object of this class is iterated over.
        Returns an iteratable generator made of namedtuples to make iterating over this class possible
        '''
        #returning a generator that is used like an iterator using list comprehension
        return ((Point._make ([x, y, z]) for (x, y, z) in zip(self.x, self.y, self.z )))

    # @classmethod
    # def __next__(self ):
    #     self.idx = self.idx + 1
    #
    #     try:
    #         print ('trying')
    #         print ('x: ' + str(self.x ))
    #         return self
    #         #return XYZCloud (self.x[self.idx-1], self.y[self.idx-1], self.z[self.idx-1] )
    #     except IndexError:
    #         print ('error')
    #         self.idx = 0
    #         raise StopIteration  # Done iterating.
    # next = __next__  # python2.x compatibility.

    # printing
    def __str__(self):
        '''
        This Method is called when an object of this class is printed.
        It returns a nicely formatted string containing the points
        '''

        # general info including name of the class and it's number of points
        string = str (type(self).__name__) + ', number of points: ' + str (len(self.x )) + '\nx   y   z'
        margin = 3  # how many points are printed before output is clipped

        # add points pairs to the string, but exlcude those that are out of the margin
        for counter, (x, y, z) in enumerate (zip (self.x, self.y, self.z) ):
            if (counter >= margin and counter < len(self.x ) - margin):
                if (counter == margin + 1):
                    string = string + "\n\n...\n"
                pass
            else:
                string = string + '\n' + str(x) + ' ' + str(y) + ' ' + str(z)

        return str (string )

    # def __repr__(self):
    #     return "x, y, z"
