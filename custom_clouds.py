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
        super(XYZCloud, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.idx = 0

    # numpy init function
    @classmethod
    def from_numpy_array (cls, input_cloud_numpy):

        x = input_cloud_numpy[:, 0].tolist ()
        y = input_cloud_numpy[:, 1].tolist ()
        z = input_cloud_numpy[:, 2].tolist ()

        return cls (x, y, z )

    def __iter__(self):
        #return self
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

    def __str__(self):
        string = str (type(self).__name__) + '\nx   y   z'
        for (x, y, z) in zip (self.x, self.y, self.z):
            string = string + '\n' + str(x) + ' ' + str(y) + ' ' + str(z)

        return str (string )

    # def __repr__(self):
    #     return "x, y, z"
