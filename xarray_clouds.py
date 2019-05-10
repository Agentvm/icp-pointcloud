import numpy as np
#import pandas as pd
import xarray as xr


def create_xyz_cloud (numpy_cloud ):
    '''
    Takes a numpy cloud and returns a labelled xarray with dimensions 'length' and 'fields',
    and 'fields': ['x', 'y', 'z']
    '''

    # check
    if (numpy_cloud.shape[1] < 3):
        raise ValueError("The given numpy array only has " + str (numpy_cloud.shape[1]) + " rows. Aborting.")

    # return array
    return xr.DataArray(numpy_cloud[:, 0:3],
                        coords={'fields': ['x', 'y', 'z']},
                        dims=('length', 'fields'),
                        name="PointCloud_XYZ")


def create_xyzi_cloud (numpy_cloud ):
    '''
    Takes a numpy cloud and returns a labelled xarray with dimensions 'length' and 'fields',
    and 'fields': ['x', 'y', 'z', 'i']
    '''

    # check
    if (numpy_cloud.shape[1] < 4):
        raise ValueError("The given numpy array only has " + str (numpy_cloud.shape[1]) + " rows. Aborting.")

    # return array
    return xr.DataArray(numpy_cloud[:, 0:4],
                        coords={'fields': ['x', 'y', 'z', 'i']},
                        dims=('length', 'fields'),
                        name="PointCloud_XYZI")


def create_xyzrgb_cloud (numpy_cloud ):
    '''
    Takes a numpy cloud and returns a labelled xarray with dimensions 'length' and 'fields',
    and 'fields': ['x', 'y', 'z', 'r', 'g', 'b']
    '''

    # check
    if (numpy_cloud.shape[1] < 6):
        raise ValueError("The given numpy array only has " + str (numpy_cloud.shape[1]) + " rows. Aborting.")

    # return array
    return xr.DataArray(numpy_cloud[:, 0:6],
                        coords={'fields': ['x', 'y', 'z', 'r', 'g', 'b']},
                        dims=('length', 'fields'),
                        name="PointCloud_XYZRGB")


def has_field (xarray, coordinate ):
    '''
    Hacky.

    Input:
        xarray (xarray):        The xarray DataArray
        coordinate (String):    Field name

    Output:
        Boolean
    '''
    answer = True
    try:
        xarray.sel(fields=[coordinate] )
    except:
        answer = False
    return answer


# main
if __name__ == "__main__":

    # numpy
    numpy_cloud = np.array([[1.1, 2.1, 3.1],
                            [1.2, 2.2, 3.2],
                            [1.3, 2.3, 3.3],
                            [1.4, 2.4, 3.4],
                            [1.5, 2.5, 3.5],
                            [1.6, 2.6, 3.6]] )

    # fill
    data = create_xyz_cloud (numpy_cloud )

    # print
    # full thing
    print ("data:\n" + str(data ))

    # get x by index
    print ("\n--------------------------------\ndata[]: " + str(data[:, 0] ))

    # get x in range                                                       length, fields
    print ("\n--------------------------------\ndata.loc: " + str(data.loc[0:5,   ['x', 'y']] ))

    # get x by label
    print ("\n--------------------------------\ndata.sel: " + str(data.sel(fields=['x'] )))

    # get x true false thingy
    print ("\n--------------------------------\ndata.where(data[]: "
    + str (data.where(data[:, 0] < 1.3, drop=True )))
    print ("\n--------------------------------\ndata.where(data.sel: "
    + str (data.where(data.sel(fields=['x'] ) < 1.3, drop=True )))

    # check for coordinate
    print ("\n--------------------------------\ncheck: " + str (has_field (data, "x" )) )
    print ("\n--------------------------------\ncheck: " + str (has_field (data, "lolo" )) + "\n" )
