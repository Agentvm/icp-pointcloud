import numpy as np
import pandas as pd
import xarray as xr


def create_xyz_cloud (numpy_cloud ):
    return xr.DataArray(numpy_cloud, coords={'fields': ['x', 'x', 'z']}, dims=('length', 'fields'))


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
print ("data:\n" + str(data ))
print ("data data:\n" + str(data.values ))
