"""
Copyright 2019 Jannik Busse

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.


File description:

Module that contains the NumpyPointCloud class, which effectively wraps together a pointcloud and the description of
it's data fields for precise adressing, addition, replacement and deletion of data columns.

Field naming conventions: Start with a capital letter and separate words with underscores. Examples following.
    "X", "Y", "Z":              Point coordinates
    "Nx", "Ny", "Nz":           Normal vector components
    "Sigma":                    Noise, computed from the smalles eigenvalue in normal computation
    "Intensity":                Reflection intensity
    "Classification":           Point class
    "C2C_absolute_distances":   Distance from a point to it's nearest neigbor

# Example
import numpy as np
from modules.np_pointcloud import NumpyPointCloud

## prepare numpy cloud
numpy_cloud = np.array([[1.1, 2.1, 3.1],
                        [1.2, 2.2, 3.2],
                        [1.3, 2.3, 3.3],
                        [1.4, 2.4, 3.4],
                        [1.5, 2.5, 3.5],
                        [1.6, 2.6, 3.6]] )

## create NumpyPointCloud from numpy cloud values
np_pointcloud = NumpyPointCloud (numpy_cloud, ["X", "Y", "Z"])
print (np_pointcloud )
print ("Has Field Classification: " + str (np_pointcloud.has_fields ("Classification" )))

## add data
np_pointcloud.add_fields ([0, 0, 0, 20, 0, 0], "Classification")
np_pointcloud.add_fields (np.random.uniform (-1, 1, size=(6, 3)), ["Nx", "Ny", "Nz"] )
print (np_pointcloud )
print ("Has Field Classification, Normals: " + str (np_pointcloud.has_fields (["Nx", "Classification", "Ny", "Nz"] )))

## delete data
np_pointcloud.delete_fields ("Classification")
np_pointcloud.delete_fields (["Nx", "Ny", "Nz"])
print (np_pointcloud )
"""

import numpy as np
import warnings


class NumpyPointCloud (object ):
    """
    A Class that contains a 2-D numpy.ndarray and field labels describing the content of the arrays columns for easy
    access

    Field naming conventions: Start with a capital letter and separate words with underscores. Examples following.
        "X", "Y", "Z":              Point coordinates
        "Nx", "Ny", "Nz":           Normal vector components
        "Sigma":                    Noise, computed from the smalles eigenvalue in normal computation
        "Intensity":                Reflection intensity
        "Classification":           Point class
        "C2C_absolute_distances":   Distance from a point to it's nearest neigbor

    Functions:
        get_fields (field_labels_list ):    Extract a copy of one or multiple fields
        has_fields (field_labels_list ):    See if requested fields are in this cloud
        get_xyz_coordinates ():             shorthand for self.get_fields (["X", "Y", "Z"] )
        get_normals ():                     shorthand for self.get_fields (["Nx", "Ny", "Nz"] )
        get_normals_and_sigma ():           shorthand for self.get_fields (["Nx", "Ny", "Nz", "Sigma"] )
        add_fields (field_data, field_labels_list, replace=False ):     Add one or multiple fields
        delete_fields (field_labels_list, warn=True ):                  Delete one or multiple fields
    """

    def __init__(self, numpy_ndarray=None, field_labels_list=[] ):
        """
        Constructor. Creates a cloud consisting of an numpy.ndarray data matrix and a list of field labels to describe
        the data contained in each matrix column.

        Input:
            numpy_ndarray: (np.ndarray)         Input Cloud. Minimum size is (n, 3) for the X, Y, Z point data
            field_labels_list: (list(string))   Labels of the columns of this cloud, describing the type of data
        """

        # Assign a copy of labels and data, remove any spaces around the labels
        self.field_labels = [label.strip () for label in field_labels_list].copy ()
        self.points = numpy_ndarray.copy ()
        #self.shape = self.points.shape  # wraps np.ndarray.shape, this has to be updated with changes
        # (and does not work reliably)

    def __get_indices (self, field_labels_list ):
        """Returns all indices of fields in this cloud that correspond to the requested labels"""

        if (type (field_labels_list) is str):
            field_labels_list = [field_labels_list]

        indices = []
        for field in field_labels_list:
            indices.append (self.field_labels.index(field ))

        return indices

    def has_fields (self, field_labels_list ):
        """Checks if all given field names are in this cloud"""

        # cast string into list
        if (type (field_labels_list) is str):
            field_labels_list = [field_labels_list]

        # See if there is only one entry in the list
        return all (label in self.field_labels for label in field_labels_list )

    def get_fields (self, field_labels_list ):
        """
        Extract one or multiple columns (fields) of this cloud by name. Returns a copy.

        Input:
            field_labels_list: (list(string))    The names of the fields to be returned, in a list

        Output:
            data (numpy.ndarray):   The requested column(s). If one single column, it's shape will be 1 dimensional
        """

        if (type (field_labels_list) is str):
            field_labels_list = [field_labels_list]

        # check if all requested_fields are contained in this cloud
        if (all (field in self.field_labels for field in field_labels_list )):

            # add all indices that correspond to the requested labels
            indices = self.__get_indices (field_labels_list )

        # if something is missing raise an error, so no lengthy computations are done with incomplete data
        else:
            raise ValueError ("This Cloud is missing one or more requested fields: "
                              + str(field_labels_list)
                              + ".\nCloud fields are: " + str(self.field_labels ))

        # return a copy of the requested data
        data = self.points[:, indices].copy ()

        # if only one column is requested, reshape it to 1-D, so boolean comparison doesn't throw an error
        if (data.shape[1] == 1):
            data = data[:, 0]

        return data

    def get_xyz_coordinates (self ):
        """Get the (X, Y, Z) coordinates of this cloud (usually the first three fields)"""
        return self.get_fields (["X", "Y", "Z"] )

    def get_normals (self ):
        """Get the three fields containing the normals of this cloud (Nx, Ny, Nz)"""
        return self.get_fields (["Nx", "Ny", "Nz"] )

    def get_normals_and_sigma (self ):
        """Get the three normal vector fields and the Sigma field containing the noise from normal estimation"""
        return self.get_fields (["Nx", "Ny", "Nz", "Sigma"] )

    def add_fields (self, field_data, field_labels_list, replace=False ):
        """
        Add one or more data fields to this cloud, optionally replacing existing fields.

        Input:
            field_data: (np.ndarray)            Data in the shape of (n, m), where n = number of points in cloud
            field_labels_list: (list(string))   Labels of field_data, length of list = m
            replace: (boolean)                  If True, fields already contained in this cloud are replaced

        Output:
            self: (NumpyPointCloud)             The object of this cloud (optional)
        """

        # # input checks
        # check type of input
        if (type (field_labels_list ) is str ):
            field_labels_list = [field_labels_list]
        if (type (field_data ) is not np.ndarray ):
            field_data = np.array (field_data)  # try casting

        # check dimesions of input data
        if (len (field_data.shape ) < 2 ):
            field_data = field_data.reshape (-1, 1)
        if (self.points.shape[0] != field_data.shape[0] ):
            raise ValueError ("The number of points in the input data is different from this cloud."
                             + "\nNumber of input points: " + str (field_data.shape[0] )
                             + "\nNumber of cloud points: " + str (self.points.shape[0] ))

        # check number of labels
        if (len (field_labels_list ) != field_data.shape[1] ):
            raise ValueError ("Supplied data has " + str (field_data.shape[1] )
                              + " fields, but there were " + str (len (field_labels_list )) + " Labels supplied." )

        # # addition
        # check if any field to be added is already in the cloud
        # create a truth_array that shows which of the labels in field_labels_list are contained in self.field_labels
        truth_array = [new_field_name in self.field_labels for new_field_name in field_labels_list]     # is a list
        if (any(truth_array ) and not replace ):
            raise ValueError ("This Cloud already has fields named " + str (np.array(field_labels_list)[truth_array])
                              + ".\nIf you want to replace the data, set replace=True."
                              + "\nCloud fields are: " + str(self.field_labels ))
        # replace
        elif (any(truth_array ) and replace):
            # replacing the fields already present
            # prepare field_labels_list for indexing and get the indices of the fields already present in the cloud
            field_labels_list = np.array (field_labels_list )
            indices = self.__get_indices (field_labels_list[truth_array] )

            # replace the existing fields with the new data
            self.points[:, indices] = field_data[:, truth_array]

            # concatenate the rest of the fields (which are not already contained in the cloud)
            reversed_truth_array = [not boolean for boolean in truth_array]
            self.points = np.concatenate ((self.points, field_data[:, reversed_truth_array]), axis=1 )
            self.field_labels += field_labels_list[reversed_truth_array].tolist ()

        # simple add
        else:
            self.points = np.concatenate ((self.points, field_data), axis=1 )
            self.field_labels += field_labels_list

        # update shape (does not work reliably)
        #self.shape = self.points.shape

        return self

    def delete_fields (self, field_labels_list, warn=True ):
        """
        Delete one or multiple columns of this cloud by name.

        Input:
            field_labels_list: (list(string))   The names of the fields to be deleted, in a list
            warn: (boolean)                     If True, prints a warning if fields to be deleted could not be found

        Output:
            self: (NumpyPointCloud)             The object of this cloud (optional)
        """

        if (type (field_labels_list) is str):
            field_labels_list = [field_labels_list]

        # check if all requested_fields are contained in this cloud
        truth_array = [field in self.field_labels for field in field_labels_list]
        if (all (truth_array )):
            indices = self.__get_indices (field_labels_list )

        else:

            # issue a warning
            if (warn ):
                message = str ("This Cloud is missing one of the fields to be deleted: "
                              + str(field_labels_list )
                              + ".\nCloud fields are: " + str(self.field_labels )
                              + "\nDeleting fields: " + str(np.array (field_labels_list )[truth_array] ))
                warnings.warn (message )

            # cut away the labels that don't exist and delete the remaining fields
            field_labels_list = np.array (field_labels_list)[truth_array].tolist ()     # convert to array and back
            indices = self.__get_indices (field_labels_list )

        # remove columns and their labels
        self.points = np.delete(self.points, indices, axis=1 )
        for field_to_delete in field_labels_list:
            self.field_labels.remove (field_to_delete )

        # update shape  (does not work reliably)
        #self.shape = self.points.shape

        return self

    def __str__(self ):
        """
        This Method is called when an object of this class is printed.
        It returns a nicely formatted string containing the labels and points.
        """

        # change np printoptions to make linebreaks happen later and suppress -3e10 display style
        np.set_printoptions(precision=6, linewidth=120, suppress=True )
        string = ("\n" + str (self.field_labels ) + "\n" + str (self.points ))
        np.set_printoptions ()

        return string
