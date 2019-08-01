"""
Module that contains the NumpyPointCloud class, which effectively wraps together a pointcloud and the description of
it's data fields for precise adressing, addition, replacement and deletion of data rows.

Field naming conventions: Start with a capital letter and separate words with underscores. Examples following.
    "X", "Y", "Z":              Point coordinates
    "Nx", "Ny", "Nz":           Normal vector components
    "Sigma":                    Noise, computed from the smalles eigenvalue in normal computation
    "Intensity":                Reflection intensity
    "Classification":           Point class
    "C2C_absolute_distances":   Distance from a point to it's nearest neigbor
"""

import numpy as np
import warnings


class NumpyPointCloud (object ):
    """
    A Class that contains a 2-D numpy.ndarray and field labels describing the content of the arrays rows for easy access

    Field naming conventions: Start with a capital letter and separate words with underscores. Examples following.
        "X", "Y", "Z":              Point coordinates
        "Nx", "Ny", "Nz":           Normal vector components
        "Sigma":                    Noise, computed from the smalles eigenvalue in normal computation
        "Intensity":                Reflection intensity
        "Classification":           Point class
        "C2C_absolute_distances":   Distance from a point to it's nearest neigbor

    Functions:
        get_fields (field_labels_list ):    Extract a copy of one or multiple fields
        get_xyz_coordinates ():             shorthand for self.get_fields (["X", "Y", "Z"] )
        get_normals ():                     shorthand for self.get_fields (["Nx", "Ny", "Nz"] )
        get_normals_and_sigma ():           shorthand for self.get_fields (["Nx", "Ny", "Nz", "Sigma"] )
        add_fields (field_data, field_labels_list, replace=False ):     Add one or multiple fields
        delete_fields (field_labels_list, warn=True ):                  Delete one or multiple fields
    """

    def __init__(self, numpy_ndarray=None, field_labels_list=[] ):
        """
        Constructor. Creates a cloud consisting of an numpy.ndarray data matrix and a list of field labels to describe
        the data contained in each matrix row.

        Input:
            numpy_ndarray (np.ndarray):         Input Cloud. Minimum size is (n, 3) for the X, Y, Z point data
            field_labels_list (list(string)):   Labels of the rows of this cloud, describing the type of data contained
        """

        # # Inheritance
        # super (NumpyPointCloud, self ).__init__ ()

        # Assign a copy of labels and data, remove any spaces around the labels
        self.field_labels = [label.strip () for label in field_labels_list].copy ()
        self.points = numpy_ndarray.copy ()
        self.shape = self.points.shape  # wraps np.ndarray.shape, this has to be updated with changes

    def __get_indices (self, field_labels_list ):
        """Returns all indices of fields in this cloud that correspond to the requested labels"""

        if (type (field_labels_list) is str):
            field_labels_list = [field_labels_list]

        indices = []
        for field in field_labels_list:
            indices.append (self.field_labels.index(field ))

        return indices

    def get_fields (self, field_labels_list ):
        """
        Extract one or multiple rows (fields) of this cloud by name. Returns a copy.

        Input:
            field_labels_list (list(string)):    The names of the fields to be returned, in a list

        Output:
            data (numpy.ndarray):   The requested row(s). If only one row is fetched, it's shape will be 1 dimensional
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

        # if only one row is requested, reshape it to 1-D, so boolean comparison doesn't throw an error
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
            field_data (np.ndarray):            Data in the shape of (n, m), where n = number of points in cloud
            field_labels_list (list(string)):   Labels of field_data, length of list = m
            replace (boolean):                  If True, fields already contained in this cloud are replaced

        Output:
            self (NumpyPointCloud):             The object of this cloud (optional)
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

        # update shape
        self.shape = self.points.shape

        return self

    def delete_fields (self, field_labels_list, warn=True ):
        """
        Delete one or multiple rows of this cloud by name.

        Input:
            field_labels_list (list(string)):   The names of the fields to be deleted, in a list
            warn (boolean):                     If True, prints a warning if fields to be deleted could not be found

        Output:
            self (NumpyPointCloud):             The object of this cloud (optional)
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

        # remove rows and their labels
        self.points = np.delete(self.points, indices, axis=1 )
        for field_to_delete in field_labels_list:
            self.field_labels.remove (field_to_delete )

        # update shape
        self.shape = self.points.shape

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
