import numpy as np
import warnings


class NumpyPointCloud (object ):
    """docstring for NumpyPointCloud ."""

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

        # Assign labels and data, remove any spaces around the labels
        #self.field_labels = np.array ([label.strip () for label in field_labels_list] )
        self.field_labels = [label.strip () for label in field_labels_list]
        self.points = numpy_ndarray

    def __get_indices (self, field_labels_list ):
        "Return all indices that correspond to the requested labels"

        if (type (field_labels_list) is str):
            field_labels_list = [field_labels_list]

        indices = []
        for field in field_labels_list:
            #indices.append (np.where (field_labels_list == field ))
            indices.append (self.field_labels.index(field ))

        return indices

    def get_fields (self, field_labels_list ):
        '''
        Extract multiple rows of this cloud by name.

        Input:
            requested_fields (list(string)):    The names of the fields to be returned, in a list
        '''

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

        # return the requested data
        return self.points[:, indices]

    def add_fields (self, field_data, field_labels_list, replace=False ):
        '''
        Input:
            field_data (np.ndarray):    Data in the shape of (n, m), where n = number of points in cloud
        '''

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

        truth_array = [new_field_name in self.field_labels for new_field_name in field_labels_list]

        if (any(truth_array ) and not replace ):
            raise ValueError ("This Cloud already has fields named \'" + str (np.array(field_labels_list)[truth_array])
                              + "\'."
                              + "\nIf you want to replace the data, set replace=True."
                              + "\nCloud fields are: " + str(self.field_labels ))

        else:
            # delete all fields that need to be overwritten, then append the values to the cloud
            # refactor: This is time consuming and alters the order of fields
            self.delete_fields (field_labels_list, warn=False )

            self.points = np.concatenate ((self.points, field_data), axis=1 )
            self.field_labels += field_labels_list

        return self

    def delete_fields (self, field_labels_list, warn=True ):
        '''
        Delete multiple rows of this cloud by name.

        Input:
            requested_fields (list(string)):    The names of the fields to be returned, in a list
        '''

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

    def __str__(self ):
        '''
        This Method is called when an object of this class is printed.
        It returns a nicely formatted string containing the labels and points.
        '''

        return ("\n" + str (self.field_labels )
                + "\n" + str (self.points ))
