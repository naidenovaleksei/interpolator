__author__ = 'Fernando Gonzalez del Cueto'

import numpy as np


def bincombs(N):
    """
    Creates a list of all possible binary arrays of dimension N.
    This is used by the multilinear interpolator to find all neighboring grid points of a point.
    """
    L = []
    tmp = np.zeros(N, dtype=int)
    for j1 in range(2 ** N):
        s = bin(j1)[2:]

        for j2 in range(1, len(s) + 1):
            tmp[-j2] = s[-j2]
        L.append(tuple(tmp))
    return L


class multilinear_interpolator:
    """
    Create a multidimensional linear interpolator on numpy arrays.

    To create an interpolator, you need two inputs:
    X : coordinates. It must be a tuple containing vectors that define the coordinates for each dimension
    Y : data values. It can be a tuple or dictionary of numpy.ndarrays.

    Example:

    latitude, longitude, depth, time = load_coords(some_file1) # all these are numpy arrays.
                                                               # Monotonically increasing, not
                                                               # necessarily equally spaced.
    Temperature, Pressure, Density = load_data(some_file2) # each of these are numpy ndarrays, all same shape.

    coords = (latitude, longitude, depth, time)
    data = dict()
    data['Temperature'] = Temperature
    data['Pressure']    = Pressure
    data['Density']     = Density

    # or it can also be a tuple:

    data = (Temperature, Pressure, Density)

    f = multilinear_interpolator( coords, data )

    # Then one can call the interpolator at any coordinate for example

    (error_flag, output) = f( [ 29.97, -45.0, 1.5, 101 ] )

    error_flag is True if the coordinates are outside the domain specified by coords
    error_flag is False if the interpolation is done successfully

    # If data is a dictionary, output will be a dictionary as well, with the interpolated values for each
    # of the keys contained in the original dictionary. Example:

    print output

    {'Temperature': 35.645003322189929,
     'Pressure': 10.224311117453488,
     'Density': 0.030242050807863668}

    # If data is a tuple, output will be a numpy array with the interpolated values,
    # indexed the same way as the original tuple. Example:

    print output

    array([35.645003322189929, 10.224311117453488, 0.030242050807863668])


    # any questions please contact Fernando del Cueto: fcueto@gmail.com
    # May 1, 2014
    # Houston, TX
    # http://github.com/fejikso/interpolator
    """

    # Constructor
    def __init__(self, X, Y):
        assert isinstance(X, tuple)

        self.dim = len(X)

        self.n_variables = len(Y)

        self.X = X
        self.Y = Y
        if isinstance(Y, tuple) or isinstance(Y, list):
            self.output_type = 'list_tuple'
        else:
            self.output_type = 'dict'

        self.neighbors = bincombs(self.dim)

        self.dimlen = list()
        for k in range(0, self.dim):
            #store dimensions for each coordinate vector
            self.dimlen.append(len(X[k]))

    def __call__(self, coords):
        if len(coords) != self.dim:
            raise Exception('Wrong dimensions')

        ind = []
        alpha = np.ndarray(self.dim)
        breakflag = False

        for k in range(0, self.dim):

            float_index = np.interp(coords[k], self.X[k], np.arange(0, self.dimlen[k]), left=np.nan, right=np.nan)

            if np.isnan(float_index) or (float_index >= self.dimlen[k]):
                # error: trying to access data outside data domain. Return nan
                breakflag = True
                break
            else:
                tmp = float_index
                ind.append(tmp)
                # alpha[k] = 1.0 - (float_index - ind[k])

        beta = np.ndarray(self.dim)
        if breakflag:
            # coordinates outside domain. Return nan

            return (True, np.nan)
        else:

            #initialize output array

            if self.output_type == 'list_tuple':
                output = np.zeros(self.n_variables)
            elif self.output_type == 'dict':
                output = dict(zip(self.Y.keys(), np.zeros(len(self.Y.keys()))))
            else:
                raise Exception('Unknown output type')

            J = np.zeros(self.dim, dtype=int)

            for c in self.neighbors:

                for jx in range(self.dim):
                    J[jx] = int(np.floor(ind[jx])) + c[jx]
                    beta[jx] = 1.0 - abs(J[jx] - ind[jx])

                gamma = np.prod(beta)

                if self.output_type == 'list_tuple':
                    for jv in range(self.n_variables):
                        if gamma and tuple(J + 1) < self.Y[0].shape:
                            output[jv] = output[jv] + gamma * self.Y[jv][tuple(J)]
                elif self.output_type == 'dict':
                    for jv in self.Y.keys():
                        if gamma and tuple(J + 1) < self.Y[0].shape:
                            output[jv] = output[jv] + gamma * self.Y[jv][tuple(J)]
                else:
                    raise Exception('Unknown output type')

            return (False, output)
