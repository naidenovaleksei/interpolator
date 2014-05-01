__author__ = 'Fernando Gonzalez del Cueto'

import numpy as np


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

    (flag, output) = f( [ 29.97, -45.0, 1.5, 101 ] )

    flag is true if the coordinates are outside the domain specified by coords
    flag is false if the interpolation is done successfully

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
    """

    # Constructor
    def __init__(self, X, Y):
        assert isinstance(X, tuple)

        self.dim = len(X)
        self.dimlen = np.zeros(self.dim, dtype=np.int64)

        self.n_variables = len(Y)

        self.X = X
        self.Y = Y
        if isinstance(Y, dict):
            self.output_type = 'dict'
        elif isinstance(Y, tuple) or isinstance(Y, tuple):
            self.output_type = 'list_tuple'
        else:
            raise Exception('Y must be dictionary, list or tuple')

        for k in range(0, self.dim):
            self.dimlen[k] = len(X[k])

    def __call__(self, coords):
        if len(coords) != self.dim:
            raise Exception('Wrong dimensions')

        ind = np.zeros(self.dim, dtype=np.int32)
        alpha = np.ndarray(self.dim)
        breakflag = False
        for k in range(0, self.dim):
            tmp = np.interp(coords[k], self.X[k], range(0, self.dimlen[k]), left=np.nan, right=np.nan)

            if np.isnan(tmp):
                # error: trying to access data outside data domain. Return nan
                breakflag = True
                break
            else:
                ind[k] = np.floor(tmp)
                alpha[k] = 1 - (tmp - ind[k])

        if breakflag:
            # coordinates outside domain. Return nan

            return (True, np.nan)
        else:

            if self.output_type == 'dict':
                #initialize output dictionary
                output = dict(zip(self.Y.keys(), np.zeros(len(self.Y.keys()))))

                beta = alpha.copy()
                I = ind.copy()

                for jx in range(self.dim):

                    for jb in range(0, 2):

                        if jb == 0:
                            beta[jx] = alpha[jx]
                            I[jx] = I[jx]
                        else:
                            beta[jx] = 1.0 - alpha[jx]
                            I[jx] = I[jx] + jb

                        gamma = np.prod(beta)

                        for jv in self.Y.keys():
                            output[jv] = output[jv] + gamma * self.Y[jv][tuple(I)]

                return (False, output)
            elif self.output_type == 'list_tuple':
                #initialize output array

                output = np.zeros(self.n_variables)

                beta = alpha.copy()
                I = ind.copy()

                for jx in range(self.dim):

                    for jb in range(0, 2):

                        if jb == 0:
                            beta[jx] = alpha[jx]
                            I[jx] = I[jx]
                        else:
                            beta[jx] = 1.0 - alpha[jx]
                            I[jx] = I[jx] + jb

                        gamma = np.prod(beta)

                        for jv in range(self.n_variables):
                            output[jv] = output[jv] + gamma * self.Y[jv][tuple(I)]

                return (False, output)
            else:
                raise Exception('Do not know how to process output type: %s ' % self.output_type)

