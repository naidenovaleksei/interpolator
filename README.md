interpolator
============

Multidimensional linear interpolation for numpy arrays


Create a multidimensional linear interpolator on numpy arrays.

To create an interpolator, you need two inputs:
X : coordinates. It must be a tuple containing vectors that define the coordinates for each dimension
Y : data values. It can be a tuple or dictionary of numpy.ndarrays.

Example:

Suppose we load the coordinate arrays for some high-dimensional data. In this case we are working with 4D datasets. The arrays must be strictly monotonically increasing, not necessarily equally spaced.

```
latitude, longitude, depth, time = load_coords(some_file1) 
```

And we also load the datasets. Each of these are numpy ndarrays, all same shape.

```
Temperature, Pressure, Density = load_data(some_file2)
```

Now, we create the coordinate tuple.

```
coords = (latitude, longitude, depth, time)
```

The shape of each ndarray must be in the right order. That is, if Temperature is a N1 x N2 x N3 x N4 ndarray, then the lenght of latitude must be N1, the length of longitude must be N2, etc.

Then we can created the dataset dictionary:

```
data = dict()
data['Temperature'] = Temperature
data['Pressure']    = Pressure
data['Density']     = Density
```

But it can also be a tuple:

```
data = (Temperature, Pressure, Density)
```

If you are interested in interpolating only one ndarray, you can use a 1-tuple. For example: ```data = (Temperature,)```

Now, we construct the interpolator f:

```
f = interpolator.multilinear_interpolator( coords, data )
```

Now one can call the interpolator at any coordinate. For example

```
(flag, output) = f( [ 29.97, -45.0, 1.5, 101 ] )
```

flag is true if the coordinates are outside the domain specified by coords
flag is false if the interpolation is done successfully

If data is a dictionary, output will be a dictionary as well, with the interpolated values for each
of the keys contained in the original dictionary. Example:

```
print output

{'Temperature': 35.645003322189929,
 'Pressure': 10.224311117453488,
 'Density': 0.030242050807863668}
```

If data is a tuple, output will be a numpy array with the interpolated values,
indexed the same way as the original tuple. Example:

```
print output

array([35.645003322189929, 10.224311117453488, 0.030242050807863668])
```


