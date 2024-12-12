"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with positive values outside the shape and negative values inside.
    """

    # -----------------------------------------------------------------------------------------------------------
    
    # create a linear space for both dimensions which would create a unit cube centered at 0
    x_range = y_range = z_range = np.linspace(-0.5, 0.5, resolution)

    # define seperate grids, indexing provides column major indexing
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # to be compatible with afore SDF implementation, convert to flat inputs that are arrays of coordinates
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    # subsequently generating an array of SDF values, reshape to the same resolution as cube to have a 3D volume
    sdf_values = sdf_function(grid_x, grid_y, grid_z).reshape((resolution, resolution, resolution))
    return sdf_values
    # -----------------------------------------------------------------------------------------------------------
