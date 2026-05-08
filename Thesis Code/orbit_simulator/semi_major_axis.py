import numpy as np
from major_bodies_parameters_optimized import constants

earth_mass = 5.9722e24
const = constants()
G = const.G  # gravitational constant [km^3/s^2]

mu = G * earth_mass  # Gravitational parameter for Earth (km^3/s^2)

def semi_major_axis(mean_motion):
    """
    Calculate the semi-major axis (scalar) given mean motion (revolutions per day).
    """
    mean_motion_rad = mean_motion * 2.0 * np.pi / 86400.0
    a = (mu ** (1.0 / 3.0)) / (mean_motion_rad ** (2.0 / 3.0))
    return a

def semi_major_axis_vector(mean_motion_array):
    """
    Vectorized calculation of semi-major axis for an array of mean motions.
    mean_motion_array is in rev/day. Returns array of semi-major axes in km.
    """
    mean_motion_rad = mean_motion_array * 2.0 * np.pi / 86400.0
    a_array = (mu ** (1.0 / 3.0)) / (mean_motion_rad ** (2.0 / 3.0))
    return a_array