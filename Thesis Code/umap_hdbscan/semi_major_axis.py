import numpy as np

MU = 398600.4418 # Gravitational parameter for Earth (km^3/s^2)

def semi_major_axis(mean_motion):
    """
    Calculate the semi-major axis (scalar) given mean motion (revolutions per day).
    """
    mean_motion_rad = mean_motion * 2.0 * np.pi / 86400.0
    a = (MU ** (1.0 / 3.0)) / (mean_motion_rad ** (2.0 / 3.0))
    return a

def semi_major_axis_vector(mean_motion_array):
    """
    Vectorized calculation of semi-major axis for an array of mean motions.
    mean_motion_array is in rev/day. Returns array of semi-major axes in km.
    """
    mean_motion_rad = mean_motion_array * 2.0 * np.pi / 86400.0
    a_array = (MU ** (1.0 / 3.0)) / (mean_motion_rad ** (2.0 / 3.0))
    return a_array