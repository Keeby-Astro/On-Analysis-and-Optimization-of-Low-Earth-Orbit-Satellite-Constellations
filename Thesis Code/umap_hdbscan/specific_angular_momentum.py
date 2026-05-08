import numpy as np

MU = 398600.4418 # Gravitational parameter for Earth (km^3/s^2)

def specific_angular_momentum(a, e):
    """
    Calculate the specific angular momentum (scalar) given semi-major axis (km) and eccentricity.
    """
    return np.sqrt(MU * a * (1 - e**2))

def specific_angular_momentum_vector(a_array, e_array):
    """
    Vectorized calculation of specific angular momentum for arrays.
    a_array in km, e_array dimensionless. Returns array of h in km^2/s.
    """
    return np.sqrt(MU * a_array * (1.0 - e_array**2))