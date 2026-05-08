import numpy as np
from major_bodies_parameters_optimized import constants

earth_mass = 5.9722e24
const = constants()
G = const.G  # gravitational constant [km^3/s^2]

mu = G * earth_mass  # Gravitational parameter for Earth (km^3/s^2)

def specific_angular_momentum(a, e):
    """
    Calculate the specific angular momentum (scalar) given semi-major axis (km) and eccentricity.
    """
    return np.sqrt(mu * a * (1 - e**2))

def specific_angular_momentum_vector(a_array, e_array):
    """
    Vectorized calculation of specific angular momentum for arrays.
    a_array in km, e_array dimensionless. Returns array of h in km^2/s.
    """
    return np.sqrt(mu * a_array * (1.0 - e_array**2))