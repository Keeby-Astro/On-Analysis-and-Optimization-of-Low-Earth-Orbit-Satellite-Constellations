import numpy as np

def mean_to_true_anomaly(M, e):
    """
    Convert a single mean anomaly (M, radians) and eccentricity (e) to true anomaly (degrees).
    Uses iterative approach to solve Kepler's equation.
    """
    epsilon = 1e-8
    # Good initial guess for E:
    E = M + 0.5*e if M < np.pi else M - 0.5*e
    ratio = 1.0

    while abs(ratio) > epsilon:
        ratio = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= ratio

    true_anomaly = 2.0 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0))
    # Convert to 0-360 range
    true_anomaly = np.mod(true_anomaly, 2 * np.pi)
    return np.degrees(true_anomaly)

def mean_to_true_anomaly_vector(M_array, e_array):
    """
    Vectorized version of mean_to_true_anomaly. M_array in radians, e_array dimensionless.
    Returns array of true anomalies in degrees.
    """
    true_anomalies = np.zeros_like(M_array)
    epsilon = 1e-8

    for i in range(M_array.size):
        M = M_array[i]
        e = e_array[i]
        E = M + 0.5*e if M < np.pi else M - 0.5*e
        ratio = 1.0

        while abs(ratio) > epsilon:
            ratio = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E -= ratio

        nu = 2.0 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0))
        nu = np.mod(nu, 2 * np.pi)
        true_anomalies[i] = np.degrees(nu)

    return true_anomalies