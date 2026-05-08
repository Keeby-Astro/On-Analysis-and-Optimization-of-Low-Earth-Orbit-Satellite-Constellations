# Function to calculate the atmospheric density
# Model USSA1976
# MATLAB original code by: Howard D. Curtis, Orbital Mechanics for Engineering Students, 4th Revised Edition, Elsevier, 2020
# Translated to Python by: Diogo Merguizo Sanchez, The University of Oklahoma, 2024
#
# Inputs:
#        z: altitude[km]
# Outputs:
#        rho: atmospheric density[kg/m^3]

import numpy as np
from numba import njit

# Module-level arrays
h = np.array([0.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
              90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
              180.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0,
              500.0, 600.0, 700.0, 800.0, 900.0, 1000.0], dtype=np.float64)

r = np.array([1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3,
              3.097e-4, 8.283e-5, 1.846e-5, 3.416e-6, 5.606e-7,
              9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 2.076e-9,
              5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12,
              2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14,
              1.136e-14, 5.759e-15, 3.561e-15], dtype=np.float64)

H = np.array([7.310, 6.427, 6.546, 7.360, 8.342, 7.583,
              6.661, 5.927, 5.533, 5.703, 6.782, 9.973,
              13.243, 16.322, 21.652, 27.974, 34.934, 43.342,
              49.755, 54.513, 58.019, 60.980, 65.654, 76.377,
              100.587, 147.203, 208.020], dtype=np.float64)

@njit(cache=True)
def rho_atm(z):
    # Clip altitude to [0, 1000]
    z = max(0.0, min(1000.0, z))
    # Find the index i such that h[i] <= z < h[i + 1] using binary search
    if z >= h[-1]:
        i = len(h) - 2
    else:
        lo = 0
        hi = len(h) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if z < h[mid]:
                hi = mid - 1
            elif z >= h[mid + 1]:
                lo = mid + 1
            else:
                lo = mid
                break
        i = lo
        if i > len(h) - 2:
            i = len(h) - 2
    # Calculate the atmospheric density
    rho = r[i] * np.exp(-(z - h[i]) / H[i])
    return rho

# Test the function
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Altitude range
    z_values = np.arange(0, 101, 1)

    # Calculate the atmospheric density for all altitudes
    rho_values = np.array([rho_atm(z) for z in z_values])
    for zi, rhoi in zip(z_values, rho_values):
        print(f"z = {zi} km, rho = {rhoi} kg/m^3")

    # Plot the atmospheric density
    plt.figure()
    plt.plot(z_values, rho_values, 'b')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Atmospheric density [kg/m^3]')
    plt.title('Atmospheric Density - USSA1976')
    plt.grid(True)
    plt.show()