# Useful functions for orbital mechanics
#
# Author:  Diogo Merguizo Sanchez
#          The University of Oklahoma
#          dmsanchez@ou.edu
#          2023
# 

import numpy as np
from numba import njit

# Function to normalize angles between 0 and 2*pi
@njit(cache=True, fastmath=True, nogil=True)
def normalize_angle(angle):
    return angle % (2 * np.pi)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Function to convert position and velocity vectors to orbital elements
@njit(cache=True, fastmath=True, nogil=True)
def xyz2orb(mu, r_vec, v_vec):
    # Unpack vectors to scalars for register-level speed (avoids array slicing/creation)
    rx, ry, rz = r_vec[0], r_vec[1], r_vec[2]
    vx, vy, vz = v_vec[0], v_vec[1], v_vec[2]

    # Magnitudes and products
    r_sq = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r_sq)
    v_sq = vx*vx + vy*vy + vz*vz
    
    # Specific angular momentum h = r x v
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h_sq = hx*hx + hy*hy + hz*hz
    h = np.sqrt(h_sq)

    p = h_sq / mu  # Semi-latus rectum

    # Node vector n = k x h = [-hy, hx, 0]
    nx = -hy
    ny = hx
    # nz = 0.0
    n_sq = nx*nx + ny*ny
    n = np.sqrt(n_sq)

    # Eccentricity vector e = ((v x h) / mu) - (r / |r|)
    # v x h components
    vxh_x = vy*hz - vz*hy
    vxh_y = vz*hx - vx*hz
    vxh_z = vx*hy - vy*hx

    ex = (vxh_x / mu) - (rx / r)
    ey = (vxh_y / mu) - (ry / r)
    ez = (vxh_z / mu) - (rz / r)
    e = np.sqrt(ex*ex + ey*ey + ez*ez)

    # Semi-major axis (using vis-viva: v^2 = mu*(2/r - 1/a) => 1/a = 2/r - v^2/mu)
    # Optimization: avoided sqrt(v_sq) earlier just to square it here
    a = 1.0 / (2.0 / r - v_sq / mu)

    # Inclination
    # Clamp value to [-1, 1] to avoid domain errors due to float precision
    inc_arg = hz / h
    if inc_arg > 1.0: inc_arg = 1.0
    elif inc_arg < -1.0: inc_arg = -1.0
    i = np.arccos(inc_arg)

    # Longitude of the ascending node (omega)
    omega = np.arctan2(ny, nx)
    omega = normalize_angle(omega)

    # Argument of perigee (w)
    # Handle the case when n = 0 (equatorial orbit)
    w = 0.0
    if n > 1e-12 and e > 1e-12: # Using epsilon instead of hard 0 for float safety
        # dot(n_vec, e_vec) -> nz is 0, so just nx*ex + ny*ey
        ne = nx * ex + ny * ey
        arg = ne / (n * e)
        if arg > 1.0: arg = 1.0
        elif arg < -1.0: arg = -1.0
        w = np.arccos(arg)
        if ez < 0:
            w = 2 * np.pi - w
    w = normalize_angle(w)

    # True anomaly (theta)
    theta = 0.0
    if e > 1e-12:
        # dot(e_vec, r_vec)
        er = ex * rx + ey * ry + ez * rz
        arg = er / (e * r)
        if arg > 1.0: arg = 1.0
        elif arg < -1.0: arg = -1.0
        theta = np.arccos(arg)
        
        # dot(r, v) check
        rv = rx * vx + ry * vy + rz * vz
        if rv < 0:
            theta = 2 * np.pi - theta
    else:
        # Circular orbit
        arg = rx / r
        if arg > 1.0: arg = 1.0
        elif arg < -1.0: arg = -1.0
        theta = np.arccos(arg)
        if ry < 0:
            theta = 2 * np.pi - theta
            
    theta = normalize_angle(theta)

    # Eccentric anomaly (Ea)
    # Optimization: sqrt((1-e)/(1+e)) can be precomputed if reused, but here it's once
    Ea = 2.0 * np.arctan(np.tan(theta / 2.0) * np.sqrt((1.0 - e) / (1.0 + e)))
    Ea = normalize_angle(Ea)

    # Mean anomaly (Me)
    Me = Ea - e * np.sin(Ea)
    Me = normalize_angle(Me)

    # Return numpy array instead of list for faster unpacking/processing
    return np.array([a, e, i, w, omega, Me])

#-------------------------------------------------------------------------------
# Function to convert orbital elements to the state vector (position and velocity)
@njit(cache=True, fastmath=True, nogil=True)
def orb2xyz(mu, oe):
    # Unpack elements
    a, e, i, w, omega, Me = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    
    p = a * (1.0 - e*e)

    # M is already normalized if coming from xyz2orb, but good to ensure
    E = newton_kepler(Me, e) 

    # Precompute trig
    cosE = np.cos(E)
    sinE = np.sin(E)
    
    # True Anomaly components
    # Optimization: algebraic simplification
    sqrt_1me2 = np.sqrt(1.0 - e*e)
    den = 1.0 - e * cosE
    
    sin_nu = (sqrt_1me2 * sinE) / den
    cos_nu = (cosE - e) / den
    
    # Radius
    r = p / (1.0 + e * cos_nu)

    # Perifocal coordinates (z is always 0)
    # r_perifocal = [r_p_x, r_p_y, 0]
    r_p_x = r * cos_nu
    r_p_y = r * sin_nu
    
    # Velocity factor
    # v_perifocal = [v_p_x, v_p_y, 0]
    factor = np.sqrt(mu / p)
    v_p_x = -factor * sin_nu
    v_p_y = factor * (e + cos_nu)

    # Rotation Matrix Elements
    cw = np.cos(w)
    sw = np.sin(w)
    cO = np.cos(omega)
    sO = np.sin(omega)
    ci = np.cos(i)
    si = np.sin(i)

    # Matrix components (R_11, R_12, etc)
    # R = [[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]]
    # Since perifocal z is 0, we only need columns 1 and 2 of R
    
    # Col 1
    R11 = cO * cw - sO * sw * ci
    R21 = sO * cw + cO * sw * ci
    R31 = sw * si
    
    # Col 2
    R12 = -cO * sw - sO * cw * ci
    R22 = -sO * sw + cO * cw * ci
    R32 = cw * si

    # Manual Matrix-Vector Multiplication (Unrolled)
    # r_vec = R * r_perifocal
    rx = R11 * r_p_x + R12 * r_p_y
    ry = R21 * r_p_x + R22 * r_p_y
    rz = R31 * r_p_x + R32 * r_p_y

    # v_vec = R * v_perifocal
    vx = R11 * v_p_x + R12 * v_p_y
    vy = R21 * v_p_x + R22 * v_p_y
    vz = R31 * v_p_x + R32 * v_p_y

    # Construct result array directly
    return np.array([rx, ry, rz, vx, vy, vz])

#-------------------------------------------------------------------------------
# Transformation from the Ecliptic plane to Planet Equatorial plane
@njit(cache=True, fastmath=True, nogil=True)
def ecl2equ(eps, xecl):
    c = np.cos(eps)
    s = np.sin(eps)
    
    # Direct construction is faster than allocating zeros then filling
    return np.array([
        xecl[0],
        xecl[1] * c - xecl[2] * s,
        xecl[1] * s + xecl[2] * c,
        xecl[3],
        xecl[4] * c - xecl[5] * s,
        xecl[4] * s + xecl[5] * c
    ])

#-------------------------------------------------------------------------------
# Newton's method to solve Kepler's equation
@njit(cache=True, fastmath=True, nogil=True)
def newton_kepler(Me, e, tol=1e-12, nmax=50):
    # Optimization: Reduced nmax (50 is plenty for Newton), removed Pi var
    # Initial guess strategy
    if e < 0.8:
        E = Me
    else:
        E = np.pi

    for _ in range(nmax):
        sE = np.sin(E)
        cE = np.cos(E)
        f = E - e * sE - Me
        f_prime = 1.0 - e * cE
        
        # Halley's method term (optional, but Newton is usually fine)
        # Keeping Newton for speed/simplicity as requested
        delta = f / f_prime
        E -= delta
        
        if np.abs(delta) < tol:
            break

    return E

#-------------------------------------------------------------------------------
# Rotation matrix: perifocal frame to geocentric equatorial frame
@njit(cache=True, fastmath=True)
def RxX(oe):
    # This creates a full array, kept if user needs the explicit matrix
    # Otherwise, orb2xyz logic is preferred
    cw = np.cos(oe[3])
    sw = np.sin(oe[3])
    cO = np.cos(oe[4])
    sO = np.sin(oe[4])
    ci = np.cos(oe[2])
    si = np.sin(oe[2])

    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                cw * si,                 ci]
    ])

#-------------------------------------------------------------------------------
# Function to calculate the Julian Date (JD)
@njit(cache=True, fastmath=True)
def julian_date(year, month, day, hour, minute, second):
    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    # Combined constants where possible
    JD0 = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    frac_day = (hour + minute / 60.0 + second / 3600.0) / 24.0
    
    return JD0 + frac_day

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Function to calculate the initial Greenwich Sidereal Time (GST) at 0h UT
@njit(cache=True, fastmath=True, nogil=True)
def gst0(epoch):
    T0 = (epoch - 2451545.0) / 36525.0
    
    # Optimization: Horner's method for polynomial evaluation
    # 100.4606184 + T0 * (36000.77004 + T0 * (0.000387933 - T0 / 38710000.0))
    # Precomputed division 1/38710000.0 approx 2.5833118e-8
    gst = 100.4606184 + T0 * (36000.77004 + T0 * (0.000387933 - T0 * 2.583311805734952e-08))
    
    return gst % 360.0