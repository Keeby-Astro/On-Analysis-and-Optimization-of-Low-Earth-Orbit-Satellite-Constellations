import numpy as np
from control_optimized import pause
from numba import njit
from USSA76_optimized import rho_atm

# This file contains the functions pertaining to the perturbations that
# can act on a satellite (or spacecraft) orbit around the Earth.

#---------------------------------------------------------------------------------
# Perturbation from other bodies in the system, often called 3rd-body perturbation
@njit(cache=True, fastmath=True, nogil=True)
def AC3b(rsc_vec, rd_vec, GMd, out):
    # Component-wise implementation to avoid temporary arrays.
    dx = rd_vec[0] - rsc_vec[0]
    dy = rd_vec[1] - rsc_vec[1]
    dz = rd_vec[2] - rsc_vec[2]

    rho2 = dx * dx + dy * dy + dz * dz
    rho = np.sqrt(rho2)
    # Optimization: 1/(rho^3) is faster as 1/(rho2 * rho)
    inv_rho3 = 1.0 / (rho2 * rho)

    rd2 = rd_vec[0] * rd_vec[0] + rd_vec[1] * rd_vec[1] + rd_vec[2] * rd_vec[2]
    rd = np.sqrt(rd2)
    inv_rd3 = 1.0 / (rd2 * rd)

    # GMd distributed
    out[0] = GMd * (dx * inv_rho3 - rd_vec[0] * inv_rd3)
    out[1] = GMd * (dy * inv_rho3 - rd_vec[1] * inv_rd3)
    out[2] = GMd * (dz * inv_rho3 - rd_vec[2] * inv_rd3)

#---------------------------------------------------------------------------------
# Function to calculate the acceleration due to the direct solar radiation
# pressure (SRP) with shadow effect.
@njit(cache=True, fastmath=True, nogil=True)
def SRPacc(xsat, xsun, AtoM, Cr, Re, out):
    # Constants
    AU = 149597870.700
    Psun = 4.56316e-6
    reqsun = 695700.0
    
    # Precompute SRP constant
    SRP_const = Psun * AtoM * Cr * 1e-3

    # Scalar expansion for Rsat
    xs0, xs1, xs2 = xsat[0], xsat[1], xsat[2]
    Rsat2 = xs0 * xs0 + xs1 * xs1 + xs2 * xs2
    Rsat = np.sqrt(Rsat2)

    # Relative position spacecraft ---> Sun
    dx = xsun[0] - xs0
    dy = xsun[1] - xs1
    dz = xsun[2] - xs2
    Rsatsun2 = dx * dx + dy * dy + dz * dz
    Rsatsun = np.sqrt(Rsatsun2)

    # Shadow effect coefficient calculation (Montenbruck & Gill, 2001)
    
    # Dot product for angle
    dotprod = -(xs0 * dx + xs1 * dy + xs2 * dz)
    
    # Optimization: Early exit for un-shadowed regions (if satellite is between sun and earth)
    # If dotprod is negative, the Earth is "behind" the satellite relative to the Sun
    # so no shadow is possible.
    if dotprod < 0:
        nu = 1.0
    else:
        # Apparent radius of the Sun
        a = np.arcsin(reqsun / Rsatsun)
        # Apparent radius of the planet
        b = np.arcsin(Re / Rsat)
        # Apparent separation
        arg = dotprod / (Rsat * Rsatsun)
        # Clamp arg
        if arg > 1.0: arg = 1.0
        elif arg < -1.0: arg = -1.0
        c = np.arccos(arg)

        # Penumbra/Umbra logic
        if (a + b) <= c:
            nu = 1.0
        elif c < abs(a - b):
            # Checking if inside Earth or Sun (unlikely for Sun, likely for Earth eclipse)
            # If a < b (Sun smaller than Earth visually), total eclipse
            nu = 0.0
        else:
            # Partial shadow
            # Optimization: Re-use squares
            c2 = c*c
            a2 = a*a
            b2 = b*b
            x_val = (c2 + a2 - b2) / (2.0 * c)
            y2 = a2 - x_val * x_val
            if y2 < 0.0: y2 = 0.0
            y = np.sqrt(y2)

            Area = a2 * np.arccos(x_val / a) + b2 * np.arccos((c - x_val) / b) - c * y
            nu = 1.0 - Area / (np.pi * a2)

        if np.isnan(nu):
            nu = 0.0

    # Acceleration due to the SRP
    # factor = (AU / Rsatsun)**2
    # scale = -nu * SRP0 * factor / Rsatsun 
    # Combined: -nu * SRP0 * AU^2 / Rsatsun^3
    
    scale = (-nu * SRP_const * AU * AU) / (Rsatsun2 * Rsatsun)
    
    out[0] = scale * dx
    out[1] = scale * dy
    out[2] = scale * dz

#-------------------------------------------------------------------------------
# Specialized fast path for the common degree-2/order-0 case.
# This keeps the same normalized-harmonics convention used by EGM2008.
@njit(cache=True, fastmath=True, nogil=True)
def _EGM2008_n2_m0_fast(xi, C, t, GM, Re, spin, srtime0, out):
    mst = sidereal_time(t, spin, srtime0)
    smst = np.sin(mst)
    cmst = np.cos(mst)

    x0 = xi[0] * cmst + xi[1] * smst
    x1 = -xi[0] * smst + xi[1] * cmst
    x2 = xi[2]

    r2 = x0 * x0 + x1 * x1 + x2 * x2
    r = np.sqrt(r2)
    inv_r = 1.0 / r
    ct = x2 * inv_r

    st2 = 1.0 - ct * ct
    if st2 < 0.0:
        st2 = 0.0
    st = np.sqrt(st2)
    if st < 1e-16:
        st = 1e-16

    lamb = np.arctan2(x1, x0)
    sl = np.sin(lamb)
    cl = np.cos(lamb)

    c20 = C[2, 0]
    sqrt5 = 2.23606797749979
    p20 = 0.5 * sqrt5 * (3.0 * ct * ct - 1.0)
    pl20 = -3.0 * sqrt5 * ct * st

    q2 = (Re * inv_r) * (Re * inv_r)
    GMoR = GM * inv_r

    Vr = -GMoR * inv_r * (3.0 * q2 * p20 * c20)
    Vt = GMoR * (q2 * pl20 * c20)
    Vl = 0.0

    inv_r_st = inv_r / st
    ac0 = st * cl * Vr + ct * cl * Vt * inv_r - sl * Vl * inv_r_st
    ac1 = st * sl * Vr + ct * sl * Vt * inv_r + cl * Vl * inv_r_st
    ac2 = ct * Vr - st * Vt * inv_r

    out[0] = ac0 * cmst - ac1 * smst
    out[1] = ac0 * smst + ac1 * cmst
    out[2] = ac2

@njit(cache=True, fastmath=True, nogil=True)
def _EGM2008_n2_m0_reference_generic(xi, C, S, t, GM, Re, spin, srtime0, P, Pl, sml, cml, out):
    mst = sidereal_time(t, spin, srtime0)
    smst = np.sin(mst)
    cmst = np.cos(mst)

    x0 = xi[0] * cmst + xi[1] * smst
    x1 = -xi[0] * smst + xi[1] * cmst
    x2 = xi[2]
    r2 = x0 * x0 + x1 * x1 + x2 * x2
    r = np.sqrt(r2)

    inv_r = 1.0 / r
    ct = x2 * inv_r

    st2 = 1.0 - ct * ct
    if st2 < 0.0:
        st2 = 0.0
    st = np.sqrt(st2)
    if st < 1e-16:
        st = 1e-16

    lamb = np.arctan2(x1, x0)
    sl = np.sin(lamb)
    cl = np.cos(lamb)

    GMoR = GM * inv_r
    q_base = Re * inv_r

    P[0, 0] = 1.0
    sqrt3 = 1.7320508075688772
    P[1, 0] = sqrt3 * ct
    P[1, 1] = sqrt3 * st
    Pl[0, 0] = 0.0
    Pl[1, 0] = -P[1, 1]
    Pl[1, 1] = P[1, 0]

    dn = 2.0
    dm = 0.0
    denom_diff = dn - dm
    denom_sum = dn + dm
    q1 = (2.0 * dn - 1.0) * (2.0 * dn + 1.0) / (denom_diff * denom_sum)
    q2 = (2.0 * dn + 1.0) * (denom_sum - 1.0) * (denom_diff - 1.0) / (
        denom_diff * denom_sum * (2.0 * dn - 3.0)
    )
    anm = np.sqrt(q1)
    bnm = np.sqrt(q2)
    p20 = anm * ct * P[1, 0] - bnm * P[0, 0]
    P[2, 0] = p20
    fnm = (2.0 * dn + 1.0) / anm
    Pl[2, 0] = dn * ct / st * p20 - fnm * P[1, 0] / st

    sml[0] = 0.0
    cml[0] = 1.0

    qn = q_base * q_base
    C20 = C[2, 0]
    S20 = S[2, 0]
    P20 = P[2, 0]
    Pl20 = Pl[2, 0]

    Vr = (3.0 * qn * P20) * C20
    Vt = (qn * Pl20) * C20
    Vl = 0.0 * (S20)

    Vr = -GMoR * inv_r * Vr
    Vt = GMoR * Vt
    Vl = -GMoR * Vl

    inv_r_st = inv_r / st
    ac0 = st * cl * Vr + ct * cl * Vt * inv_r - sl * Vl * inv_r_st
    ac1 = st * sl * Vr + ct * sl * Vt * inv_r + cl * Vl * inv_r_st
    ac2 = ct * Vr - st * Vt * inv_r

    out[0] = ac0 * cmst - ac1 * smst
    out[1] = ac0 * smst + ac1 * cmst
    out[2] = ac2

#-------------------------------------------------------------------------------
# Gravitational potential of Earth expanded in spherical harmonics (EGM2008).
@njit(cache=True, fastmath=True, nogil=True)
def EGM2008(nmax, mmax, xi, C, S, t, GM, Re, spin, srtime0, P, Pl, sml, cml, out):
    if nmax == 2 and mmax == 0:
        _EGM2008_n2_m0_fast(xi, C, t, GM, Re, spin, srtime0, out)
        return

    # Optimization: Replaced generic 'Rm' with 'Re' directly
    # P, Pl, sml, cml are scratchpads provided by caller
    
    # Coordinate rotation
    mst = sidereal_time(t, spin, srtime0)
    smst = np.sin(mst)
    cmst = np.cos(mst)

    # Inertial frame ---> PCPF
    x0 = xi[0] * cmst + xi[1] * smst
    x1 = -xi[0] * smst + xi[1] * cmst
    x2 = xi[2]
    r2 = x0 * x0 + x1 * x1 + x2 * x2
    r = np.sqrt(r2)

    # Auxiliary variables
    inv_r = 1.0 / r
    ct = x2 * inv_r  # cos(theta)
    
    st2 = 1.0 - ct * ct
    if st2 < 0.0: st2 = 0.0
    st = np.sqrt(st2)
    if st < 1e-16: st = 1e-16
    
    lamb = np.arctan2(x1, x0)
    sl = np.sin(lamb)
    cl = np.cos(lamb)
    
    GMoR = GM * inv_r
    q_base = Re * inv_r

    # --- Initialize Recursion ---
    P[0, 0] = 1.0
    sqrt3 = 1.7320508075688772 # np.sqrt(3.0)
    P[1, 0] = sqrt3 * ct
    P[1, 1] = sqrt3 * st
    Pl[0, 0] = 0.0
    Pl[1, 0] = -P[1, 1]
    Pl[1, 1] = P[1, 0]

    # Pre-calculate Diagonals (P[n,n]) separately (dependency chain)
    # This is O(nmax), cheap
    for n in range(2, nmax + 1):
        dn = float(n)
        # (2n+1)/(2n) = 1 + 0.5/n
        fac = np.sqrt((2.0 * dn + 1.0) / (2.0 * dn))
        P[n, n] = st * fac * P[n - 1, n - 1]
        Pl[n, n] = dn * ct / st * P[n, n]

    # Initialize harmonics trig
    sml[0] = 0.0
    sml[1] = sl
    cml[0] = 1.0
    cml[1] = cl
    for m in range(2, mmax + 1):
        cml[m] = cml[m - 1] * cl - sml[m - 1] * sl
        sml[m] = sml[m - 1] * cl + cml[m - 1] * sl

    # --- FUSED LOOP: Generation + Summation ---
    Vr = 0.0
    Vt = 0.0
    Vl = 0.0

    for m in range(0, mmax + 1):
        mi = max(2, m)
        
        # Accumulators for this order m
        XRC = 0.0
        XRS = 0.0
        XTC = 0.0
        XTS = 0.0
        XLC = 0.0
        XLS = 0.0
        
        qn = q_base ** mi
        
        for n in range(mi, nmax + 1):
            dn = float(n)
            dm = float(m)

            # 1. Compute P[n, m] if not diagonal
            if n > m:
                # Optimized factors
                # (2n-1)(2n+1)/((n-m)(n+m))
                denom_diff = dn - dm
                denom_sum = dn + dm
                q1 = (2.0 * dn - 1.0) * (2.0 * dn + 1.0) / (denom_diff * denom_sum)
                q2 = (2.0 * dn + 1.0) * (denom_sum - 1.0) * (denom_diff - 1.0) / (
                    denom_diff * denom_sum * (2.0 * dn - 3.0)
                )
                anm = np.sqrt(q1)
                bnm = np.sqrt(q2)
                
                # Write to buffer P for next iter (and caller)
                p_curr = anm * ct * P[n - 1, m] - bnm * P[n - 2, m]
                P[n, m] = p_curr
                
                # Derivative Pl
                fnm = (2.0 * dn + 1.0) / anm
                # Note: code uses stored P[n, m] here
                Pl[n, m] = dn * ct / st * p_curr - fnm * P[n - 1, m] / st

            # 2. Accumulate immediately (Fusion)
            # Fetch from buffer (L1 cache hit)
            Pnm = P[n, m]
            Plnm = Pl[n, m]
            Cnm = C[n, m]
            Snm = S[n, m]

            # Common subexpressions
            term_RC = (dn + 1.0) * qn * Pnm
            XRC += term_RC * Cnm
            XRS += term_RC * Snm

            term_TC = qn * Plnm
            XTC += term_TC * Cnm
            XTS += term_TC * Snm

            term_LC = qn * Pnm
            XLC += term_LC * Cnm
            XLS += term_LC * Snm

            qn *= q_base

        # Summation for specific m
        Vr += cml[m] * XRC + sml[m] * XRS
        Vt += cml[m] * XTC + sml[m] * XTS
        Vl += dm * (sml[m] * XLC - cml[m] * XLS)

    # Final scaling
    inv_r_st = inv_r / st
    Vr = -GMoR * inv_r * Vr
    Vt = GMoR * Vt
    Vl = -GMoR * Vl

    # Spherical -> Cartesian (Earth-fixed)
    ac0 = st * cl * Vr + ct * cl * Vt * inv_r - sl * Vl * inv_r_st
    ac1 = st * sl * Vr + ct * sl * Vt * inv_r + cl * Vl * inv_r_st
    ac2 = ct * Vr - st * Vt * inv_r

    # Earth-fixed -> inertial
    out[0] = ac0 * cmst - ac1 * smst
    out[1] = ac0 * smst + ac1 * cmst
    out[2] = ac2

#-------------------------------------------------------------------------------
# Function sidereal_time
@njit(cache=True, fastmath=True, nogil=True)
def sidereal_time(t, spin, srtime0):
    pi2 = 2.0 * np.pi
    srtime = (srtime0 + spin * t) % pi2
    # Ensure positive remainder
    if srtime < 0.0:
        srtime += pi2
    return srtime

#-------------------------------------------------------------------------------
# Perturbation due to the oblateness only
@njit(cache=True, fastmath=True, nogil=True)
def J2acc(GM, J2, Re, xi, out):
    x = xi[0]
    y = xi[1]
    z = xi[2]
    r2 = x * x + y * y + z * z
    # Use r2 directly for inv_r5
    inv_r2 = 1.0 / r2
    # r = sqrt(r2), inv_r5 = (1/r^2)^2 * (1/r)
    inv_r = 1.0 / np.sqrt(r2)
    inv_r5 = inv_r2 * inv_r2 * inv_r

    fJ2 = 1.5 * J2 * GM * Re * Re * inv_r5
    zz_r2 = z * z * inv_r2

    common = 5.0 * zz_r2 - 1.0
    out[0] = fJ2 * x * common
    out[1] = fJ2 * y * common
    out[2] = fJ2 * z * (5.0 * zz_r2 - 3.0)

#-------------------------------------------------------------------------------
# Perturbation due to the atmospheric drag
# Module-level constants
earth_Re = 6378.1366
factor_rho = 1e9 # kg/m^3 to kg/km^3
factor_AtoM = 1e-6 # m^2/kg to km^2/kg

@njit(cache=True, fastmath=True, nogil=True)
def atm_drag(Xsat, Cd, AtoM, spin, out):
    # Unpack scalars
    xs0, xs1, xs2 = Xsat[0], Xsat[1], Xsat[2]
    vs0, vs1, vs2 = Xsat[3], Xsat[4], Xsat[5]
    
    rsat2 = xs0 * xs0 + xs1 * xs1 + xs2 * xs2
    rsat = np.sqrt(rsat2)
    altitude = rsat - earth_Re

    # External call (assumed compatible with njit/fastmath)
    rho = rho_atm(altitude) * factor_rho

    # omega x r = [-spin*y, spin*x, 0]
    v_atm0 = -spin * xs1
    v_atm1 = spin * xs0
    # v_atm2 = 0.0

    vrel0 = vs0 - v_atm0
    vrel1 = vs1 - v_atm1
    vrel2 = vs2 # - 0.0
    
    vrel2_mag = vrel0 * vrel0 + vrel1 * vrel1 + vrel2 * vrel2
    vrel = np.sqrt(vrel2_mag)

    # -0.5 * rho * Cd * AtoM * vrel
    # Combined constants
    coeff = -0.5 * rho * Cd * (AtoM * factor_AtoM) * vrel

    out[0] = coeff * vrel0
    out[1] = coeff * vrel1
    out[2] = coeff * vrel2