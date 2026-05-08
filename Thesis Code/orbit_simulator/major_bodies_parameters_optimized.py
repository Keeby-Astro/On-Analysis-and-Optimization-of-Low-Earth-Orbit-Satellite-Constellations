# This file contains parameters and constants used in the simulation.


class _SingletonSlotsBase:
    __slots__ = ()
    _instance = None

    def __new__(cls):
        # Safe optimization: return a per-class singleton instance while keeping
        # class-based construction API (`constants()`, `earth()`, etc.) unchanged.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class constants(_SingletonSlotsBase):
    au = 149597870.7 # km (Astronomical Almanac 2024)
    G = 6.67428e-20  # km^3/kg/s^2 (Astronomical Almanac 2024)
    SRP0 = 4.56e-6   # N/m^2 Solar radiation pressure at 1 au
    day = 86400      # seconds


# Major bodies physical parameters
class sun(_SingletonSlotsBase):
    # Mass: kg (NASA NSSDC/GSFC 2024)
    mass = 1.9884e30
    # GM: Precomputed (G * mass) to save runtime ops: 1.3271168592e11
    GM = 1.3271168592e11
    # Radius: km (volumetric mean radius - JPL Horizons rev. 2013)
    Re = 695700
    # Spin: rad/s (NASA NSSDC/GSFC 2024)
    spin = 2.86533e-6
    # Obliquity: deg (obliquity of the ecliptic - NASA NSSDC/GSFC 2024)
    eps = 7.25


class earth(_SingletonSlotsBase):
    mass = 5.9722e24 # kg
    # GM: Precomputed: 3.9860157436e5 km^3/s^2
    GM = 3.9860157436e5
    # Radius: km (equatorial radius - Astronomical Almanac 2024)
    Re = 6378.1366
    # Spin: rad/s (rotational speed - Astronomical Almanac 2024)
    spin = 7.292115e-5
    # Obliquity: deg (obliquity of the ecliptic - Astronomical Almanac 2024)
    eps = 23.4392911
    # J2 Harmonic (Astronomical Almanac 2024)
    J2 = 1.0826359e-3


class moon(_SingletonSlotsBase):
    mass = 7.345828157e22 # kg (Astronomical Almanac 2024)
    # GM: Precomputed: 4902.8114309196 km^3/s^2
    GM = 4902.8114309196
    # Radius: km (equatorial radius - Astronomical Almanac 2024)
    Re = 1737.4
    # Spin: rad/s (rotational speed - JPL Horizons rev. 2013)
    spin = 2.6617e-6
    # Obliquity: deg (obliquity of the ecliptic - JPL Horizons rev. 2013)
    eps = 6.67


# Optional module-level singletons for callers that prefer direct instances.
CONSTANTS = constants()
SUN = sun()
EARTH = earth()
MOON = moon()


def _self_test():
    c0 = constants()
    c1 = constants()
    assert c0 is c1
    assert c0.au == 149597870.7
    assert earth().Re == 6378.1366
    assert moon().GM == 4902.8114309196
    print("major_bodies_parameters_optimized self-test passed")


if __name__ == '__main__':
    _self_test()