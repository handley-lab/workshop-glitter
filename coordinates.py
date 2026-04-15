"""JAX coordinate transforms: TEME -> AltAz.

Ported from astropy's coordinate transform pipeline:
  TEME -> ITRS (intermediate_rotation_transforms.py, using erfa.gmst82)
  ITRS -> AltAz (itrs_observed_transforms.py)
  EarthLocation.from_geodetic (earth.py, using erfa.gd2gc)

All functions operate on JAX arrays and are jit/vmap/grad compatible.
No refraction correction.

Time arguments are UT1 Julian Date (two-part for numerical precision),
matching ERFA convention. Use utc_to_ut1_jd() to convert from UTC + DUT1.
"""

import jax.numpy as jnp


# WGS-84 ellipsoid constants (ERFA, n=1)
_a = 6378.137  # equatorial radius, km
_f = 1 / 298.257223563  # flattening

# ERFA constants (from erfam.h)
_DJ00 = 2451545.0  # J2000.0 Julian Date
_DJC = 36525.0  # Julian centuries per day
_DAYSEC = 86400.0  # seconds per day
_DS2R = 7.272205216643039903848712e-5  # seconds of time to radians


def utc_to_ut1_jd(utc_jd1, utc_jd2, dut1_sec):
    """Convert UTC Julian Date to UT1 Julian Date.

    UT1 = UTC + DUT1 (where DUT1 is from IERS bulletins, typically |DUT1| < 0.9s).
    """
    return utc_jd1, utc_jd2 + dut1_sec / _DAYSEC


def gmst82(ut1_jd1, ut1_jd2):
    """Greenwich Mean Sidereal Time (IAU 1982 model).

    Literal port of ERFA eraGmst82 (gmst82.c).
    Takes UT1 Julian Date as two-part (ut1_jd1 + ut1_jd2).
    Returns GMST in radians, normalised to [0, 2pi).
    """
    # IAU 1982 GMST-UT1 coefficients (seconds of time)
    # A is adjusted by 12 hours for Julian Date noon phasing
    A = 24110.54841 - _DAYSEC / 2.0
    B = 8640184.812866
    C = 0.093104
    D = -6.2e-6

    # ERFA sorts so d1 <= d2 for numerical precision
    d1 = jnp.where(ut1_jd1 < ut1_jd2, ut1_jd1, ut1_jd2)
    d2 = jnp.where(ut1_jd1 < ut1_jd2, ut1_jd2, ut1_jd1)

    # Julian centuries since J2000.0
    t = (d1 + (d2 - _DJ00)) / _DJC

    # Fractional part of JD(UT1), in seconds
    f = _DAYSEC * (jnp.fmod(d1, 1.0) + jnp.fmod(d2, 1.0))

    # GMST at this UT1, normalised to [0, 2pi)
    return jnp.mod(_DS2R * ((A + (B + (C + D * t) * t) * t) + f), 2 * jnp.pi)


def _rotation_matrix(axis, angle):
    """Rotation matrix about x/y/z axis (scalar angle).

    Port of astropy rotation_matrix(angle, axis) from matrix_utilities.py.
    Counterclockwise looking down the +axis direction.
    For batched use, vmap the calling function.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    i = axis  # 0=x, 1=y, 2=z
    a1 = (i + 1) % 3
    a2 = (i + 2) % 3
    R = jnp.zeros((3, 3))
    R = R.at[i, i].set(1.0)
    R = R.at[a1, a1].set(c)
    R = R.at[a1, a2].set(s)
    R = R.at[a2, a1].set(-s)
    R = R.at[a2, a2].set(c)
    return R


def rotation_x(angle):
    """Rotation matrix about x-axis (radians)."""
    return _rotation_matrix(0, angle)


def rotation_y(angle):
    """Rotation matrix about y-axis (radians)."""
    return _rotation_matrix(1, angle)


def rotation_z(angle):
    """Rotation matrix about z-axis (radians)."""
    return _rotation_matrix(2, angle)


def pom00(xp, yp, sp=0.0):
    """Polar motion matrix (IAU 2000).

    Port of erfa.pom00 (pom00.c): Rz(sp) @ Ry(-xp) @ Rx(-yp).
    xp, yp in radians. sp (TIO locator) set to 0 for TEME consistency
    with Vallado (2006).
    """
    return rotation_z(sp) @ rotation_y(-xp) @ rotation_x(-yp)


def c2tcio(rc2i, era, rpom):
    """Celestial-to-terrestrial matrix from CIO components.

    Port of erfa.c2tcio (c2tcio.c): rpom @ Rz(era) @ rc2i.
    For TEME, rc2i is the identity matrix.
    """
    return rpom @ rotation_z(era) @ rc2i


def teme_to_itrs(r_teme, ut1_jd1, ut1_jd2, xp=0.0, yp=0.0):
    """TEME -> ITRS rotation.

    Port of astropy teme_to_itrs_mat.

    Parameters
    ----------
    r_teme : array (3,)
        Position in TEME frame (km).
    ut1_jd1, ut1_jd2 : float
        UT1 Julian Date (two-part). Use utc_to_ut1_jd() to convert from UTC.
    xp, yp : float
        Polar motion parameters (radians, from IERS tables). Default 0.
    """
    gst = gmst82(ut1_jd1, ut1_jd2)
    pmmat = pom00(xp, yp, 0.0)
    return c2tcio(jnp.eye(3), gst, pmmat) @ r_teme


def geodetic_to_ecef(lon_rad, lat_rad, height_km):
    """Geodetic -> ECEF (geocentric Cartesian).

    Port of erfa.gd2gce (gd2gce.c) with WGS-84 constants.
    """
    e2 = (2.0 - _f) * _f
    sp = jnp.sin(lat_rad)
    cp = jnp.cos(lat_rad)
    ac = _a / jnp.sqrt(1.0 - e2 * sp * sp)
    acp = (ac + height_km) * cp
    x = acp * jnp.cos(lon_rad)
    y = acp * jnp.sin(lon_rad)
    z = (ac * (1.0 - e2) + height_km) * sp
    return jnp.array([x, y, z])


def itrs_to_altaz_mat(lon_rad, lat_rad):
    """ITRS -> AltAz rotation matrix.

    Port of astropy itrs_to_altaz_mat from itrs_observed_transforms.py.
    AltAz frame is left-handed: minus_x @ Ry(90-lat) @ Rz(lon).
    """
    minus_x = jnp.array([[-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    return minus_x @ rotation_y(jnp.pi / 2 - lat_rad) @ rotation_z(lon_rad)


def cartesian_to_altaz(xyz):
    """AltAz Cartesian -> (azimuth, altitude) in radians.

    After itrs_to_altaz_mat, the components are:
      x = North, y = East, z = Up.
    Azimuth is measured North through East: az = atan2(East, North) = atan2(y, x).
    Normalised to [0, 2pi) to match astropy convention.
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    hyp = jnp.sqrt(x * x + y * y)
    alt = jnp.arctan2(z, hyp)
    az = jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi)
    return az, alt


def epoch_to_jd(epochyr, epochdays):
    """TLE epoch (year + fractional day) -> Julian Date.

    Matches the convention used by jaxsgp4.
    """
    yr = jnp.floor(epochyr)
    jd_jan1 = (367.0 * yr
               - jnp.floor(7.0 * (yr + jnp.floor(10.0 / 12.0)) / 4.0)
               + jnp.floor(275.0 / 9.0)
               + 1721013.5)
    return jd_jan1 + epochdays


def observe(r_teme, ut1_jd1, ut1_jd2, station_lon_rad, station_lat_rad,
            station_height_km=0.0, xp=0.0, yp=0.0):
    """Full pipeline: TEME position -> (azimuth, altitude) from ground station.

    Parameters
    ----------
    r_teme : array (3,)
        Satellite TEME position (km).
    ut1_jd1, ut1_jd2 : float
        UT1 Julian Date (two-part). Use utc_to_ut1_jd() to convert from UTC.
    station_lon_rad, station_lat_rad : float
        Station geodetic coordinates (radians).
    station_height_km : float
        Station height above ellipsoid (km).
    xp, yp : float
        Polar motion (radians, from IERS tables). Default 0.

    Returns
    -------
    az, alt : float
        Azimuth and altitude (radians).
    """
    r_itrs = teme_to_itrs(r_teme, ut1_jd1, ut1_jd2, xp, yp)
    r_station = geodetic_to_ecef(station_lon_rad, station_lat_rad, station_height_km)
    dr = r_itrs - r_station
    xyz_altaz = itrs_to_altaz_mat(station_lon_rad, station_lat_rad) @ dr
    return cartesian_to_altaz(xyz_altaz)
