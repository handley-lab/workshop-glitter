"""JAX ports of ERFA astrometry routines for the apparent-place pipeline.

Ported from ERFA C source (liberfa/erfa):
  ab.c    — stellar aberration
  ld.c    — light deflection by a solar-system body
  ldsun.c — light deflection by the Sun
  atciqz.c — ICRS -> GCRS direction (zero proper motion/parallax)
  aticq.c  — GCRS -> ICRS direction (inverse, iterative)

These operate on unit direction vectors. For finite-distance objects,
the calling code decomposes into (direction, distance), transforms the
direction, and reconstructs with the same distance.
"""

import jax
import jax.numpy as jnp

# ERFA constants (from erfam.h)
_SRS = 1.97412574336e-8   # Schwarzschild radius of the Sun (au): 2*GM/(c^2)
_GMAX = lambda a, b: jnp.maximum(a, b)


def ab(pnat, v, s, bm1):
    """Stellar aberration: natural direction -> proper direction.

    Port of erfa.ab (ab.c).

    Parameters
    ----------
    pnat : array (3,)
        Natural direction to source (unit vector).
    v : array (3,)
        Observer barycentric velocity in units of c.
    s : float
        Distance from Sun to observer (au).
    bm1 : float
        sqrt(1 - |v|^2), reciprocal of Lorentz factor.

    Returns
    -------
    array (3,)
        Proper direction to source (unit vector).
    """
    pdv = jnp.dot(pnat, v)
    w1 = 1.0 + pdv / (1.0 + bm1)
    w2 = _SRS / s
    p = pnat * bm1 + w1 * v + w2 * (v - pdv * pnat)
    return p / jnp.linalg.norm(p)


def ld(bm, p, q, e, em, dlim):
    """Light deflection by a solar-system body.

    Port of erfa.ld (ld.c).

    Parameters
    ----------
    bm : float
        Mass of deflecting body (solar masses).
    p : array (3,)
        Direction from observer to source (unit vector).
    q : array (3,)
        Direction from body to source (unit vector).
    e : array (3,)
        Direction from body to observer (unit vector).
    em : float
        Distance from body to observer (au).
    dlim : float
        Deflection limiter.

    Returns
    -------
    array (3,)
        Deflected direction (approximately unit vector).
    """
    qpe = q + e
    qdqpe = jnp.dot(q, qpe)
    w = bm * _SRS / em / jnp.maximum(qdqpe, dlim)
    eq = jnp.cross(e, q)
    peq = jnp.cross(p, eq)
    return p + w * peq


def ldsun(p, e, em):
    """Light deflection by the Sun.

    Port of erfa.ldsun (ldsun.c). Calls ld with solar mass = 1.

    Parameters
    ----------
    p : array (3,)
        Direction from observer to source (unit vector).
    e : array (3,)
        Direction from Sun to observer (unit vector).
    em : float
        Distance from Sun to observer (au).

    Returns
    -------
    array (3,)
        Deflected direction (approximately unit vector).
    """
    em2 = jnp.maximum(em * em, 1.0)
    dlim = 1e-6 / em2
    return ld(1.0, p, p, e, em, dlim)


def atciqz(rc, dc, v, em, eh, bm1, bpn):
    """ICRS -> GCRS direction (zero proper motion/parallax).

    Port of erfa.atciqz (atciqz.c).
    Applies light deflection by Sun, then aberration, then bias-precession-nutation.

    Parameters
    ----------
    rc, dc : float
        ICRS RA, Dec (radians).
    v : array (3,)
        Observer barycentric velocity / c.
    em : float
        Distance Sun-observer (au).
    eh : array (3,)
        Sun-to-observer unit vector.
    bm1 : float
        sqrt(1 - |v|^2).
    bpn : array (3, 3)
        Bias-precession-nutation matrix.

    Returns
    -------
    ri, di : float
        GCRS RA, Dec (radians).
    """
    # BCRS coordinate direction
    pco = jnp.array([jnp.cos(rc) * jnp.cos(dc),
                      jnp.sin(rc) * jnp.cos(dc),
                      jnp.sin(dc)])
    # Light deflection by the Sun
    pnat = ldsun(pco, eh, em)
    # Aberration
    ppr = ab(pnat, v, em, bm1)
    # Bias-precession-nutation
    pi = bpn @ ppr
    # GCRS RA, Dec
    di = jnp.arctan2(pi[2], jnp.sqrt(pi[0]**2 + pi[1]**2))
    ri = jnp.mod(jnp.arctan2(pi[1], pi[0]), 2 * jnp.pi)
    return ri, di


def aticq(ri, di, v, em, eh, bm1, bpn):
    """GCRS -> ICRS direction (inverse of atciqz, iterative).

    Port of erfa.aticq (aticq.c).
    Iteratively inverts aberration (2 iterations) and light deflection
    (5 iterations).

    Parameters
    ----------
    ri, di : float
        GCRS RA, Dec (radians).
    v : array (3,)
        Observer barycentric velocity / c.
    em : float
        Distance Sun-observer (au).
    eh : array (3,)
        Sun-to-observer unit vector.
    bm1 : float
        sqrt(1 - |v|^2).
    bpn : array (3, 3)
        Bias-precession-nutation matrix.

    Returns
    -------
    rc, dc : float
        ICRS RA, Dec (radians).
    """
    # GCRS to Cartesian
    pi = jnp.array([jnp.cos(ri) * jnp.cos(di),
                     jnp.sin(ri) * jnp.cos(di),
                     jnp.sin(di)])

    # Undo bias-precession-nutation
    ppr = bpn.T @ pi

    # Undo aberration (2 iterations of Newton-Raphson)
    def aberration_step(d, _):
        before = ppr - d
        before = before / jnp.linalg.norm(before)
        after = ab(before, v, em, bm1)
        d = after - before
        return d, None

    d_ab = jnp.zeros(3)
    d_ab, _ = jax.lax.scan(aberration_step, d_ab, None, length=2)
    pnat = ppr - d_ab
    pnat = pnat / jnp.linalg.norm(pnat)

    # Undo light deflection (5 iterations of Newton-Raphson)
    def deflection_step(d, _):
        before = pnat - d
        before = before / jnp.linalg.norm(before)
        after = ldsun(before, eh, em)
        d = after - before
        return d, None

    d_ld = jnp.zeros(3)
    d_ld, _ = jax.lax.scan(deflection_step, d_ld, None, length=5)
    pco = pnat - d_ld
    pco = pco / jnp.linalg.norm(pco)

    # ICRS RA, Dec
    dc = jnp.arctan2(pco[2], jnp.sqrt(pco[0]**2 + pco[1]**2))
    rc = jnp.mod(jnp.arctan2(pco[1], pco[0]), 2 * jnp.pi)
    return rc, dc


def observe_apparent(r_teme, R_teme_to_itrs, R_itrs_to_cirs, R_cirs_to_gcrs,
                     R_gcrs_to_cirs, R_cirs_to_itrs, R_itrs_to_altaz,
                     eb_geo, v_geo, em_geo, eh_geo, bm1_geo, bpn_geo,
                     eb_site, v_site, em_site, eh_site, bm1_site, bpn_site):
    """Full apparent-place pipeline: TEME position -> (az, alt).

    Matches astropy's TEME -> ITRS(geocentric) -> CIRS -> GCRS ->
    ICRS -> GCRS(station) -> CIRS -> ITRS(station) -> AltAz path.

    Parameters
    ----------
    r_teme : array (3,)
        Satellite TEME position (km).
    R_* : array (3, 3)
        Precomputed rotation matrices (from erfa/astropy).
    eb_*, v_*, em_*, eh_*, bm1_*, bpn_* :
        Astrom parameters for geocentric and station observers
        (from erfa.apcs, precomputed).

    Returns
    -------
    az, alt : float
        Azimuth and altitude (radians).
    """
    from coordinates import cartesian_to_altaz

    _DAU = 149597870.7  # km per au

    # 1. TEME -> ITRS (geocentric)
    r_itrs_geo = R_teme_to_itrs @ r_teme

    # 2. ITRS -> CIRS (geocentric)
    r_cirs_geo = R_itrs_to_cirs @ r_itrs_geo

    # 3. CIRS -> GCRS (geocentric), convert to au for observer-change
    r_gcrs_geo = (R_cirs_to_gcrs @ r_cirs_geo) / _DAU  # au

    # 4. GCRS(geocentric) -> ICRS (all in au)
    #    Extract angles directly from the full vector (no explicit normalisation,
    #    matching astropy's SphericalRepresentation conversion)
    dist_geo = jnp.linalg.norm(r_gcrs_geo)
    ra_gcrs = jnp.arctan2(r_gcrs_geo[1], r_gcrs_geo[0])
    dec_gcrs = jnp.arctan2(r_gcrs_geo[2], jnp.sqrt(r_gcrs_geo[0]**2 + r_gcrs_geo[1]**2))

    ra_icrs, dec_icrs = aticq(ra_gcrs, dec_gcrs, v_geo, em_geo, eh_geo, bm1_geo, bpn_geo)

    u_icrs = jnp.array([jnp.cos(ra_icrs) * jnp.cos(dec_icrs),
                         jnp.sin(ra_icrs) * jnp.cos(dec_icrs),
                         jnp.sin(dec_icrs)])
    r_icrs = dist_geo * u_icrs + eb_geo  # au

    # 5. ICRS -> GCRS(station) (all in au)
    #    Subtract station eb first (astropy does this before spherical conversion)
    r_rel = r_icrs - eb_site
    dist_site = jnp.linalg.norm(r_rel)
    ra_rel = jnp.arctan2(r_rel[1], r_rel[0])
    dec_rel = jnp.arctan2(r_rel[2], jnp.sqrt(r_rel[0]**2 + r_rel[1]**2))

    ra_gcrs_site, dec_gcrs_site = atciqz(ra_rel, dec_rel, v_site, em_site, eh_site, bm1_site, bpn_site)

    u_gcrs_site = jnp.array([jnp.cos(ra_gcrs_site) * jnp.cos(dec_gcrs_site),
                              jnp.sin(ra_gcrs_site) * jnp.cos(dec_gcrs_site),
                              jnp.sin(dec_gcrs_site)])
    r_gcrs_site = dist_site * u_gcrs_site  # au

    # 6-8: Rotations are scale-invariant, keep in au through to AltAz
    r_cirs_site = R_gcrs_to_cirs @ r_gcrs_site
    r_itrs_site = R_cirs_to_itrs @ r_cirs_site
    r_altaz = R_itrs_to_altaz @ r_itrs_site
    return cartesian_to_altaz(r_altaz)
