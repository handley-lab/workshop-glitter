# HANDOVER: GLITTER Workshop Development

## What this project is

A 2-hour workshop for the GLITTER MSCA Doctoral Training Network training school on **Bayesian Techniques for GNSS-R**. Delivered at University of Luxembourg, 15 April 2026, 11:00-13:00 CEST.

- **Hour 1**: Talk (beamer slides)
- **Hour 2**: Hands-on Jupyter notebook

The audience are ~12 PhD students in GNSS reflectometry, satellite remote sensing, and signal processing. They know "inverse problem" and "forward model" but not Bayesian statistics.

## Repository structure

```
workshop-glitter/
├── README.md                  # Workshop overview with talk slides link
├── CLAUDE.md                  # Development guide and assembly instructions
├── HANDOVER.md                # This file
├── .gitignore                 # Ignores context/, jaxsgp4/, slides/
├── workshop_glitter.py        # Workshop notebook source (py2nb format)
├── coordinates.py             # JAX coordinate transforms (TEME -> AltAz)
├── line_fitting.py            # Original PolyChord line fitting script (legacy)
├── line_fitting/              # Original line fitting module (legacy)
├── jaxsgp4/                   # Cloned from cmpriestley/jaxsgp4 (not tracked)
├── slides/                    # Clone of williamjameshandley/talks on branch glitter_2026
└── context/                   # Reference materials (not tracked)
    ├── talks/                 # Clone of williamjameshandley/talks
    ├── all_talks/             # All 81 talk .tex files extracted
    ├── talk_sources/          # Key talk .tex files for slide assembly
    ├── astropy/               # Clone of astropy (for coordinate transform reference)
    ├── erfa/                  # Clone of liberfa/erfa (C source for coordinate transforms)
    ├── slide_recommendations.md   # Gemini analysis of 6 key talks
    └── deep_analysis.md       # Gemini analysis of all 81 talks
```

## Current state

### Slides (complete draft)
- **Location**: `slides/` (branch `glitter_2026` of `williamjameshandley/talks`)
- **Pushed**: yes, to `origin/glitter_2026`
- **Structure**:
  - Part 1: Inverse problems & Bayesian inference
    - Full line-fitting pedagogy from the IoA Bayesian data analysis lectures
    - Chi-squared → likelihood → parameter estimation → predictive posterior → model comparison
    - Himmelblau animations (optimization, MCMC, nested sampling)
    - LIGO example (masses degeneracy, sky localisation, full corner plot)
    - All figures regenerated using JAX + BlackJAX NSS (`slides/figures/line_fitting_jax.py`)
  - Part 2: AI-augmented scientific workflows
    - Three ways AI meets science, 80/20 rule, context engineering
  - Part 3: Nested sampling & BlackJAX
    - JAX two capabilities, BlackJAX, jaxsgp4 (Charlotte Priestley's SGP4)
    - Workshop handoff
- **Line-fitting figures**: regenerated with `slides/figures/line_fitting_jax.py` using BlackJAX NSS (runs all 31 polynomial models). Uses the **current** blackjax API (`live.integrator.logZ_live`, `dead.particles.position`, etc.)

### Workshop notebook (Part 1 complete, Part 2 pending)
- **Location**: `workshop_glitter.py` (py2nb format)
- **Convert with**: `py2nb workshop_glitter.py`
- **Part 1 covers**:
  - Installation (blackjax v0.1.0-beta from handley-lab fork, jaxsgp4, anesthetic, fgivenx)
  - Generate noisy polynomial data (y = 1 + x³)
  - Chi-squared → likelihood motivation
  - "Why sampling?" interlude (grids fail, samples as error bars, golden rule)
  - `run_line_fitting()` function using BlackJAX NSS
  - **Parameter space**: corner plot (`samples.plot_2d()`)
  - **Data space**: predictive posterior via `fgivenx.plot_contours()`
  - **Model comparison**: evidence bar chart across 7 polynomial models
- **Part 2** (satellite orbit determination with jaxsgp4): to be developed collaboratively with the user. See CLAUDE.md for the assembly guide and design decisions.

### Coordinate transforms (`coordinates.py`)
- **Purpose**: JAX-differentiable TEME → AltAz pipeline for use as the forward model in the workshop's satellite tracking inverse problem
- **Ported from**: astropy + ERFA C source, reviewed by OpenAI (5 iterations approved for mathematical correctness)
- **API**:
  ```python
  from coordinates import observe, utc_to_ut1_jd, epoch_to_jd

  # All functions take UT1 Julian Date (two-part) for numerical precision
  ut1_jd1, ut1_jd2 = utc_to_ut1_jd(utc_jd, 0.0, dut1_sec)
  az, alt = observe(r_teme, ut1_jd1, ut1_jd2, lon_rad, lat_rad, height_km, xp, yp)
  ```
- **External parameters needed** (from IERS bulletins):
  - `dut1_sec`: UT1-UTC in seconds
  - `xp, yp`: polar motion in radians

## Known issue: coordinate transform precision

**Status**: The TEME → ITRS step matches astropy to floating-point precision (verified: 0.000 m with identical JD decomposition). But the full TEME → AltAz pipeline has a **3-9 arcsecond residual** vs astropy.

**Root cause**: Astropy's `ITRS(geocentric) → AltAz(location=station)` path does NOT just subtract the station vector and rotate. It routes through CIRS and GCRS, applying an observer-dependent aberration correction:

```
ITRS(geocentric) → CIRS(geocentric)      [transpose of cirs_to_itrs_mat]
CIRS(geocentric) → GCRS(geocentric)      [transpose of c2i06a]
GCRS(geocentric) → GCRS(observer=station) [via ICRS, using erfa.apcs/aticq/atciqz]
GCRS(station)    → CIRS(station)          [c2i06a]
CIRS(station)    → ITRS(station)          [cirs_to_itrs_mat]
ITRS(station)    → AltAz                  [itrs_to_altaz_mat]
```

The GCRS→GCRS step (via ICRS) applies both a **translation** (barycentric observer offset) and an **aberration remapping** of the direction. For a satellite at ~7000km, this aberration is ~20 arcsec (v_station/c * angular geometry), which matches our residual.

**What's needed to fix it**: JAX ports of the ERFA astrometry routines:
- `erfa.apcs` — compute astrometry parameters for a given observer
- `erfa.aticq` — GCRS → ICRS (inverse aberration + light deflection)
- `erfa.atciqz` — ICRS → GCRS (forward aberration + light deflection)
- Earth barycentric position/velocity (from ephemeris, equivalent to astropy's `prepare_earth_position_vel`)
- `erfa.c2i06a` — celestial-to-intermediate matrix (precession-nutation)
- `erfa.era00` — Earth Rotation Angle
- `erfa.sp00` — TIO locator s'

The ERFA C source for all of these is in `context/erfa/src/`. The astropy Python wrappers showing how they're composed are in `context/astropy/astropy/coordinates/builtin_frames/`.

**Alternative approaches**:
1. **Port the full ERFA astrometry stack** (~500 lines of C → JAX). Correct but heavy.
2. **Direct velocity aberration formula**: for a finite-distance satellite, compute the topocentric direction change due to observer velocity using the relativistic aberration formula. Simpler than full ERFA but should give the same result.
3. **Accept geometric version**: document it as geometric topocentric (not apparent), note the ~10 arcsec discrepancy. May be acceptable for a workshop.

**OpenAI review history**: 8 iterations on branch `coordinates-review`. The mathematical correctness of GMST, geodetic→ECEF, rotation matrices, polar motion, TEME→ITRS, and AltAz extraction are all confirmed. Only the missing CIRS/GCRS observer-change step remains.

## BlackJAX API notes

The installed version is `v0.1.0-beta dev` from `handley-lab/blackjax`. The API has changed from what the `blackjax-nss` skill documents:

| Skill says | Actual API |
|---|---|
| `live.logZ_live` | `live.integrator.logZ_live` |
| `live.logZ` | `live.integrator.logZ` |
| `dead.particles` (dict) | `dead.particles.position` (dict) |
| `dead.loglikelihood` | `dead.particles.loglikelihood` |
| `dead.loglikelihood_birth` | `dead.particles.loglikelihood_birth` |

The convergence criterion is:
```python
while not live.integrator.logZ_live - live.integrator.logZ < -3:
```

Converting to anesthetic:
```python
samples = NestedSamples(
    dead.particles.position,
    logL=dead.particles.loglikelihood,
    logL_birth=dead.particles.loglikelihood_birth,
    labels=labels,
)
```

## Key dependencies

| Package | Source | Notes |
|---|---|---|
| blackjax | `git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta` | Must use handley-lab fork, not upstream |
| jaxsgp4 | `pip install jaxsgp4` or editable from `jaxsgp4/` | Charlotte Priestley's differentiable SGP4 |
| anesthetic | PyPI | Nested sampling visualisation |
| fgivenx | PyPI | Predictive posterior contour plots |
| jax | PyPI | Must enable x64: `jax.config.update("jax_enable_x64", True)` |

Note: if running from the `workshop-glitter/` directory, the `jaxsgp4/` subdirectory can shadow the installed package (namespace package collision). Run from elsewhere or use `PYTHONPATH` to include `jaxsgp4/` subdirectory explicitly.

## py2nb syntax

```python
#| Markdown cell content (after the pipe)
#- Starts a new code cell
#! pip install package   # Becomes !pip install in notebook
# Regular code is a code cell
```

Both `#|` and `# |` work. This project uses `#|`.

## User preferences (from memory)

- **Build collaboratively**: Don't draft large sections autonomously. Propose structure, get approval per section, write one section at a time.
- **Times**: The talk is 11:00-13:00 CEST (10:00-12:00 UK). Never get times/dates wrong.
- **Parameter space vs data space**: Always make explicit which space a plot is in.
- **Use skills**: The `blackjax-nss` skill has the current API pattern (though note the API differences above). Use `iterative-review` skill for OpenAI review.
- **Scripts not inline**: Write scripts to files rather than inline bash commands, to minimise approval overhead.
- **Copy from source**: When porting code (e.g. astropy), read the actual source rather than writing from memory/textbooks.

## Next steps

1. **Fix coordinate transform precision** — port the missing ERFA astrometry routines or implement velocity aberration directly
2. **Generate synthetic observation data** — a separate script that creates the "real data" file participants will load
3. **Build Part 2 of the notebook** — satellite orbit determination using jaxsgp4 + coordinates + BlackJAX NSS (collaboratively with user)
4. **Convert and test** — `py2nb workshop_glitter.py` and run through
5. **Push everything** — workshop repo and slides
