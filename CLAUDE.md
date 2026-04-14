# CLAUDE.md - GLITTER Bayesian Techniques Workshop

## Project Context

This is a 2-hour workshop for the GLITTER (GNSS-r SateLlITe EarTh ObsERvation) MSCA Doctoral Training Network training school. The audience are ~12 PhD students working on GNSS reflectometry, satellite remote sensing, ground-penetrating radar, and signal processing. They are NOT statisticians -- they know "inverse problem" but not necessarily "posterior distribution".

The workshop is delivered by Will Handley (Cambridge/PolyChord) at the University of Luxembourg on 15 April 2026, 11:00-13:00 CEST (10:00-12:00 UK).

## Structure

- **Hour 1**: Talk in three parts (inverse problems, AI-augmented workflows, nested sampling)
- **Hour 2**: Hands-on notebook using jaxsgp4 + BlackJAX NSS

## Key Dependencies

### jaxsgp4
Charlotte Priestley's differentiable SGP4 satellite orbit propagator. Cloned locally in `jaxsgp4/`. Provides the forward model for the inverse problem example.

Key API:
```python
from jaxsgp4 import tle2sat, sgp4
sat = tle2sat(tle_line1, tle_line2)
rv, error_code = sgp4(sat, tsince_minutes)  # returns [x,y,z,vx,vy,vz] in km, km/s
```

### BlackJAX NSS (v0.1.0-beta)
Use the blackjax-nss skill for the current API. Key pattern:

```python
import blackjax

algo = blackjax.nss(
    logprior_fn=logprior_fn,        # single particle (dict of scalars)
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_dims * 5,
)

live = jax.jit(algo.init)(particles)
step_fn = jax.jit(algo.step)

dead = []
while not live.logZ_live - live.logZ < -3:
    rng_key, subkey = jax.random.split(rng_key)
    live, dead_info = step_fn(subkey, live)
    dead.append(dead_info)

dead = blackjax.ns.utils.finalise(live, dead)
```

### anesthetic
For posterior visualisation:
```python
from anesthetic import NestedSamples
samples = NestedSamples(dead.particles, logL=dead.loglikelihood, logL_birth=dead.loglikelihood_birth, labels=labels)
samples.plot_2d(axes)
```

## Development Workflow

1. Develop workshop as `workshop_glitter.py` using py2nb comment syntax
2. Convert to notebook using py2nb
3. The old blackjax workshop at `../workshop-blackjax-nested-sampling/` uses an OUTDATED blackjax API -- do NOT copy its API patterns. Use the blackjax-nss skill for the current API.

### py2nb Comment Syntax

```python
#| # Markdown heading
#| Markdown cell content
#| - bullets, $math$, etc.

# Regular python code

#- # Starts a new code cell

#! pip install package  # Command/install cell
```

## Workshop Script Assembly Guide

The workshop script (`workshop_glitter.py`) should be assembled as follows:

### The Inverse Problem

We frame a satellite orbit determination problem as Bayesian inference:
- **Forward model**: `jaxsgp4.sgp4()` takes orbital elements → predicts satellite position at given times
- **Synthetic data**: Pick a real Starlink TLE, propagate to ~10-20 time points, add Gaussian noise to positions
- **Inference**: Given noisy positions, infer orbital elements back using nested sampling

### Parameter Choice

The `Satellite` NamedTuple has 9 fields: `n0, e0, i0, w0, Omega0, M0, Bstar, epochdays, epochyr`.
For a tractable workshop example, fix most and infer 2-3:
- **Good candidates to infer**: `i0` (inclination), `Omega0` (RAAN), possibly `e0` (eccentricity)
- **Fix**: `n0`, `w0`, `M0`, `Bstar`, `epochdays`, `epochyr` at their true TLE values
- This keeps the problem low-dimensional (fast convergence, clear corner plots) while being physically meaningful

### Script Structure (py2nb format)

```
1. Installation cell (#! pip install ...)
2. Imports and setup
3. Part 1: The Forward Model (~15 min)
   - Parse a real Starlink TLE
   - Propagate orbit, plot trajectory
   - Show JAX differentiability: jax.grad of position w.r.t. orbital element
4. Part 2: Setting Up the Inverse Problem (~10 min)
   - Generate synthetic observations (true TLE + noise)
   - Plot observed vs true positions
   - Define loglikelihood_fn and logprior_fn
5. Part 3: Running Nested Sampling (~20 min)
   - Set up blackjax.nss() with the current API (see skill)
   - Run the sampler with tqdm progress bar
   - Finalise dead points
6. Part 4: Visualisation & Interpretation (~15 min)
   - Convert to anesthetic NestedSamples
   - Corner plot with true values overlaid
   - Report log-evidence, discuss what it means
   - Extension exercise: change priors, add/remove parameters
```

### Key Implementation Details

**Building the Satellite with inferred parameters**:
The `Satellite` is a NamedTuple, so to vary some elements while fixing others:
```python
def make_satellite(params):
    return Satellite(
        n0=true_sat.n0,          # fixed
        e0=true_sat.e0,          # fixed (or inferred)
        i0=params["i0"],         # inferred
        w0=true_sat.w0,          # fixed
        Omega0=params["Omega0"], # inferred
        M0=true_sat.M0,          # fixed
        Bstar=true_sat.Bstar,    # fixed
        epochdays=true_sat.epochdays,
        epochyr=true_sat.epochyr,
    )
```

**Likelihood**: Gaussian on position residuals
```python
def loglikelihood_fn(params):
    sat = make_satellite(params)
    predicted, _ = jax.vmap(sgp4, in_axes=(None, 0))(sat, times)
    positions_pred = predicted[:, :3]
    return -0.5 * jnp.sum(((positions_obs - positions_pred) / sigma_obs) ** 2)
```

**Prior**: Uniform around the true values (wide enough to be non-trivial, narrow enough to converge in a workshop)
```python
def logprior_fn(params):
    lp = jax.scipy.stats.uniform.logpdf(params["i0"], true_i0 - 5.0, 10.0)
    lp += jax.scipy.stats.uniform.logpdf(params["Omega0"], true_Omega0 - 10.0, 20.0)
    return lp
```

**Initial particles**: Draw from the prior
```python
particles = {
    "i0": jax.random.uniform(key1, (num_live,), minval=true_i0 - 5.0, maxval=true_i0 + 5.0),
    "Omega0": jax.random.uniform(key2, (num_live,), minval=true_Omega0 - 10.0, maxval=true_Omega0 + 10.0),
}
```

**Note on `jax.scipy.stats.uniform.logpdf`**: the signature is `logpdf(x, loc, scale)` where `scale` is the width, NOT the upper bound. So `uniform.logpdf(x, a, b-a)` for x in [a, b].

### Tone and Style

- The notebook is for hour 2 (hands-on). Hour 1 (the talk) has already introduced the concepts.
- Keep markdown cells brief -- the students have just heard the theory.
- Use comments in code to explain what's happening, not long prose.
- Include exercises: "Try changing the noise level", "Add e0 as an inferred parameter", "What happens to the evidence?"
- Use a real Starlink TLE so it feels concrete and relevant to GNSS.

### Reference Workshops

- `../workshop-blackjax-nested-sampling/workshop_nested_sampling.py` -- form factor reference (py2nb style), but uses OUTDATED blackjax API
- `../workshop-monte-carlo-methods/workshop.py` -- exercise/answer style with `%load solutions/` pattern

## Anti-patterns

- Do NOT use the old `blackjax.nested_sampling` API or `@nested_sampling` branch
- Do NOT use `blackjax.nss` with flat array parameters -- use pytree (dict) parameters
- Do NOT make the notebook Colab-specific -- it should work in any Jupyter environment
- Do NOT over-explain Bayesian fundamentals in the notebook -- that's covered in the talk
- Do NOT infer all 9 orbital elements -- keep it to 2-3 for tractability in a workshop setting
