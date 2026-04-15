#| # Bayesian Techniques for GNSS-R
#|
#| **GLITTER Training School Workshop**
#|
#| University of Luxembourg, 15 April 2026
#|
#| Will Handley (University of Cambridge / PolyChord Ltd)
#|
#| Workshop materials: [github.com/handley-lab/workshop-glitter](https://github.com/handley-lab/workshop-glitter)

#| ## Installation
#|
#| **Important**: This workshop uses the `handley-lab` fork of BlackJAX, not the upstream BlackJAX package.
#| You must install from the correct tag:

#! pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta"
#! pip install jaxsgp4 anesthetic fgivenx tqdm matplotlib

#| ## Part 1: Bayesian Inference via Line Fitting
#|
#| We start with the simplest possible inverse problem: fitting a line to data.
#| This is the same problem you solve with least squares, but we'll see how
#| Bayesian inference gives you much more.

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import blackjax
import tqdm
from anesthetic import NestedSamples

jax.config.update("jax_enable_x64", True)

#| ### 1.1 Generate some noisy data
#|
#| True model: $y = 1 + x^3$ with non-uniform Gaussian noise.

np.random.seed(0)
n_data = 100
x = np.random.uniform(0, 1, n_data)
sigma = np.random.uniform(0.1, 0.5, n_data)
y_true = 1 + x**3
y = np.random.normal(y_true, sigma)

#-
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt='.', color='k', capthick=0.1, markersize=0, linewidth=0.1)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Noisy data');

#| ### 1.2 The $\chi^2$ approach
#|
#| You already know how to fit parameters to this data. For a model $y = f(x; \theta)$,
#| minimise:
#| $$\chi^2(\theta) = \sum_i \frac{|y_i - f(x_i;\theta)|^2}{\sigma_i^2}$$
#|
#| But this only gives you a **point estimate** — the "best fit". No error bars,
#| no way to compare models, and no way to know if the fit is any good.

#| ### 1.3 From $\chi^2$ to likelihood
#|
#| If the errors are Gaussian, then $\chi^2$ minimisation is equivalent to
#| **maximum likelihood**:
#| $$P(D|\theta) \propto e^{-\chi^2(\theta)/2}$$
#|
#| This is your **likelihood function** — the probability of observing the data
#| given specific parameter values. Now we can do proper Bayesian inference.

#| ### 1.4 Why sampling?
#|
#| Why not just evaluate the posterior on a grid? In 2D with 100 grid points
#| per dimension, that's $100^2 = 10{,}000$ evaluations. In 15D (like the
#| gravitational wave problem from the talk), it's $100^{15} = 10^{30}$.
#| Grids are hopeless in high dimensions.
#|
#| **Samples** solve this. If you can generate $N$ representative draws
#| $\theta_i \sim P(\theta|D)$, then:
#|
#| - **Error bars**: just compute the spread of $\theta_i$
#| - **Propagation**: want to know how $f(\theta)$ is distributed?
#|   Just compute $f(\theta_i)$ for each sample. Sampling turns uncertainty
#|   quantification into repeated forward models.
#| - **Marginals for free**: want $P(\theta_1)$ ignoring all other parameters?
#|   Just ignore the other columns.
#| - **The golden rule**: stay in samples until the last moment, because
#|   $f(\langle \theta \rangle) \ne \langle f(\theta) \rangle$.

#| ### 1.5 Set up the Bayesian problem
#|
#| We'll fit polynomials of different orders to the data.
#| For a polynomial $y = a + bx$, we have parameters $\theta = (a, b)$.
#|
#| We need:
#| - **Likelihood**: $P(D|\theta) \propto e^{-\chi^2/2}$
#| - **Prior**: uniform over a reasonable range
#| - **Nested sampling** will give us the **posterior** $P(\theta|D)$ and **evidence** $\mathcal{Z}$

x_jax = jnp.array(x)
y_jax = jnp.array(y)
sigma_jax = jnp.array(sigma)

def run_line_fitting(active_powers, num_live=500, rng_seed=42):
    """Run nested sampling for a polynomial model.

    active_powers: list of ints, e.g. [0, 1] for y = a + bx
    """
    param_names = [f"c{p}" for p in active_powers]
    num_dims = len(param_names)

    prior_min, prior_width = -3.0, 6.0

    def logprior_fn(params):
        lp = 0.0
        for name in param_names:
            lp = lp + jax.scipy.stats.uniform.logpdf(params[name], prior_min, prior_width)
        return lp

    def loglikelihood_fn(params):
        y_model = sum(params[f"c{p}"] * x_jax**p for p in active_powers)
        return jnp.sum(jax.scipy.stats.norm.logpdf(y_jax, y_model, sigma_jax))

    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_live // 10,
        num_inner_steps=num_dims * 5,
    )

    rng_key = jax.random.PRNGKey(rng_seed)
    rng_key, *prior_keys = jax.random.split(rng_key, num_dims + 1)
    particles = {
        name: jax.random.uniform(key, (num_live,), minval=prior_min, maxval=prior_min + prior_width)
        for name, key in zip(param_names, prior_keys)
    }

    live = jax.jit(algo.init)(particles)
    step_fn = jax.jit(algo.step)

    dead = []
    with tqdm.tqdm(desc=f"NS {active_powers}", unit=" dead") as pbar:
        while not live.integrator.logZ_live - live.integrator.logZ < -3:
            rng_key, subkey = jax.random.split(rng_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_live // 10)

    dead = blackjax.ns.utils.finalise(live, dead)

    labels = {name: f"${name}$" for name in param_names}
    samples = NestedSamples(
        dead.particles.position,
        logL=dead.particles.loglikelihood,
        logL_birth=dead.particles.loglikelihood_birth,
        labels=labels,
    )
    return samples

#| ### 1.5 Run nested sampling for a linear model
#|
#| Model: $y = a + bx$ (powers 0 and 1)

samples_linear = run_line_fitting([0, 1])

#| ### 1.6 Parameter space: the posterior
#|
#| The posterior $P(\theta|D)$ lives in **parameter space** — the space of
#| model coefficients. This is what nested sampling directly gives us.

samples_linear.plot_2d(['c0', 'c1'])
plt.suptitle('Parameter space: $y = a + bx$');

#| Each point in this triangle plot is a set of parameters $(a, b)$.
#| The contours show which combinations are consistent with the data.
#| Notice the correlation — if the intercept $a$ is higher, the slope $b$
#| must be lower to still fit the data.

#| ### 1.7 Data space: the predictive posterior
#|
#| But we often care about predictions, not parameters. We can project the
#| posterior into **data space** by running each posterior sample through the
#| forward model. This is the predictive posterior $P(y|x, D)$.
#|
#| `fgivenx` computes this properly: for each $x$, it builds the full
#| distribution $P(y|x) = \int P(y|x,\theta) P(\theta|D) d\theta$
#| and plots it as filled contours showing 1- and 2-sigma regions.

from fgivenx import plot_contours

def linear_model(x_val, theta):
    return theta[0] + theta[1] * x_val

x_plot = np.linspace(0, 1, 200)
theta_samples = samples_linear.sample(500)[['c0', 'c1']].to_numpy()

fig, ax = plt.subplots()
cbar = plot_contours(linear_model, x_plot, theta_samples, ax=ax)
ax.errorbar(x, y, yerr=sigma, fmt='.', color='k', capthick=0.1, markersize=0, linewidth=0.1)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Data space: predictive posterior $P(y|x, D)$')
plt.colorbar(cbar, label='posterior mass');

#| The colours show the posterior probability mass in data space.
#| This is sampling in action: each posterior sample is propagated through the
#| forward model, and the envelope shows our uncertainty about the prediction —
#| proper error bars on the model, not just the parameters.

#| ### 1.8 Model comparison
#|
#| Now the real power: which polynomial order best explains the data?
#| The **evidence** $\mathcal{Z}$ answers this automatically.

models = {
    '$a$': [0],
    '$a+bx$': [0, 1],
    '$a+bx^2$': [0, 2],
    '$a+bx^3$': [0, 3],
    '$a+bx+cx^2$': [0, 1, 2],
    '$a+bx+cx^3$': [0, 1, 3],
    '$a+bx+cx^2+dx^3$': [0, 1, 2, 3],
}

results = {}
for label, powers in models.items():
    print(f"\nRunning {label}...")
    results[label] = run_line_fitting(powers)

#-
labels = list(results.keys())
logZs = np.array([float(results[l].logZ()) for l in labels])
logZs -= logZs.max()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(labels)), np.exp(logZs - np.max(logZs)))
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Betting odds')
ax.set_title('Model comparison: which polynomial fits best?')
fig.tight_layout();

#| The evidence automatically penalises overly complex models (Occam's razor).
#| The model $a + bx^3$ should win — it matches the true data-generating process.

#| ## Part 2: Satellite Orbit Determination with jaxsgp4
#|
#| Now we apply the same Bayesian framework to a real GNSS problem:
#| inferring satellite orbital elements from noisy position observations.
#|
#| The forward model is [jaxsgp4](https://github.com/cmpriestley/jaxsgp4)
#| — a differentiable SGP4 propagator written in pure JAX by Charlotte Priestley.
#| SGP4 is the standard algorithm for propagating Two-Line Element (TLE) sets
#| to predict satellite positions.

from jaxsgp4 import tle2sat, sgp4

#| ### 2.1 The forward model: TLE → satellite positions
#|
#| We start with a real Starlink TLE and propagate it to get the satellite
#| trajectory. This is the **forward model**: orbital elements in, positions out.

tle_line1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
tle_line2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"

true_sat = tle2sat(tle_line1, tle_line2)
print("Satellite orbital elements:")
print(f"  Inclination:  {true_sat.i0:.4f}°")
print(f"  RAAN:         {true_sat.Omega0:.4f}°")
print(f"  Eccentricity: {true_sat.e0:.6f}")
print(f"  Mean motion:  {true_sat.n0:.4f} rev/day")

#| ### 2.2 Data space: the satellite trajectory
#|
#| Propagate the orbit over one orbital period (~95 minutes for LEO)
#| and plot the trajectory. This is **data space**.

times = jnp.linspace(0, 95, 200)  # minutes
sgp4_vmap = jax.vmap(sgp4, in_axes=(None, 0))
rvs_true, _ = sgp4_vmap(true_sat, times)
positions_true = rvs_true[:, :3]  # x, y, z in km

#-
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(131)
ax1.plot(times, positions_true[:, 0])
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('x (km)')

ax2 = fig.add_subplot(132)
ax2.plot(times, positions_true[:, 1])
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('y (km)')

ax3 = fig.add_subplot(133)
ax3.plot(times, positions_true[:, 2])
ax3.set_xlabel('Time (min)')
ax3.set_ylabel('z (km)')

fig.suptitle('Data space: satellite trajectory (TEME frame)')
fig.tight_layout();

#| ### 2.3 Generate synthetic observations
#|
#| In practice, we don't observe the full trajectory — we get noisy position
#| measurements at discrete times (e.g. from radar tracking or GNSS-R reflections).

n_obs = 15
obs_times = jnp.linspace(5, 90, n_obs)  # minutes
rvs_obs, _ = sgp4_vmap(true_sat, obs_times)
positions_obs = rvs_obs[:, :3]

# Add Gaussian noise (10 km — large enough to make the problem interesting)
sigma_obs = 10.0  # km
rng_key = jax.random.PRNGKey(0)
noise = sigma_obs * jax.random.normal(rng_key, positions_obs.shape)
positions_obs = positions_obs + noise

#-
fig = plt.figure(figsize=(10, 4))
labels_xyz = ['x', 'y', 'z']
for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.plot(times, positions_true[:, i], 'C0-', alpha=0.5, label='True orbit')
    ax.errorbar(obs_times, positions_obs[:, i], yerr=sigma_obs,
                fmt='k.', capsize=3, label='Observations')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(f'{labels_xyz[i]} (km)')
    if i == 0:
        ax.legend()

fig.suptitle('Data space: noisy satellite observations')
fig.tight_layout();

#| The inverse problem: given these noisy observations, what are the
#| orbital elements? This is exactly the same structure as line fitting —
#| just with a more interesting forward model.

#| ### 2.4 Set up the Bayesian problem
#|
#| We'll infer two orbital elements — inclination $i_0$ and RAAN $\Omega_0$ —
#| keeping the others fixed at their true values. This keeps the problem
#| 2D so we get clear corner plots.

from jaxsgp4 import Satellite

true_i0 = float(true_sat.i0)
true_Omega0 = float(true_sat.Omega0)

def make_satellite(params):
    """Build a Satellite with inferred i0 and Omega0, everything else fixed."""
    return Satellite(
        n0=true_sat.n0,
        e0=true_sat.e0,
        i0=params["i0"],
        w0=true_sat.w0,
        Omega0=params["Omega0"],
        M0=true_sat.M0,
        Bstar=true_sat.Bstar,
        epochdays=true_sat.epochdays,
        epochyr=true_sat.epochyr,
    )

# Prior: uniform, ±5° around the true values
prior_half_width = 5.0

def logprior_fn(params):
    lp = jax.scipy.stats.uniform.logpdf(
        params["i0"], true_i0 - prior_half_width, 2 * prior_half_width)
    lp += jax.scipy.stats.uniform.logpdf(
        params["Omega0"], true_Omega0 - prior_half_width, 2 * prior_half_width)
    return lp

def loglikelihood_fn(params):
    sat = make_satellite(params)
    predicted, _ = jax.vmap(sgp4, in_axes=(None, 0))(sat, obs_times)
    positions_pred = predicted[:, :3]
    return -0.5 * jnp.sum(((positions_obs - positions_pred) / sigma_obs) ** 2)

#| ### 2.5 Run nested sampling

num_live = 500
num_dims = 2
num_delete = num_live // 10
num_inner_steps = num_dims * 5

algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)

rng_key = jax.random.PRNGKey(1)
rng_key, key1, key2 = jax.random.split(rng_key, 3)

particles = {
    "i0": jax.random.uniform(key1, (num_live,),
        minval=true_i0 - prior_half_width, maxval=true_i0 + prior_half_width),
    "Omega0": jax.random.uniform(key2, (num_live,),
        minval=true_Omega0 - prior_half_width, maxval=true_Omega0 + prior_half_width),
}

live = jax.jit(algo.init)(particles)
step_fn = jax.jit(algo.step)

dead = []
with tqdm.tqdm(desc="NS orbit", unit=" dead") as pbar:
    while not live.integrator.logZ_live - live.integrator.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key)
        live, dead_info = step_fn(subkey, live)
        dead.append(dead_info)
        pbar.update(num_delete)

dead = blackjax.ns.utils.finalise(live, dead)

orbit_samples = NestedSamples(
    dead.particles.position,
    logL=dead.particles.loglikelihood,
    logL_birth=dead.particles.loglikelihood_birth,
    labels={"i0": r"$i_0$ (°)", "Omega0": r"$\Omega_0$ (°)"},
)

print(f"\nLog evidence: {orbit_samples.logZ():.1f}")
print(f"True:      i0 = {true_i0:.4f}°,  Omega0 = {true_Omega0:.4f}°")
print(f"Posterior:  i0 = {orbit_samples['i0'].mean():.4f} ± {orbit_samples['i0'].std():.4f}°")
print(f"           Omega0 = {orbit_samples['Omega0'].mean():.4f} ± {orbit_samples['Omega0'].std():.4f}°")

#| ### 2.6 Parameter space: posterior over orbital elements

axes = orbit_samples.plot_2d(['i0', 'Omega0'])
axes.axlines({"i0": true_i0, "Omega0": true_Omega0}, color='red', linestyle='--')
plt.suptitle('Parameter space: orbital elements');

#| The red dashed lines mark the true values. The posterior recovers them
#| with uncertainties that reflect the noise level and the geometry of the problem.

#| ### 2.7 Data space: predictive posterior
#|
#| Project the posterior back into data space using fgivenx.
#| For each posterior sample, propagate the orbit and plot the predicted
#| trajectory.

theta_orbit = orbit_samples.sample(200)[['i0', 'Omega0']].to_numpy()
t_plot = np.linspace(0, 95, 100)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i_coord in range(3):
    def orbit_coord(t, theta, coord=i_coord):
        sat = make_satellite({"i0": theta[0], "Omega0": theta[1]})
        rv, _ = sgp4(sat, t)
        return rv[coord]

    cbar = plot_contours(orbit_coord, t_plot, theta_orbit, ax=axes[i_coord])
    axes[i_coord].errorbar(obs_times, positions_obs[:, i_coord], yerr=sigma_obs,
                           fmt='k.', capsize=3)
    axes[i_coord].set_xlabel('Time (min)')
    axes[i_coord].set_ylabel(f'{labels_xyz[i_coord]} (km)')

fig.suptitle('Data space: predictive posterior over satellite trajectory')
fig.tight_layout();

#| The contours show where the satellite could be, given our noisy observations.
#| This is exactly what you'd want for GNSS-R: not just a best-fit orbit,
#| but the full uncertainty on the satellite position at any future time.
