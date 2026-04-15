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
#! pip install jaxsgp4 anesthetic tqdm matplotlib

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

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt='.', color='k', capthick=0.1, markersize=0, linewidth=0.1)

x_plot = np.linspace(0, 1, 200)
posterior = samples_linear.sample(200)
for _, row in posterior.iterrows():
    ax.plot(x_plot, row['c0'] + row['c1'] * x_plot, 'C0-', alpha=0.02, linewidth=0.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Data space: predictive posterior $P(y|x, D)$');

#| This is sampling in action: each line is one posterior sample propagated
#| through the forward model. The envelope of lines shows our uncertainty
#| about the prediction — proper error bars on the model, not just the
#| parameters.

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
