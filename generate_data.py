"""Generate synthetic satellite observation data for the GLITTER workshop.

Picks a real Starlink TLE, propagates to multiple time points,
adds Gaussian noise to positions, and saves the observations.
The participants receive only the noisy observations and the TLE --
they must infer orbital elements back.

This script is NOT distributed to participants.
"""

import sys
sys.path.insert(0, "jaxsgp4")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxsgp4 import tle2sat, sgp4, Satellite

# A real Starlink TLE (Starlink-1007, from early 2026)
tle_line1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
tle_line2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"

true_sat = tle2sat(tle_line1, tle_line2)
print("True satellite parameters:")
for name, val in zip(Satellite._fields, true_sat):
    print(f"  {name:12s} = {float(val)}")

# Generate observation times: 20 points over 2 orbits (~190 minutes)
orbital_period = 1440.0 / float(true_sat.n0)  # minutes
print(f"\nOrbital period: {orbital_period:.1f} minutes")

np.random.seed(42)
n_obs = 20
times = np.sort(np.random.uniform(0, 2 * orbital_period, n_obs))

# Propagate to get true positions
sgp4_vmap = jax.vmap(sgp4, in_axes=(None, 0))
times_jax = jnp.array(times)
rvs_true, errors = sgp4_vmap(true_sat, times_jax)
positions_true = np.array(rvs_true[:, :3])  # km, TEME frame

# Add Gaussian noise to positions
sigma_obs = 10.0  # km (realistic for noisy GNSS-R range measurements)
noise = np.random.normal(0, sigma_obs, positions_true.shape)
positions_obs = positions_true + noise

# Save observation data (what participants receive)
np.savez(
    "observations.npz",
    times=times,
    positions_obs=positions_obs,
    sigma_obs=sigma_obs,
    tle_line1=tle_line1,
    tle_line2=tle_line2,
)

print(f"\nGenerated {n_obs} observations over {times[-1]:.1f} minutes")
print(f"Position noise: {sigma_obs} km (1-sigma per component)")
print(f"Saved to observations.npz")

# Also save the ground truth (NOT distributed)
np.savez(
    "ground_truth.npz",
    times=times,
    positions_true=positions_true,
    positions_obs=positions_obs,
    sigma_obs=sigma_obs,
    tle_line1=tle_line1,
    tle_line2=tle_line2,
    true_params={name: float(val) for name, val in zip(Satellite._fields, true_sat)},
)
print("Ground truth saved to ground_truth.npz (DO NOT distribute)")
