# GLITTER Training School: Bayesian Techniques

**Event**: 4th GLITTER Training School, University of Luxembourg
**Date**: Wednesday 15 April 2026, 11:00-13:00 CEST (10:00-12:00 UK)
**Speaker**: Will Handley (University of Cambridge / PolyChord Ltd)
**Audience**: ~12 GLITTER Doctoral Candidates (GNSS-R, remote sensing, signal processing backgrounds)

## Structure

### Hour 1: Talk (slides)

#### Part 1: Inverse Problems & Bayesian Inference (~20 min)
- Open with "inverse problem" -- language the audience already uses
- Forward model: parameters -> observables (they know this)
- Inverse problem: observables -> parameters (what they want)
- Bayes' theorem as the principled solution:
  - Prior: what you knew before
  - Likelihood: how well the model fits the data
  - Posterior: what you know after
  - Evidence: how good is the model itself
- Why not just chi-squared / least squares? What Bayesian gives you that frequentist doesn't

#### Part 2: AI-Augmented Scientific Workflows (~20 min)
- How modern scientific work looks with AI tools
- The interface vs the thing being interfaced
- Ambient AI: scientists empowered to do what they do, everything else handled by agents
- Live demonstration context: "I built this workshop with AI"
- Normalise the AI usage so that when it appears in the hands-on session it's not surprising

#### Part 3: Nested Sampling & BlackJAX (~20 min)
- What nested sampling gives you: evidence AND posteriors simultaneously
- Why evidence matters: Bayesian model comparison (Occam's razor, quantified)
- BlackJAX: GPU-native, JAX-based, differentiable, JIT-compiled
- jaxsgp4: Charlotte Priestley's differentiable SGP4 propagator -- a real GNSS forward model in JAX
- Preview of the hands-on session

### Hour 2: Hands-on Workshop (notebook)

#### Part 1: jaxsgp4 Forward Model (~15 min)
- Install dependencies
- Parse a TLE, propagate a satellite orbit
- Visualise the trajectory
- Generate synthetic "observed" positions with noise (simulating GNSS-R measurements)

#### Part 2: Bayesian Inference with BlackJAX NSS (~30 min)
- Define the inverse problem:
  - Parameters: orbital elements (subset -- e.g. inclination, eccentricity)
  - Forward model: jaxsgp4.sgp4()
  - Likelihood: Gaussian noise on position observations
  - Prior: uniform/informative over orbital element ranges
- Run blackjax.nss() nested sampling
- Monitor convergence

#### Part 3: Visualisation & Interpretation (~15 min)
- Convert to anesthetic NestedSamples
- Posterior corner plots with true values overlaid
- Evidence value and what it means
- Extension: model comparison (e.g. circular vs elliptical orbit)

## Dependencies

```bash
pip install jaxsgp4
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta"
pip install anesthetic tqdm matplotlib
```

## Key Resources

- jaxsgp4: https://github.com/cmpriestley/jaxsgp4 (arXiv:2603.27830)
- BlackJAX NSS: https://github.com/handley-lab/blackjax (tag v0.1.0-beta)
- Nested Sampling Book: https://handley-lab.co.uk/nested-sampling-book
- anesthetic: https://anesthetic.readthedocs.io

## Repository Structure

```
workshop-glitter/
├── README.md                    # This file
├── workshop_glitter.py          # Workshop script (development)
├── workshop_glitter.ipynb       # Interactive notebook
├── jaxsgp4/                     # Charlotte's SGP4 propagator (cloned)
└── slides/                      # Talk slides (TBD format)
```
