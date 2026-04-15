# Bayesian Inverse Problems, GPUs, and AI

**GLITTER Training School Workshop**

University of Luxembourg, 15 April 2026

Will Handley (University of Cambridge / PolyChord Ltd)

## Workshop Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/handley-lab/workshop-glitter/blob/master/workshop_glitter.ipynb)

Click the badge above to open the workshop notebook in Google Colab.

**Talk slides**: [[PDF](https://github.com/williamjameshandley/talks/raw/glitter_2026/will_handley_glitter_2026.pdf)] [[source](https://github.com/williamjameshandley/talks/tree/glitter_2026)]

## What this workshop covers

### Part 1: Bayesian Inference via Line Fitting

Starting from least-squares fitting (which you already know), we show that $\chi^2$ minimisation is equivalent to maximum likelihood, and that Bayes' theorem extends this to give:

- **Parameter estimation** with proper error bars (posterior distributions)
- **Predictive posteriors** showing model uncertainty in data space
- **Model comparison** via the Bayesian evidence (automatic Occam's razor)

All inference is done with [BlackJAX](https://github.com/handley-lab/blackjax) nested sampling, visualised with [anesthetic](https://anesthetic.readthedocs.io).

### Part 2: Satellite Orbit Determination

The same framework applied to a GNSS-relevant problem: inferring orbital elements from noisy satellite position observations using [jaxsgp4](https://github.com/cmpriestley/jaxsgp4), a differentiable SGP4 propagator in JAX.

- **2 parameters**: infer inclination and RAAN, visualise the posterior and predictive orbits
- **4 parameters**: add eccentricity and mean motion, see parameter degeneracies
- **Model comparison**: circular vs elliptical orbit -- can the data detect eccentricity?

### Part 3: Extensions

Open-ended exploration: change noise levels, add parameters, or apply the framework to your own forward model.

## Local installation

If not using Colab:

```bash
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git"
pip install jaxsgp4 anesthetic fgivenx tqdm matplotlib
```

Then open `workshop_glitter.ipynb` in Jupyter.

## Key resources

- [jaxsgp4](https://github.com/cmpriestley/jaxsgp4) -- differentiable SGP4 in JAX ([arXiv:2603.27830](https://arxiv.org/abs/2603.27830))
- [BlackJAX](https://github.com/handley-lab/blackjax) -- GPU-native nested sampling
- [Nested Sampling Book](https://handley-lab.co.uk/nested-sampling-book) -- tutorial and reference
- [anesthetic](https://anesthetic.readthedocs.io) -- nested sampling visualisation
