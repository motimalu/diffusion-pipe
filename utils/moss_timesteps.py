"""Discrete timestep distribution using the static Kohaku MOSS training recipe.

Independent numerical implementation of the probability law: condition
sigmoid(N(-mu, std)) on [0, cutoff], rescale to [0, 1], then quantize upwards
onto a grid. Mix with a uniform grid distribution and optional mass at t=1.
This is a static recipe, not the reference trainer's staged training schedule.

Ported from mage-flow-trainer (commit 6a39bfa4056b9c97e8965d1bb12adf7c9f0d1caf),
which itself is an independent reimplementation of the MOSS probability law.
"""

import math
from functools import lru_cache

import torch

MOSS_DEFAULTS = {
    'moss_mu': -1.09,
    'moss_std': 0.95,
    'moss_cutoff': 0.95,
    'moss_uniform_mix': 0.01,
    'moss_p_full': 0.0,
    'moss_steps': 1000,
}


@lru_cache(maxsize=32)
def probability_table(mu=-1.09, std=0.95, cutoff=0.95, uniform_mix=0.01,
                      p_full=0.0, steps=1000):
    """Return CPU FP32 probabilities for grid points j/steps, j=1..steps.

    Callers must not mutate this cached tensor. Integrate normal probabilities
    in double precision, using the survival function in the upper tail to
    avoid cancellation near CDF=1.
    """
    def normal_coordinate(x):
        if x == 0:
            return -math.inf
        if x == 1:
            return math.inf
        return (math.log(x) - math.log1p(-x) + mu) / std

    edges = [normal_coordinate(cutoff * i / steps) for i in range(steps + 1)]
    root_two = math.sqrt(2)
    masses = []
    for left, right in zip(edges, edges[1:]):
        if left >= 0:
            mass = (math.erfc(left / root_two) - math.erfc(right / root_two)) / 2
        else:
            mass = (math.erfc(-right / root_two) - math.erfc(-left / root_two)) / 2
        masses.append(max(0.0, mass))
    total = math.fsum(masses)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("MOSS timestep parameters leave no numerically representable mass below cutoff")
    probabilities = [(1 - p_full) * ((1 - uniform_mix) * m / total + uniform_mix / steps)
                     for m in masses]
    probabilities[-1] += p_full
    return torch.tensor(probabilities, dtype=torch.float32)


@lru_cache(maxsize=32)
def _cdf(mu, std, cutoff, uniform_mix, p_full, steps):
    cdf = probability_table(mu, std, cutoff, uniform_mix, p_full, steps).cumsum(0)
    cdf[-1] = 1.0
    return cdf


def moss_options(model_config):
    """Read the MOSS sampling options from a [model] config section, applying defaults."""
    return {name: model_config.get(name, default) for name, default in MOSS_DEFAULTS.items()}


def inverse_cdf(quantiles, model_config):
    """Map uniform quantiles in [0, 1] to discrete MOSS grid timesteps in (0, 1].

    Quantizes upwards, so t=0 (clean) is never returned and the pure-noise grid
    point t=1 is reachable even with moss_p_full=0.
    """
    options = moss_options(model_config)
    cdf = _cdf(options['moss_mu'], options['moss_std'], options['moss_cutoff'],
               options['moss_uniform_mix'], options['moss_p_full'],
               options['moss_steps']).to(quantiles.device)
    steps = options['moss_steps']
    indices = torch.searchsorted(cdf, quantiles.contiguous()).clamp_max(steps - 1)
    return (indices.float() + 1) / steps


def validate_moss_sampling(model_config):
    """Validate a [model] config section's MOSS timestep options. No-op for other methods.

    Raises ValueError on the first invalid option, mirroring the reference
    trainer's config-time safeguards.
    """
    if model_config.get('timestep_sample_method', 'logit_normal') != 'moss':
        return
    options = moss_options(model_config)
    if model_config.get('sigmoid_scale', 1.0) != 1.0:
        raise ValueError("Kohaku MOSS sampling locks sigmoid_scale=1; use moss_std for its distribution width")
    if (shift := model_config.get('shift', None)) is not None and shift != 1.0:
        raise ValueError("Kohaku MOSS sampling locks shift=1; remove the shift key or set it to 1.0")
    if model_config.get('flux_shift', False):
        raise ValueError("Kohaku MOSS sampling locks flux_shift=false")
    if not math.isfinite(options['moss_mu']):
        raise ValueError("model.moss_mu must be finite")
    if not math.isfinite(options['moss_std']) or options['moss_std'] <= 0:
        raise ValueError("model.moss_std must be finite and positive")
    if not 0 < options['moss_cutoff'] <= 1:
        raise ValueError("model.moss_cutoff must be in (0, 1]")
    if not 0 <= options['moss_uniform_mix'] <= 1:
        raise ValueError("model.moss_uniform_mix must be in [0, 1]")
    if not 0 <= options['moss_p_full'] <= 1:
        raise ValueError("model.moss_p_full must be in [0, 1]")
    if type(options['moss_steps']) is not int or not 2 <= options['moss_steps'] <= 100000:
        raise ValueError("model.moss_steps must be an integer in [2, 100000]")
    # Must leave numerically representable mass below the cutoff.
    probability_table(options['moss_mu'], options['moss_std'], options['moss_cutoff'],
                      options['moss_uniform_mix'], options['moss_p_full'], options['moss_steps'])
