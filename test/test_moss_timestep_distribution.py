"""Static MOSS recipe, quantile sampling, and config safeguards.

Mirrors mage-flow-trainer's tests/test_moss_timestep_distribution.py
(commit 6a39bfa4056b9c97e8965d1bb12adf7c9f0d1caf), adapted to this
repository's dict-based [model] config.
"""
import math

import pytest
import torch

from utils.moss_timesteps import inverse_cdf, probability_table, validate_moss_sampling
from utils.common import get_t_distribution, sample_t


def model_config(**kwargs):
    return {'timestep_sample_method': 'moss', 'shift': 1.0, **kwargs}


def test_default_recipe_probability_and_high_noise_tail():
    p = probability_table()
    t = torch.arange(1, 1001) / 1000
    assert p.dtype == torch.float32 and p.shape == (1000,)
    assert (p > 0).all()
    assert p.sum().item() == pytest.approx(1, abs=2e-7)
    # Checked against the reference's static default table, not its staged recipe.
    assert (t * p).sum().item() == pytest.approx(.74416944, abs=2e-7)
    assert p[950:].sum().item() == pytest.approx(.09250056, abs=2e-7)
    # Upward quantization: the median grid point and the quantile endpoints.
    assert inverse_cdf(torch.full((1,), .5), model_config()).item() == pytest.approx(.781)
    assert inverse_cdf(torch.full((1,), 1.), model_config()).item() == 1
    assert p[-1] > .001  # pure-noise bin is present even with p_full=0
    # Empirical mean of many draws matches the table mean.
    q = torch.rand(100_000)
    assert inverse_cdf(q, model_config()).mean().item() == pytest.approx(.74416944, abs=1e-2)
    # t=0 (clean) is never sampled; values are exactly on the grid.
    samples = inverse_cdf(torch.rand(10_000), model_config())
    assert (samples > 0).all() and (samples <= 1).all()
    assert torch.equal(samples, torch.round(samples * 1000) / 1000)


def test_untruncated_recipe_matches_logistic_normal_bin_integrals():
    p = probability_table(0, 1, 1, 0, 0, 32).double()
    expected = [0.0]
    for j in range(1, 32):
        t = j / 32
        expected.append(.5 * math.erfc(-math.log(t / (1 - t)) / math.sqrt(2)))
    expected.append(1.0)
    torch.testing.assert_close(p, torch.tensor(expected, dtype=torch.float64).diff(),
                               atol=1e-8, rtol=1e-6)


def test_uniform_mixture_and_extra_pure_noise_mass():
    p = probability_table(uniform_mix=1, p_full=.2, steps=10)
    torch.testing.assert_close(p, torch.tensor([.08] * 9 + [.28]))
    torch.testing.assert_close(
        inverse_cdf(torch.rand(24), model_config(moss_p_full=1)), torch.ones(24))


def test_quantile_sampling_is_deterministic():
    a = inverse_cdf(torch.full((1000,), .37), model_config())
    b = inverse_cdf(torch.full((1000,), .37), model_config())
    assert torch.equal(a, b)
    assert a.min().item() == a.max().item()  # fixed quantile -> single grid point
    lo = inverse_cdf(torch.full((8,), .1), model_config())[0].item()
    hi = inverse_cdf(torch.full((8,), .9), model_config())[0].item()
    assert lo < hi


def test_cdf_computation_does_not_mutate_cached_table():
    from utils.moss_timesteps import _cdf
    p = probability_table()
    _cdf(-1.09, 0.95, 0.95, 0.01, 0.0, 1000)
    assert torch.equal(p, probability_table())
    assert probability_table().sum().item() == pytest.approx(1, abs=2e-7)


def test_wan_bucket_distribution_samples_moss_grid():
    t = get_t_distribution(model_config())
    assert t.shape == (10_000,)
    assert torch.equal(t, torch.round(t * 1000) / 1000)
    samples = sample_t(t, 10_000)
    assert samples.mean().item() == pytest.approx(.74416944, abs=1e-2)
    # The bucket table approximates the inverse CDF at bucket centers.
    expected = inverse_cdf(torch.linspace(1 / 10_000, 1 - 1 / 10_000, 10_000), model_config())
    assert torch.equal(t, expected)


@pytest.mark.parametrize("options", [
    {'shift': 3}, {'sigmoid_scale': 3}, {'flux_shift': True},
    {'moss_std': 0}, {'moss_std': float('nan')}, {'moss_mu': float('inf')},
    {'moss_cutoff': 0}, {'moss_cutoff': 1.01}, {'moss_uniform_mix': -.1},
    {'moss_p_full': 1.01}, {'moss_steps': 1}, {'moss_steps': 1000.5},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        validate_moss_sampling(model_config(**options))


def test_valid_alternatives():
    # shift may be omitted entirely (identity shift) or set to 1.
    validate_moss_sampling({'timestep_sample_method': 'moss'})
    validate_moss_sampling(model_config(shift=1))
    validate_moss_sampling(model_config(shift=1.0, moss_cutoff=1.0, moss_steps=2))
    # Other methods pass through untouched.
    validate_moss_sampling({'timestep_sample_method': 'logit_normal', 'shift': 3})
    validate_moss_sampling({'timestep_sample_method': 'uniform'})
    validate_moss_sampling({})


def test_toml_defaults_and_conflict(tmp_path):
    import toml
    from utils.moss_timesteps import moss_options
    path = tmp_path / 'test.toml'
    content = '[model]\ndtype="bfloat16"\ntimestep_sample_method="moss"\n'
    path.write_text(content)
    cfg = toml.loads(path.read_text())['model']
    # Unset shift is fine (identity shift); the recipe defaults apply.
    validate_moss_sampling(cfg)
    assert moss_options(cfg) == {'moss_mu': -1.09, 'moss_std': 0.95, 'moss_cutoff': 0.95,
                                 'moss_uniform_mix': 0.01, 'moss_p_full': 0.0, 'moss_steps': 1000}
    path.write_text(content + 'shift=3\n')
    with pytest.raises(ValueError, match='locks'):
        validate_moss_sampling(toml.loads(path.read_text())['model'])
