# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import json
from dataclasses import asdict

import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule


# Full dataclass-field fingerprints for the 80 configurations of each route.
# Sorted JSON makes the comparison independent of formatting and field order.
ROUTES = [
    ('base', 't2i', '1231ada44ab4acef0b364a0c4a6ea732b2c992e36c0aeb1af10cf279c6466508'),
    ('base', 'i2i', '434e461d1d53daae2659a998886c3dee5ede501e718c6fe9c3daf4c21e4a4abb'),
    ('distilled', 't2i', 'd672db4bef3aaa838e72b4767b2db64f160c05e6db966c056e1065db318f6c5a'),
    ('distilled', 'i2i', '1cbe3c37e83c946850e6ea91b1f1bb74d5f5ea227a3037944a7b7669c203d9cc'),
]


@pytest.mark.parametrize("checkpoint,workflow,expected", ROUTES)
def test_flux3_variable_full_config_matrix(checkpoint, workflow, expected):
    configs = []
    warmup = 5 if checkpoint == "base" else 4
    cap = 2 if (checkpoint, workflow) == ("base", "t2i") else 1
    for n in range(1, 81):
        cfg = SpectrumCacheConfig.for_flux3_variable(n, checkpoint=checkpoint, workflow=workflow)
        schedule = SpectrumSchedule(cfg)
        steps = tuple(i for i in range(n) if not schedule.decide(i))
        assert steps == cfg.forecast_step_indices
        assert cfg.warmup_steps == min(warmup, n)
        assert all(warmup <= i < n - 2 for i in steps)
        assert all(steps[i + cap] - steps[i] != cap for i in range(len(steps) - cap))
        assert n > 11 or not steps
        assert cfg.allow_flux3_distilled == (checkpoint == "distilled")
        assert cfg.allow_flux3_image_conditioned == (checkpoint == "base" and workflow == "i2i")
        assert cfg.allow_flux3_distilled_image_conditioned == (checkpoint == "distilled" and workflow == "i2i")
        configs.append(asdict(cfg))
    actual = hashlib.sha256(json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert actual == expected


@pytest.mark.parametrize("checkpoint,workflow,n,steps", [
    ("base", "t2i", 50, (6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29,
                           30, 32, 33, 35, 36, 38, 39, 41, 42, 44)),
    ("base", "i2i", 50, (8, 12, 18, 24, 30, 36, 42, 44)),
    ("distilled", "t2i", 16, (6, 10)),
    ("distilled", "i2i", 16, (6, 10)),
])
def test_flux3_variable_preserves_anchor(checkpoint, workflow, n, steps):
    expected = SpectrumCacheConfig(
        num_inference_steps=n, warmup_steps=5 if checkpoint == "base" else 4,
        window_size=2.0, flex_window=0.75, forecast_step_indices=steps,
        degree=3 if checkpoint == "base" else 2, blend_w=0.35,
        coordinate_policy="runtime_horizon_normalized",
        allow_flux3_image_conditioned=checkpoint == "base" and workflow == "i2i",
        allow_flux3_distilled=checkpoint == "distilled",
        allow_flux3_distilled_image_conditioned=checkpoint == "distilled" and workflow == "i2i",
    )
    assert asdict(SpectrumCacheConfig.for_flux3_variable(n, checkpoint=checkpoint, workflow=workflow)) == asdict(expected)


@pytest.mark.parametrize("n", [0, -1, 81, 12.0, True, "12", None])
def test_flux3_variable_invalid_n(n):
    with pytest.raises(ValueError):
        SpectrumCacheConfig.for_flux3_variable(n, checkpoint="base", workflow="t2i")


@pytest.mark.parametrize("checkpoint,workflow", [("other", "t2i"), ("base", "other"), (None, "i2i")])
def test_flux3_variable_invalid_route(checkpoint, workflow):
    with pytest.raises(ValueError):
        SpectrumCacheConfig.for_flux3_variable(16, checkpoint=checkpoint, workflow=workflow)
