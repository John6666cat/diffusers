# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumKrea2RawState, SpectrumState


def runtime_config(num_inference_steps=4, **kwargs):
    return SpectrumCacheConfig(
        num_inference_steps=num_inference_steps,
        coordinate_policy="runtime_index_normalized",
        forecast_step_indices=(1,),
        **kwargs,
    )


def start_valid(state, step, total=4, source="route_counter+cache_context"):
    state.start_step(
        logical_step_index=step,
        runtime_num_inference_steps=total,
        provenance_source=source,
    )


def test_legacy_state_does_not_require_runtime_provenance():
    state = SpectrumState(SpectrumCacheConfig(num_inference_steps=2))
    state.start_step()
    state.record_real_feature(torch.zeros(1, 2, 3))
    assert not state.coordinate_failure_latched
    assert state.forecaster.steps == [0]
    assert state.coordinate_summary()["coordinate_policy"] == "legacy_fixed_max"


def test_runtime_single_lane_accepts_route_owned_logical_steps_and_reports_telemetry():
    state = SpectrumState(runtime_config())
    start_valid(state, 0)
    state.record_real_feature(torch.zeros(1, 2, 3))
    start_valid(state, 1)

    assert not state.coordinate_failure_latched
    assert state.should_compute is False
    telemetry = state.coordinate_summary()
    assert telemetry["provenance_source"] == "route_counter+cache_context"
    assert telemetry["configured_num_inference_steps"] == 4
    assert telemetry["runtime_num_inference_steps"] == 4
    assert telemetry["last_logical_step_index"] == 1


def test_runtime_missing_provenance_latches_real_compute_and_blocks_history():
    state = SpectrumState(runtime_config())
    state.start_step()
    state.record_real_feature(torch.ones(1, 2, 3))

    assert state.coordinate_failure_latched
    assert state.coordinate_failure_latched_at == 0
    assert state.should_compute is True
    assert state.forecaster.steps == []
    with pytest.raises(RuntimeError, match="coordinate provenance"):
        state.predict()


def test_runtime_count_mismatch_clears_history_and_stays_fail_closed():
    state = SpectrumState(runtime_config())
    start_valid(state, 0)
    state.record_real_feature(torch.ones(1, 2, 3))
    assert state.forecaster.steps == [0]

    state.start_step(
        logical_step_index=1,
        runtime_num_inference_steps=5,
        provenance_source="route_counter+cache_context",
    )
    assert state.coordinate_failure_latched
    assert state.forecaster.steps == []

    state.record_real_feature(torch.full((1, 2, 3), 2.0))
    assert state.forecaster.steps == []
    start_valid(state, 2)
    assert state.should_compute is True
    assert state.forecaster.steps == []


def test_runtime_out_of_range_logical_index_fails_closed():
    state = SpectrumState(runtime_config())
    state.start_step(
        logical_step_index=4,
        runtime_num_inference_steps=4,
        provenance_source="route_counter+cache_context",
    )
    assert state.coordinate_failure_latched
    assert "outside runtime range" in state.coordinate_failures[-1]["reason"]


def test_runtime_non_monotonic_logical_index_fails_closed():
    state = SpectrumState(runtime_config())
    start_valid(state, 0)
    state.start_step(
        logical_step_index=0,
        runtime_num_inference_steps=4,
        provenance_source="route_counter+cache_context",
    )
    assert state.coordinate_failure_latched
    assert "route-owned logical step" in state.coordinate_failures[-1]["reason"]


def test_reset_clears_coordinate_failure_latch_and_allows_clean_runtime_trajectory():
    state = SpectrumState(runtime_config())
    state.start_step()
    assert state.coordinate_failure_latched

    state.reset()
    start_valid(state, 0)
    assert not state.coordinate_failure_latched
    assert state.coordinate_failures == []
    assert state.coordinate_last_step_index == 0


def test_krea_raw_dual_lane_uses_one_logical_step_for_each_cfg_pair():
    state = SpectrumKrea2RawState(runtime_config())
    sequence = [("positive", "pos", 0), ("negative", "neg", 0), ("positive", "pos", 1), ("negative", "neg", 1)]

    for expected_lane, identity, logical_step in sequence:
        state.prepare_call(identity)
        assert state.current_lane == expected_lane
        state.start_step(
            logical_step_index=logical_step,
            runtime_num_inference_steps=4,
            provenance_source="krea_cfg_pair+cache_context",
        )

    assert not state.coordinate_failure_latched
    assert state.call_index == 3
    assert state.step_index == 1
    assert state.coordinate_last_step_index == 1
    assert state.forecast_steps == [1]


def test_krea_raw_rejects_python_call_index_as_logical_step_provenance():
    state = SpectrumKrea2RawState(runtime_config())
    state.prepare_call("pos")
    state.start_step(
        logical_step_index=0,
        runtime_num_inference_steps=4,
        provenance_source="krea_cfg_pair+cache_context",
    )
    state.record_real_feature(torch.ones(1, 2, 3))

    state.prepare_call("neg")
    state.start_step(
        logical_step_index=1,
        runtime_num_inference_steps=4,
        provenance_source="krea_cfg_pair+cache_context",
    )

    assert state.call_index == 1
    assert state.step_index == 0
    assert state.coordinate_failure_latched
    assert state.should_compute is True
    assert all(forecaster.steps == [] for forecaster in state.forecasters.values())
    assert "route-owned logical step" in state.coordinate_failures[-1]["reason"]
