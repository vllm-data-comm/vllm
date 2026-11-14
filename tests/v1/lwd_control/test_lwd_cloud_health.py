# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    LwdCloudHealthTracker,
)


def test_failure_threshold_cools_cloud_until_expiry() -> None:
    tracker = LwdCloudHealthTracker((0, 1))
    for _ in range(2):
        tracker.record_failure(0, now=10.0, threshold=3, cooldown_s=5.0)
        assert tracker.candidates(now=10.0) == (0, 1)

    tracker.record_failure(0, now=10.0, threshold=3, cooldown_s=5.0)
    assert tracker.candidates(now=10.0) == (1,)
    assert tracker.candidates(now=14.9) == (1,)
    assert tracker.candidates(now=15.0) == (0, 1)


def test_all_cooling_clouds_fail_open() -> None:
    tracker = LwdCloudHealthTracker((1, 0))
    for cloud_id in (0, 1):
        tracker.record_failure(cloud_id, now=10.0, threshold=1, cooldown_s=5.0)
    assert tracker.candidates(now=10.0) == (0, 1)
    assert tracker.candidates(now=15.0) == (0, 1)


def test_success_clears_failures_and_cooling() -> None:
    tracker = LwdCloudHealthTracker((0, 1))
    for _ in range(2):
        tracker.record_failure(0, now=10.0, threshold=2, cooldown_s=60.0)
    assert tracker.candidates(now=10.0) == (1,)

    tracker.record_success(0)
    assert tracker.candidates(now=10.0) == (0, 1)
    tracker.record_failure(0, now=11.0, threshold=2, cooldown_s=60.0)
    assert tracker.candidates(now=11.0) == (0, 1)
