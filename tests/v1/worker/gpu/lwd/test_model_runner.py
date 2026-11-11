# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.lwd.model_runner import _validate_lwd_v2_config


def test_lwd_v2_rejects_deferred_context_parallelism() -> None:
    config = VllmConfig()
    config.parallel_config.prefill_context_parallel_size = 2

    with pytest.raises(ValueError, match="PCP support is deferred"):
        _validate_lwd_v2_config(config)


def test_lwd_v2_rejects_batch_sharded_sampling() -> None:
    config = VllmConfig()
    config.parallel_config.enable_batch_sharded_sampling = True

    with pytest.raises(ValueError, match="batch-sharded sampling"):
        _validate_lwd_v2_config(config)
