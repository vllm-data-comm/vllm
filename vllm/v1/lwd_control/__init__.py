# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD (layerwise disaggregated) edge-cloud control.

Diagnostics go through `lwd_control_diag`: functional classes neither read
environment variables nor import `vllm.v1.lwd_debug` directly.

Design: docs/design/lwd_control_plane.md.
"""
