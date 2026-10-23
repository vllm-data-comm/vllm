# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Control scheduling layer: edge/cloud scheduler and engine core bases.

Depends on `lwd_comm` in one direction only. The edge and cloud schedulers
subclass `LwdBaseScheduler` (an AsyncScheduler); engine access lives in the
two engine cores (`lwd_edge_engine` / `lwd_cloud_engine`) built on
`LwdBaseEngineCore`.
"""
