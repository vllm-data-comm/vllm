# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD edge-cloud control communication.

`lwd_messages` defines the business messages. The rest is the ZMQ
ROUTER/DEALER control link, which carries those messages as opaque payloads.
"""
