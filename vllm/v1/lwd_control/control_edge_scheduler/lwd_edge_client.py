# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Edge prefix-negotiation client for an SSE gateway.

Derived from ``edge_cloud/edge_client.py`` on the reference branch: POST an
HMAC manifest shadow body, parse the Probe control message from SSE, and drain
the usage stream in the background. Raw messages and sampling parameters never
cross the network.

OpenAI-compatible protocol, forwarded transparently by the gateway:
  Request: POST control_url with a manifest shadow body, X-Edge-Cloud-*
           headers, and an optional Authorization Bearer header. Image
           requests declare ``edge-cloud-prefix-v2`` and include the
           ``X-Edge-Cloud-MM-ABI`` fingerprint.
  Response: an SSE stream whose first chunk embeds the edge_cloud_probe JSON
            control message in delta.content, followed by empty delta frames
            and heartbeat comments. The final chunk carries standard OpenAI
            usage and [DONE]. A gateway that preserves delta.content can use
            this stream for billing.

Usage handling: drain the SSE stream in the background and record usage in
edge logs. This complements ``LwdC2eNotify.usages`` on the data plane, which
supplies accounting and frontend usage.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import Any

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    mm_abi_header_value,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
)

logger = init_logger(__name__)

# Maximum SSE control-message JSON size, guarding against malicious responses.
_MAX_CONTROL_MESSAGE_CHARS = 16 * 1024


async def _iter_sse_data(
    content: AsyncIterable[bytes],
) -> AsyncIterator[str]:
    """Read SSE events, preserving multiline data and ignoring heartbeats."""
    data: list[str] = []
    size = 0
    async for raw_line in content:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data.clear()
                size = 0
                yield payload
        elif line == "data" or line.startswith("data:"):
            value = line.partition(":")[2].removeprefix(" ")
            size += len(value) + 1
            if size > _MAX_CONTROL_MESSAGE_CHARS:
                raise ValueError(
                    "edge-cloud SSE event exceeds the control message limit"
                )
            data.append(value)
    if data:
        yield "\n".join(data)


class LwdEdgePrefixClient:
    """Edge admission client: HMAC manifest shadow body and SSE Probe parsing."""

    def __init__(
        self,
        control_url: str,
        block_size: int,
        tenant_key: bytes,
        edge_id: int = 0,
        consumer_id: str | None = None,
        connect_timeout: float = 10.0,
        processor_fingerprint: bytes | None = None,
        probe_timeout: float = 30.0,
    ) -> None:
        if not control_url.startswith(("http://", "https://")):
            raise ValueError("control_url must use http:// or https://")
        if probe_timeout <= 0:
            raise ValueError("probe_timeout must be positive")
        self._control_url = control_url
        self._hasher = LwdPrefixHasher(tenant_key, block_size, processor_fingerprint)
        # MM-ABI header for image requests, rendered from the local processor
        # fingerprint; without one, the client cannot issue a v2 request.
        self._mm_abi_header: str | None = (
            mm_abi_header_value(processor_fingerprint)
            if processor_fingerprint is not None
            else None
        )
        self._edge_id = edge_id
        self._consumer_id = consumer_id
        self._connect_timeout = connect_timeout
        # End-to-end bound for the whole probe (connect plus first frame). The
        # cloud bound is shorter, so normally the cloud responds first. The
        # value comes from lwd_coordination.probe_timeout.
        self._probe_timeout = probe_timeout
        # Background probe-stream drain tasks. asyncio holds only weak
        # references, so this set must retain strong references; see
        # _start_drain for why tasks are not indexed by req_id.
        self._drain_tasks: set[Any] = set()
        # Sensitive gateway Bearer token, read from the environment and never
        # logged. None means no Authorization header.
        api_key = envs.VLLM_LWD_CLOUD_API_KEY
        if api_key is not None and (
            not api_key
            or not api_key.isascii()
            or not api_key.isprintable()
            or any(c.isspace() for c in api_key)
        ):
            raise ValueError(
                "VLLM_LWD_CLOUD_API_KEY must contain a non-empty ASCII "
                "API token without whitespace"
            )
        self._api_key: str | None = api_key

    @property
    def block_size(self) -> int:
        return self._hasher.block_size

    def build_manifest(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        media_items: Sequence[Any] = (),
    ) -> LwdPrefixManifest:
        """Build an HMAC-chained manifest without exposing the raw prompt."""
        return self._hasher.build_manifest(
            request_id, prompt_token_ids, media_items=media_items
        )

    def build_control_request(
        self,
        manifest: LwdPrefixManifest,
        model: str | None = None,
        *,
        multimodal: bool = False,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Build a whitelisted shadow body; sampling parameters stay local.

        ``LwdPrefixManifest.to_headers()`` supplies all required control
        headers, including ``X-Edge-Cloud-Prompt-Tokens``. Add only edge
        identity and gateway headers here; optionally pass ``model`` through
        for gateway channel selection. The caller sets ``multimodal`` based on
        the ``media_items`` used to build the chain. Media-covered blocks use
        hash domains 0x04/0x05, so the request must declare v2 and carry the
        MM-ABI fingerprint for the cloud to interpret the chain correctly.
        """
        headers = manifest.to_headers()
        if multimodal:
            if self._mm_abi_header is None:
                raise ValueError(
                    "multimodal probe needs a processor fingerprint for "
                    f"{HEADER_MM_ABI}"
                )
            headers[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers[HEADER_MM_ABI] = self._mm_abi_header
        headers[HEADER_EDGE_ID] = str(self._edge_id)
        headers["Accept"] = "text/event-stream"
        if self._consumer_id:
            headers["X-Mse-Consumer"] = self._consumer_id
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "messages": manifest.to_messages(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if model is not None:
            body["model"] = model
        return headers, body

    async def negotiate(
        self,
        session: Any,
        request_id: str,
        prompt_token_ids: list[int],
        manifest: LwdPrefixManifest | None = None,
        media_items: Sequence[Any] = (),
        model: str | None = None,
    ) -> LwdProbeResult:
        """POST a probe and parse Probe from SSE delta.content; fail closed.

        The caller consumes the hit and advances request.num_computed_tokens
        so the edge embeds and sends only the suffix. An existing ``manifest``
        can be reused to avoid building the O(N) chain twice per probe;
        otherwise one is built now, and ``media_items`` must match the caller's
        chain inputs so multimodal identity is not omitted.

        ``probe_timeout`` bounds connection and first-frame receipt. The cloud
        handler uses ``probe_timeout - 5``, so a slow cloud normally responds
        first with 200 or 503. This timeout is only a fallback when the cloud
        or gateway does not respond; it must not abandon a stream that is still
        awaiting usage. See ``_post_and_read_probe`` for ownership on abort.
        """
        if manifest is None:
            manifest = self.build_manifest(request_id, prompt_token_ids, media_items)
        # A nonempty media identity means domains 0x04/0x05 were mixed into
        # the chain, so this request must use protocol v2.
        multimodal = bool(media_items)
        headers, body = self.build_control_request(
            manifest, model=model, multimodal=multimodal
        )
        try:
            response, sse_data, probe = await asyncio.wait_for(
                self._post_and_read_probe(
                    session,
                    headers,
                    body,
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    multimodal=multimodal,
                ),
                self._probe_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"prefix probe timed out after {self._probe_timeout}s "
                f"url={self._control_url}"
            ) from exc
        # Return as soon as the probe arrives, but do not close the stream:
        # the cloud handler waits for inference to finish before sending the
        # final usage chunk (up to 600s), which carries actual token usage.
        self._start_drain(request_id, response, sse_data)

        logger.info(
            "[LWD][edge-client] negotiate req=%s hit_tokens=%d cloud_id=%s",
            request_id,
            probe.hit_tokens,
            probe.cloud_id,
        )
        return probe

    async def _post_and_read_probe(
        self,
        session: Any,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        request_id: str,
        prompt_token_ids: list[int],
        multimodal: bool,
    ) -> tuple[Any, AsyncIterator[str], LwdProbeResult]:
        """Send the shadow body and read Probe; drain or close on abort.

        Receiving HTTP 200 means the cloud handler is already awaiting usage
        for this request. After that, never close on timeout, cancellation, or
        echo-validation failure: an early disconnect makes the gateway record
        client_gone/context canceled and permanently loses actual usage.
        """
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=None, connect=self._connect_timeout)
        response = await session.post(
            self._control_url, json=body, headers=headers, timeout=timeout
        )
        # Non-None means HTTP 200 was received and the stream iterator exists;
        # on abort this determines whether to drain or close.
        sse_data: AsyncIterator[str] | None = None
        try:
            if response.status != 200:
                detail = await self._read_error_body(response)
                raise RuntimeError(
                    f"prefix negotiation HTTP {response.status} "
                    f"url={self._control_url} body={detail}"
                )
            # Share this iterator with the drain task, which continues reading
            # frames after the probe frame.
            sse_data = _iter_sse_data(response.content)
            probe = await self._read_probe_from_sse(sse_data, multimodal=multimodal)
            if probe.request_id != request_id:
                raise RuntimeError("cloud returned a different request ID")
            # Block granularity need not match: for hybrid models such as
            # Mamba/SWA, the cloud hash granularity may be smaller than the
            # edge KV block size. The caller learns probe.block_size and
            # rebuilds the chain (see _lwd_hash_bs_by_cloud).
            if probe.hit_tokens > len(prompt_token_ids):
                raise RuntimeError("cloud prefix hit exceeds the prompt length")
        except BaseException:
            if sse_data is not None:
                self._start_drain(request_id, response, sse_data)
            else:
                response.close()
            raise
        assert sse_data is not None  # The success path always creates a stream.
        return response, sse_data, probe

    def _start_drain(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> Any:
        """Drain frames after Probe in a background task on the same event loop.

        Registration must use a set, not a dict indexed by req_id: asyncio
        holds only weak references, and a coroutine blocked on a network read
        has no pending callback, so this set is its only strong reference.
        Overwriting an earlier task with the same req_id on re-add/retry would
        drop that stream's only strong reference; it could be garbage
        collected before usage arrives, causing gateway client_gone and lost
        usage. Return the task only so tests can assert its registration.
        """
        task = asyncio.create_task(self._drain_stream(request_id, response, sse_data))
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)
        return task

    async def _drain_stream(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> None:
        """Drain the probe stream through its final usage chunk for billing.

        The edge only logs usage; accounting uses the cloud-to-edge C2e
        ``[LWD][edge] usage`` record. The stream must still be consumed so the
        cloud handler reaches its final chunk and [DONE]. The cloud waits at
        most 600s for usage and then sends [DONE] without a usage frame, so
        task count is bounded by in-flight requests and needs no extra timeout.
        Missing usage is logged as a warning because the gateway cannot bill
        upstream usage for that request.
        """
        usage: dict | None = None
        try:
            async for payload in sse_data:
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
            if usage is None:
                logger.warning(
                    "[LWD][edge-client] usage missing in probe stream req=%s",
                    request_id,
                )
                return
            details = usage.get("prompt_tokens_details") or {}
            logger.info(
                "[LWD][edge-client] usage req=%s prompt=%s completion=%s "
                "total=%s cached=%s",
                request_id,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
                details.get("cached_tokens"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  Only this gateway bill is lost.
            logger.warning(
                "[LWD][edge-client] probe stream drain failed req=%s",
                request_id,
                exc_info=True,
            )
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001  Do not propagate close failures.
                logger.debug(
                    "[LWD][edge-client] probe stream close failed req=%s",
                    request_id,
                    exc_info=True,
                )

    @staticmethod
    async def _read_error_body(response: Any, limit: int = 512) -> str:
        """Read a non-200 error body, truncated and whitespace-normalized."""
        try:
            # aiohttp's ClientResponse.read() takes no size argument; the
            # capped read lives on the content stream. Passing the limit to
            # read() raises TypeError and hides every non-200 body (including
            # the cloud's own error text) behind "<unreadable: TypeError>".
            raw = await response.content.read(limit)
            return " ".join(raw.decode("utf-8", "replace").split())
        except Exception as exc:  # noqa: BLE001  Diagnostics must not raise.
            return f"<unreadable: {type(exc).__name__}>"

    async def _read_probe_from_sse(
        self, sse_data: AsyncIterator[str], *, multimodal: bool
    ) -> LwdProbeResult:
        """Parse the first edge_cloud_probe message from SSE delta.content.

        Validate every echoed field and fail closed: the cloud protocol must
        match the request. Silently downgrading a v2 image request to v1 would
        account for media-covered blocks as text and misalign cache hits. V2
        must echo the local MM-ABI fingerprint, while text-only replies must
        omit it. A usage/error frame before Probe indicates invalid stream
        ordering and is an error. Gateway-fragmented content is accumulated,
        so validate only after a complete JSON object has been assembled.
        """
        expected_protocol = PROTOCOL_VERSION_MM if multimodal else PROTOCOL_VERSION
        content_buf = ""
        async for payload in sse_data:
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if not isinstance(chunk, dict):
                raise ValueError("cloud returned a non-object SSE chunk")
            if chunk.get("usage") is not None or chunk.get("error") is not None:
                raise RuntimeError("cloud returned usage or an error before the probe")
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content_buf += delta.get("content") or ""
            if not content_buf:
                continue
            try:
                message = json.loads(content_buf)
            except json.JSONDecodeError:
                continue  # Fragmented JSON is incomplete; keep reading.
            if not isinstance(message, dict):
                raise ValueError("cloud probe must contain a JSON object")
            if "protocol" not in message:
                continue  # Split after the first key; read one more frame.
            if message["protocol"] != expected_protocol:
                raise RuntimeError(
                    f"cloud did not acknowledge {expected_protocol} "
                    f"(returned {message['protocol']!r})"
                )
            mm_abi = message.get("mm_abi")
            if multimodal:
                if mm_abi != self._mm_abi_header:
                    raise RuntimeError(
                        "cloud MM-ABI fingerprint does not match the local "
                        "processor fingerprint"
                    )
            elif mm_abi is not None:
                raise RuntimeError("cloud returned MM-ABI for a text-only probe")
            return LwdProbeResult.from_control_message(message)
        raise RuntimeError("edge-cloud probe control message not found in SSE stream")
