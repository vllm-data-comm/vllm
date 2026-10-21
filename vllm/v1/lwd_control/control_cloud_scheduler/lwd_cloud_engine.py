# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine core: serves the edge of the same DP rank over the control
plane. Edge messages are handled on an ingest thread (range notifies feed the
scheduler and the pre-recv hint, request notifies become native ADDs); step
metadata is flushed to the edge in post_step."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import torch

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_INGEST_JOIN_TIMEOUT_S = 5.0

_LwdIngestItem = tuple[LwdPeer, LwdEdgeToCloudMessage]


class LwdCloudEngineCore(LwdBaseEngineCore):
    """Cloud PO engine: control plane + ingest thread, the rest is native."""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The workers inherit the hint MQ handle through the env, so the
        # hint MQ must exist before super() builds the executor.
        lwd_config = kwargs["vllm_config"].lwd_config
        self._lwd_early_recv = lwd_config.topology.feature_ctrl.enable_early_recv
        self._lwd_recv_hint_mq: MessageQueue | None = None
        self._lwd_irecv_done_mq: MessageQueue | None = None
        if self._lwd_early_recv:
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                create_hint_mq,
            )

            self._lwd_recv_hint_mq = create_hint_mq()
        super().__init__(*args, **kwargs)

    def _lwd_assemble(self) -> None:
        self._lwd_ingest_thread: threading.Thread | None = None
        self._lwd_edge: LwdPeer | None = None
        sched_config = self.vllm_config.scheduler_config
        # Scheduling limits the edge checks (same inputs of the cap/preflight
        # formulas on both sides).
        self._lwd_cloud_params = LwdCloudParamsNotify(
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
        )
        self.scheduler.lwd_early_recv = self._lwd_early_recv
        if self._lwd_early_recv:
            self._lwd_attach_early_recv()
        lwd_logger.info("[pre-recv] early receive %s", "on" if self._lwd_early_recv else "off")
        # Edge messages are handled off the IO thread: building a request
        # allocates its placeholder embeds and the pre-recv hint may block,
        # which must not delay step metadata going out.
        self._lwd_ingest_queue: queue.SimpleQueue[_LwdIngestItem | None] = (
            queue.SimpleQueue()
        )
        self._lwd_ingest_thread = threading.Thread(
            target=self._lwd_ingest_loop, name="lwd-cloud-ingest", daemon=True
        )
        self._lwd_ingest_thread.start()
        plane = self._lwd_start_control_plane()
        lwd_logger.info("[cloud] engine assembled: control plane on %s", plane.endpoint)

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_edge_to_cloud(payload)

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: remember the edge and send it the scheduling
        limits before anything else."""
        self._lwd_edge = peer
        self._lwd_send_until_queued(peer, self._lwd_cloud_params)
        lwd_logger.info(
            "[cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            self._lwd_cloud_params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full outbound queue until the message is queued; gives up
        with an error once shutdown is requested, so a stalled edge cannot
        block it."""
        while not self._lwd_send(peer, message):
            if self.shutdown_state != EngineShutdownState.RUNNING:
                lwd_logger.error(
                    "[cloud] %s to edge %s dropped: shutting down with the "
                    "outbound queue full",
                    type(message).__name__,
                    peer,
                )
                return
            time.sleep(_LWD_SEND_RETRY_SLEEP_S)

    def _lwd_attach_early_recv(self) -> None:
        """Wire the two early-recv side channels to the cloud endpoint
        worker's comm thread.

        They bypass rpc_broadcast_mq: the worker busy loop consumes it on one
        thread and blocks for a whole prefill forward, so a hint queued there
        would not be early. Hints go out on the hint MQ created before the
        executor; completions come back on the done MQ, whose writer the
        endpoint worker created in init_device.
        """
        # TODO(lwd): the protocol lives in vllm-ascend, so this vLLM engine
        # imports vllm_ascend; move the hint/watermark protocol into vLLM or
        # behind a platform hook, issue #19.
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            register_hint_sender,
        )

        hint_mq = self._lwd_recv_hint_mq
        assert hint_mq is not None
        # Blocking delivery: with the gate on, a lost hint means nobody
        # pre-posts that recv, the watermark stalls, and the batch behind the
        # gate is never dispatched.
        register_hint_sender(hint_mq.enqueue)
        # TODO(lwd): one endpoint per cloud instance today; key the done MQs
        # by DP connection once LWD supports several DPs per cloud.
        handles = [
            handle
            for handle in self.model_executor.collective_rpc(
                "lwd_early_recv_done_handle"
            )
            if handle is not None
        ]
        if len(handles) != 1:
            raise RuntimeError(
                f"[LWD] early receive expects one endpoint worker reporting "
                f"completions, got {len(handles)}"
            )
        self._lwd_irecv_done_mq = MessageQueue.create_from_handle(handles[0], 0)

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: hand range and request notifies to the ingest thread;
        anything else (incl. abort, not wired yet) is unknown."""
        if not isinstance(msg, (LwdRangeNotify, LwdRequestNotify)):
            return False
        self._lwd_ingest_queue.put((peer, msg))
        return True

    def _lwd_ingest_loop(self) -> None:
        while (item := self._lwd_ingest_queue.get()) is not None:
            try:
                self._lwd_handle_edge_message(*item)
            except Exception as e:
                lwd_logger.exception(
                    "[cloud] ingest of %s failed", type(item[1]).__name__
                )
                self._lwd_fail(f"ingest of {type(item[1]).__name__}: {e!r}")
                return

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        """Ingest thread: range notify -> notify queue + pre-recv hint;
        request notify -> native ADD."""
        # TODO(lwd): several edges per cloud DP rank need requests tracked by
        # the peer that owns them, issue #8.
        if isinstance(msg, LwdRangeNotify):
            # 范围预告:整条入通知队列(重复预告即重复点名,调度器分类
            # 幂等剔除,无副作用)。预告自带 seqno,出批时作 UP 链配对号。
            self.scheduler.prefill_notify_queue.append(msg)
            if self._lwd_early_recv:
                self._lwd_post_recv_hint(msg)
        elif isinstance(msg, LwdRequestNotify):
            request = self._lwd_build_request(msg)
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))

    @staticmethod
    def _lwd_post_recv_hint(msg: LwdRangeNotify) -> None:
        """Ingest thread: hint the endpoint worker as soon as the range
        notify arrives, so its comm thread posts the exact-size UP irecv
        without waiting for this batch's SchedulerOutput."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            make_recv_hint,
            post_irecv_hint,
        )

        num_tokens = sum(item.num_tokens for item in msg.items)
        # mrope positions ride the control plane, so UP carries only the
        # main frame: a nonzero aux_rows would pre-post an aux irecv that
        # never matches and stall the watermark.
        post_irecv_hint(make_recv_hint(msg.seqno, num_tokens, aux_rows=0))

    def shutdown(self) -> None:
        # Before the executor goes: a pre-recv hint blocks on its queue.
        if self._lwd_ingest_thread is not None:
            self._lwd_ingest_queue.put(None)
            self._lwd_ingest_thread.join(_LWD_INGEST_JOIN_TIMEOUT_S)
        super().shutdown()

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """请求构建(唯一建请求点):采样参数自请求预告重建 + 占位 meta
        embeds(仅形状无 storage,prompt 长度即行数;worker 侧按 is_meta
        本地重建零缓冲,UP 注入直接写该缓冲窗口)。前缀缓存哈希链未随
        控制面移植(功能另上,届时再挂 block_hasher)。"""
        sampling_params = self._lwd_build_sampling_params(req_notify)
        # 不传真实 ids 也不造占位:按原生 prompt-embeds 语义挂 meta 占位
        # (空 storage,pickle 只带元信息,不触 MQ overflow 慢通道),
        # 行数即 prompt 长度;ids=None 时 input_batch 自动把 prompt 段
        # is_token_ids 置 False,M-RoPE 走纯文本直通构造(与扫描结果
        # 逐值一致)。
        prompt_embeds = torch.empty(
            req_notify.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
            device="meta",
        )
        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
        )
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """采样参数自请求预告重建(字段表驱动);eos_token_id 走原生回填
        入口(设 _eos_token_id 并计入 _all_stop_token_ids 供 min_tokens
        判定;云侧无客户端 generation_config,传空)。"""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(**values)
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        LwdControlDiag.cloud_request_admitted(req_notify.request_id, sampling_params)
        return sampling_params

    def step_with_batch_queue(self):
        """步首排空提前收水位(done MQ -> 闸门就绪判据),其余走原生。"""
        self._lwd_drain_irecv_completions()
        out = super().step_with_batch_queue()
        return out

    def step(self):
        """同步步路径同款:步首排水位,其余走原生。"""
        self._lwd_drain_irecv_completions()
        out = super().step()
        return out

    def _lwd_drain_irecv_completions(self) -> None:
        """每步排空 done MQ,把 worker 收完的 UP recv 并进提前收水位(闸门就绪判据)。"""
        done_mq = self._lwd_irecv_done_mq
        if done_mq is None:
            return
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        items = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                # 关停时 ring 被取消:记一条即可,水位停在原处不影响收尾
                lwd_logger.exception("[pre-recv] done MQ drain failed")
                break
        if not items:
            return
        record_irecv_completions(items)

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        if self.scheduler.pending_c2e:
            assert self._lwd_edge is not None, "c2e produced before any edge"
            for notify in self.scheduler.pending_c2e:
                self._lwd_send_until_queued(self._lwd_edge, notify)
            self.scheduler.pending_c2e.clear()
        super().post_step(model_executed)
