# LWD Host Staging

This document describes the host-staging implementation on `feat/lwd-main-migration`. UP prompt data has one arrival path: an early receive hint posts a receive into a cloud device slot, segmented D2H copies populate per-request pinned buffers, and only the resulting arrival reports make rows schedulable. There is no direct device-receive arrival path.

The cloud DP endpoint rank owns the cross-node UP receive and `LwdHostStager`; other cloud TP ranks receive the assembled window through the existing TP broadcast. This migration does not support prefix caching or cloud-side preemption, and introduces no other feature exclusions.

## Storage and watermarks

The first hint for a request allocates one pinned BF16 buffer for the **whole prompt**, including any DeepSeek V4 tail. Later chunks reuse that allocation at absolute prompt offsets. Full consumption marks the request done but does not free its buffer. Storage is released by `release_request` when the request reaches `finished_req_ids`, not when its last row is consumed.

There is no aggregate host-memory capacity or per-request byte cap. The fixed transport windows below are not host-storage limits.

For request `r`, define:

- **W1** (`recv_rows`): the contiguous prompt prefix whose D2H copies have completed and whose arrival has been published.
- **W2**: the scheduler-side check that native prefill scheduling stays within W1 and within the selected arrival window.

The stager maintains `0 <= consumed <= W1 <= prompt_rows`. Arrival gaps are rejected, repeated ranges are idempotent, and reads cannot cross W1. W2 requires `start + scheduled <= W1` and `scheduled <= window_rows`; native under-scheduling is allowed and retried later. At dispatch, the runner additionally requires `consumed == num_computed_tokens == window.start`.

The pipeline uses fixed limits: **pending=10** announced batches per cloud DP await request settlement or the UP send watermark before the edge gate holds; **early-recv=2** submitted, not yet reported UP receives per cloud communication thread; and **slots=2** reusable device receive slots.

A device slot is bounded by the maximum batched UP payload and is released after its D2H staging completes. It is distinct from the uncapped, request-lifetime host allocation.

## Single-path arrival flow

```mermaid
sequenceDiagram
    participant ES as Edge scheduler
    participant EW as Edge worker
    participant CE as Cloud engine
    participant CT as Cloud endpoint comm thread
    participant HS as Host stager / copy streams
    participant CS as Cloud scheduler
    participant CR as Cloud model runner

    ES->>CE: RequestNotify(prompt length, full-prompt M-RoPE)
    ES->>CE: RangeNotify(seqno, request rows and offsets)
    CE->>CT: Sideband hint(seqno, rows, offsets, prompt lengths)
    CT->>HS: Allocate device slot and full-prompt host buffers
    CT->>HS: Plan per-request D2H segments and post UP irecv
    ES->>EW: EMBED batch with the same seqno
    EW->>HS: UP embeddings and optional DSV4 tail
    HS->>HS: recv done_event -> segmented D2H -> staged_event
    CT->>CE: Report (UP, seqno) and cumulative rows offset + rows
    CE->>CS: Advance channel watermark and per-request W1 by max
    CS->>CS: Select windows and verify W2
    CS->>CR: Dispatch native mixed prefill/decode batch
    CR->>HS: H2D selected host windows; advance consumed
    CR->>CR: TP broadcast and inject remote embeddings / DSV4 tails
```

A hint is geometry and an early-posting trigger, **not** proof of arrival. `RangeNotify` alone never advances W1. The arrival report is emitted only after both the receive completion event and the D2H staging event complete. The engine drains that report at the start of a step, merges each cumulative request row count by maximum, and separately advances the UP channel watermark.

### Hint and arrival contract

Each hint carries `seqno`, total rows, and aligned request-ID, chunk-row, prompt-offset, and prompt-length tables. Duplicate hints and range notifications do not submit another receive. The stager plans source segments for the entire batch so that skipping an ended request does not shift the source offsets of live requests.

After D2H completion, `publish_received` advances each live request's contiguous W1 and releases the device slot. The completion report contains `(request_id, offset + rows)`, so the scheduler receives cumulative rather than per-chunk row counts.

Two different UP watermarks must not be conflated with W1:

- The **cloud receive watermark** is the highest staged-receive seqno reported on the channel, initially `-1`. It retires per-seqno engine bookkeeping; per-request W1, not this channel scalar, drives prefill readiness.
- The **edge send watermark** is the contiguous successful send-completion prefix per cloud DP. The edge worker returns it with normal and placeholder outputs, and the edge scheduler uses it to retire pending-batch backpressure. An out-of-order or failed send cannot advance it.

## D2H and H2D event ordering

For each staged receive, the UP channel stream records `done_event` after HCCL completion and any descrambling. A dedicated D2H stream waits for that event, issues non-blocking copies into the planned per-request host views, and records `staged_event`. The communication thread polls both events without blocking, publishes arrivals only after staging completes, and then drops its reference to the device receive tensor. The next channel operation waits on the predecessor's receive event, not on its D2H event.

For a scheduled prompt window, the endpoint runner selects one of two rotating device windows. Its dedicated H2D stream first waits for the window's previous-use event, then copies the selected embeddings and any DSV4 tails and records a ready event. The main stream waits for that ready event before injection and TP consumption. After dispatch, the stager advances `consumed`; injection records the window's new used event so a later H2D cannot overwrite storage still in use.

## Request completion, abort, and tombstones

Normal request completion follows the native `finished_req_ids` path. The cloud worker releases runner bookkeeping and the request's full-prompt host allocation, then removes batch metadata once no live request buffer references it.

Abort has two ordered effects:

1. The cloud engine retires the request from arrival accounting and enters the native abort path. Late arrival reports for that request are ignored.
2. An abort notice attached to an EMBED scheduler output calls `abort_request` in the stager. This records an ended-request tombstone and marks an existing buffer aborted, but does not free it immediately.

The tombstone suppresses new D2H destinations and late arrival publication for that request. Existing host storage remains frozen until the normal finish-release signal. A release also leaves a tombstone, so a late hint cannot recreate the request. The complete seqno hint and receive geometry remain intact for UP FIFO pairing; ended requests simply contribute no host segments or arrivals.

## Full-prompt M-RoPE

For multimodal requests, the edge computes the full-prompt `int64 [3, prompt_len]` M-RoPE positions once at admission and serializes them in `RequestNotify`. The cloud stores them on the request, attaches them to `scheduled_new_reqs`, and the runner validates `3 * prompt_len * 8` bytes before priming the full position tensor and its completion-position delta.

Per-chunk `LwdRangeItem.mrope_positions` remains empty. M-RoPE does not use the UP auxiliary frame and does not participate in host staging.

## DeepSeek V4 tail

For `L` hash layers and `K` selected experts per token, every UP row occupies `H + 4 * L * K` BF16 elements. The batch payload is `[all embedding rows | all encoded expert-ID bytes]`: each int32 expert ID is transported as four numeric byte values, avoiding BF16 rounding of large IDs.

Each request's host buffer has the analogous full-prompt layout `[prompt embeddings | prompt expert-ID tail]`. D2H planning splits both regions at absolute prompt offsets. H2D staging reassembles the scheduled window as embeddings followed by tails; the runner converts the tail bytes back to int32 IDs and registers them at the same absolute prompt positions for Hash MoE routing.

## Implementation map

| Area | Implementation |
| --- | --- |
| Hint creation and completion drain | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py` |
| W1 window selection and W2 verification | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py` |
| Range, send-watermark, and abort routing | `vllm/v1/lwd_control/control_edge_scheduler/` |
| Hint/arrival sideband protocol | `vllm_ascend/worker/lwd_cloud_model/lwd_pre_recv.py` |
| Per-request buffers and tombstones | `vllm_ascend/worker/lwd_cloud_model/lwd_host_stage.py` |
| Receive/D2H and window/H2D execution | `lwd_cloud_worker.py`, `lwd_cloud_model_runner.py`, and `distributed/lwd_data_plane/lwd_transfer_queue.py` |
