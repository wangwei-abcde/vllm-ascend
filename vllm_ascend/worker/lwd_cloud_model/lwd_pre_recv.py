# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD early receive: engine⇄worker bypass link + hint/watermark protocol.

There is exactly one data path (same as the existing early receive; no new bypass)::

    cloud engine IO thread                cloud endpoint comm thread
    ───────────────                       ──────────────────────
    RangeNotify arrives (edge announcement)
      └─ post_irecv_hint ──hint MQ (downstream)──►  start_early_irecv
                                              └─ submit_recv(UP, seqno, numel, aux)
                                                    │  irecv already pre-posted
                                              future.done() polling
                                                    │
    record_irecv_completions ◄──done MQ (upstream)── _report_irecv_completions
      └─ watermark → is_irecv_complete(UP, seqno)
           └─ PD_mix scheduler gate: release the prefill batch only once received

Both MQs are bypasses; neither may borrow ``rpc_broadcast_mq``: that queue
is consumed single-threaded by ``worker_busy_loop``, which blocks for
hundreds of milliseconds inside ``execute_model`` for a prefill batch — a
hint queued there would never get its recv posted "early".

* hint MQ: the engine creates the writer (handle exported to env before
  worker creation); the cloud endpoint rank attaches the reader by handle;
* done MQ: the cloud endpoint rank creates the writer; after the executor
  is up, the engine fetches its handle over ``collective_rpc``
  (``LwdCloudWorker.lwd_early_recv_done_handle``) and attaches the reader.

Watermark semantics: ``is_irecv_complete(channel, seqno)`` ==
``seqno <= watermark``. Per-channel completion order = seqno order
(``channel.py`` FIFO: if the head has not completed, nothing behind it
has), so the "completed set" is always a prefix and one max watermark
represents it exactly. The watermark is only a **readiness predicate**;
it plays no role in task priority.

Hint fields on this branch (chunked prefill + mm/hash): ``num_tokens`` =
the batch's (this chunk batch's) flat token count, ``aux_rows`` = the sum
of token rows of **mrope-carrying** requests in the batch — the UP main
frame numel is computed **worker-side** from the hash payload formula
using local config (see
``lwd_cloud_worker._lwd_build_up_recv_request``); the hint carries only
"facts" (token count, mrope row count), never the size formula, so the
two sides cannot compute it differently; aux frame numel = aux_rows*3
(int64, [n,3], same seqno as the main frame).

Non-endpoint ranks (the cloud's remaining TP ranks) hold no cross-node
channel: their data arrives via the in-group TP broadcast at inject time
(see ``lwd_cloud_model_runner._lwd_inject_remote_embeds``) — they post no
recv and report nothing; this module's hint/report line runs on the
endpoint rank only.
"""

from __future__ import annotations

import base64
import os
import pickle
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

#: Env var carrying the downstream hint MQ handle. The executor exports it
#: before creating the worker; the worker rebuilds the reader in
#: ``init_device`` — both sides honor this single constant name.
LWD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_LWD_RECV_HINT_HANDLE"

# Bypass MQ spec: 1KB per slot, 64 slots. One hint/report per batch; the
# edge pending gate (``LwdEdgeScheduler._LWD_MAX_PENDING_CHUNKS = 3``)
# keeps unconsumed chunks in the single digits, and hints are a
# correctness dependency (losing one = that recv is never pre-posted, the
# watermark stalls, and the gated batch is never dispatched) — so plenty
# of headroom, no flow control.
_SIDEBAND_MAX_CHUNK_BYTES = 1024
_SIDEBAND_MAX_CHUNKS = 64


# ---------------------------------------------------------------------- #
# Hints: produced by the scheduler, consumed by the worker comm thread   #
# ---------------------------------------------------------------------- #
def make_recv_hint(seqno: int, num_tokens: int, aux_rows: int = 0) -> dict:
    """Build one UP early-recv hint.

    * ``num_tokens``: the batch's flat token count (the main-frame size
      is derived worker-side from the hash payload formula)
    * ``aux_rows``: the batch's mrope-carrying token rows (0 = text-only
      batch, zero aux traffic)
    """
    return {
        "channel": LwdChannelType.UP,
        "seqno": int(seqno),
        "num_tokens": int(num_tokens),
        "aux_rows": int(aux_rows),
    }


def hint_fields(hint: dict) -> tuple[int, int, int] | None:
    """Hint self-check: returns (seqno, num_tokens, aux_rows); None for a malformed hint."""
    try:
        seqno = int(hint["seqno"])
        num_tokens = int(hint["num_tokens"])
        aux_rows = int(hint.get("aux_rows", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if seqno < 0 or num_tokens <= 0 or aux_rows < 0 or aux_rows > num_tokens:
        return None
    return seqno, num_tokens, aux_rows


# ---------------------------------------------------------------------- #
# Bypass MQ lifecycle                                                    #
# ---------------------------------------------------------------------- #
def _new_sideband_mq() -> MessageQueue:
    """A 1-reader 1-writer bypass MQ (hint and report share the spec; only the endpoint rank participates)."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue(
        1,
        1,
        max_chunk_bytes=_SIDEBAND_MAX_CHUNK_BYTES,
        max_chunks=_SIDEBAND_MAX_CHUNKS,
    )


def create_hint_mq() -> MessageQueue:
    """Engine side: create the hint MQ (writer) and export the handle to
    the environment.

    Must be called before workers are created (they rebuild the reader
    from the env at startup); both fork and spawn inherit via env vars.
    """
    mq = _new_sideband_mq()
    os.environ[LWD_RECV_HINT_MQ_ENV] = base64.b64encode(pickle.dumps(mq.export_handle())).decode("ascii")
    return mq


def attach_hint_mq() -> MessageQueue | None:
    """Worker side: rebuild the hint MQ reader from the handle in the
    environment.

    Returns None when no handle exists (this role takes no hints): the
    comm thread still runs reporting, so a missing hint only delays the
    early recv, never stalls the watermark.
    """
    raw = os.environ.get(LWD_RECV_HINT_MQ_ENV)
    if raw is None:
        return None
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue.create_from_handle(pickle.loads(base64.b64decode(raw)), 0)


def create_report_mq() -> MessageQueue:
    """Worker side: create the done MQ (writer); the engine fetches its
    handle over ``collective_rpc`` once the executor is up."""
    return _new_sideband_mq()


# ---------------------------------------------------------------------- #
# Scheduler process side: hint dispatch + completion watermark           #
# ---------------------------------------------------------------------- #
_hint_sender: Callable[[dict], None] | None = None
_watermarks: dict[LwdChannelType, int] = {}
_lock = threading.Lock()


def register_hint_sender(sender: Callable[[dict], None]) -> None:
    """Install the transport used to dispatch hints (once per engine
    process).

    The closure passed in must **deliver reliably** (enqueue the hint
    into the hint MQ): with the gate on, hints are a correctness
    dependency — lose one and that recv is never pre-posted, the
    watermark stalls, and the gated batch is never dispatched.
    """
    global _hint_sender
    _hint_sender = sender


def post_irecv_hint(hint: dict) -> None:
    """Send one recv hint to the local worker's comm thread.

    The healthy path logs nothing (the caller ``lwd_cloud_engine`` logs
    ``HINT ...``): this module keeps only should-never-happen diagnostics,
    and they must be visible in the logs.
    """
    if _hint_sender is None:
        # The engine registers the sender during startup, before it accepts
        # any edge message, so a missing one is a wiring error; dropping the
        # hint would stall the pre-recv gate forever (nobody would pre-post
        # that recv, the watermark would never advance).
        raise RuntimeError(f"[LWD] early-recv hint for seqno {hint.get('seqno')} has no registered sender")
    _hint_sender(hint)


def record_irecv_completions(
    items: Iterable[tuple[LwdChannelType, int]],
) -> list[tuple[LwdChannelType, int]]:
    """Merge worker-reported completions into the per-channel watermark
    (called after the engine drains the done MQ each step).

    Returns the entries that **actually advanced** the watermark this
    call, for the caller (``lwd_cloud_engine``) to log: whether the
    watermark is moving is the first-hand evidence for locating which
    segment is stuck, so the log lives in a module proven to emit.
    """
    advanced: list[tuple[LwdChannelType, int]] = []
    with _lock:
        for channel, seqno in items:
            if seqno > _watermarks.get(channel, -1):
                _watermarks[channel] = seqno
                advanced.append((channel, seqno))
    return advanced


def watermark(channel: LwdChannelType) -> int:
    """Current watermark: the max confirmed-complete seqno on the channel (-1 = none yet)."""
    with _lock:
        return _watermarks.get(channel, -1)


def is_irecv_complete(channel: LwdChannelType, seqno: int) -> bool:
    """Whether the seqno recv has completed on the channel — a pure
    readiness predicate, no logging.

    "Who is being waited on" comes from the gate's own log line
    (``lwd_cloud_mixed_scheduler`` logs ``gate seqno=.. watermark=..
    ready=..`` every step), so this stays a pure predicate.
    """
    return seqno <= watermark(channel)
