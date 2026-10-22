# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD (edge-cloud) worker subclass.

All LWD worker-side logic lives here (moved out of ``worker.py``):
init bring-up of the duplex channels + recv managers, the EDGE_EMBED
dispatch hook, the cloud finished-flush, and the
DOWN send (draining the runner's per-step hidden payload + attaching
``lwd_c2e_meta`` to the ModelRunnerOutput).

Selected via ``parallel_config.worker_cls`` (see platform.py:
``vllm_ascend.worker.lwd_cloud_worker.LwdCloudWorker``) when
``lwd_config`` is set (LWD deployment).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

# ops must initialize before the model_runner chain, otherwise device_op
# and the ops package circularly import (mirrors worker.py's import order)
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import lwd_get_data_plane
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdTransferRequest
from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
    LwdCloudModelRunner,
)
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel

if TYPE_CHECKING:
    from typing import Any

    from vllm.distributed.device_communicators.shm_broadcast import Handle
    from vllm.v1.core.sched.output import (
        GrammarOutput,
        LwdEmbedBatch,
        SchedulerOutput,
    )

# Early-recv thread loop tuning: MQ dequeue timeout, backoff sleep after a
# dequeue error, and the per-iteration sleep that caps CPU while the MQ
# reader window spins.
_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S = 0.0005
_LWD_EARLY_RECV_ERROR_SLEEP_S = 0.01
_LWD_EARLY_RECV_LOOP_SLEEP_S = 0.002


class LwdCloudWorker(LwdWorkerBase):
    """LwdWorkerBase + LWD data-plane wiring (cloud side only).

    Selected by ``platform.py`` only when LWD is enabled on a cloud
    process; the data-plane wiring below runs unconditionally."""

    def init_device(self) -> None:
        super().init_device()
        # channel-global DOWN seqno counter (worker layer, send-time alloc)
        self._lwd_down_next_seqno = 0
        # req_id -> posted UP recv futures (consumed by the runner's device-side inject)
        self._lwd_up_recv_futures: dict[int, tuple[LwdTransferFuture | None, LwdEmbedBatch]] = {}
        # [Lwd][pre-recv] Early receive: only the endpoint rank (= the only
        # cloud rank holding a cross-node P2P channel, same convention as
        # the data-plane channels) pre-posts the real irecv and reports completion
        # watermarks; every other rank gets its data via the in-group TP
        # broadcast at inject time — no recv posted, nothing reported.
        self._init_lwd_early_recv_state()
        if self.use_v2_model_runner:
            # The LWD hooks live on the V1 model runner; the V2 runner
            # (separate class) lacks them entirely — fail fast at bring-up
            # instead of AttributeError at first request.
            raise RuntimeError("LWD data plane requires the V1 model runner; use_v2_model_runner is not supported")
        # Swap in the LWD model runner BEFORE any model load/usage.
        # The cloud feeds on edge-computed prompt embeddings (no token
        # ids on its input), so the native prompt-embeds path must be
        # enabled for the runner to allocate inputs_embeds buffers.
        self.model_config.enable_prompt_embeds = True
        self.model_runner = LwdCloudModelRunner(self.vllm_config, self.device, worker=self)
        self._init_lwd_data_plane()
        # Only the endpoint rank (the cloud leader — first rank of the
        # first cloud DP in the topology) has a cross-node channel: start
        # its early-recv comm thread. The topology ranks are GLOBAL, while
        # self.rank is the executor's per-node numbering — the edge card
        # count offsets them (seen live: local rank 1 of the cloud was
        # picked as "endpoint", its comm thread then died on "rank 2 holds
        # no lwd_up channel"). Compare the distributed global rank.
        import torch.distributed as dist

        global_rank = dist.get_rank()
        self._lwd_early_recv_endpoint = global_rank == self.vllm_config.lwd_config.topology.clouds[0].dp[0].ranks[0]
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        # One log line per cloud rank: which ranks own the comm thread is obvious at a glance.
        lwd_logger.info(
            "[Lwd][pre-recv] worker local_rank=%d global_rank=%d endpoint=%s hidden=%d",
            self.rank,
            global_rank,
            self._lwd_early_recv_endpoint,
            self._lwd_up_hidden_size,
        )
        if self._lwd_early_recv_endpoint:
            self._start_lwd_early_recv_thread()
        self._register_lwd_prompt_embeds_provider()

    def load_model(self) -> None:
        """Log the real layer split after loading: layer count / first-last
        layer names / pp slicing args, so a half-model suspicion is decided
        at startup (full coverage from layer 0 = normal; halved or nonzero
        start = wrong pp slicing)."""
        super().load_model()
        # [layer-trace] Parity instrumentation (env-gated, zero overhead
        # when off): the cloud install point; edge and centralized share
        # the same code (see lwd_debug.lwd_layer_trace).
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.model_runner.get_model())
        model = self.model_runner.get_model()
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None) or getattr(backbone, "decoder_layers", None)
        pc = self.vllm_config.parallel_config
        layer_names = [
            n for n, _ in model.named_modules() if n.count("layers.") == 1 and n.endswith(tuple("0123456789"))
        ]
        lwd_logger.info(
            "[Lwd][cloud-model] loaded layers=%s first=%s last=%s "
            "(config pp=%d tp=%d, my rank=%d) — full load should cover every layer from 0",
            len(layers) if layers is not None else "?",
            layer_names[0] if layer_names else "?",
            layer_names[-1] if layer_names else "?",
            pc.pipeline_parallel_size,
            pc.tensor_parallel_size,
            self.rank,
        )

    # ------------------------------------------------------------------ #
    # Engine step wiring                                                  #
    # ------------------------------------------------------------------ #

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # [Lwd][perf] Worker start time (subtract the engine's rpc-enqueue
        # ts = dispatch-to-start latency; spread across ranks = collective
        # alignment cost)
        lwd_logger.info(
            "[Lwd][perf] worker-exec-start rank=%d ts=%.3f",
            self.rank,
            time.monotonic(),
        )
        # cloud: post the exact-size UP irecv for every incoming
        # LWD_EMBED batch (control info rides scheduler_output.lwd_batch).
        self._lwd_up_post_recvs(scheduler_output)

        # cloud: finished requests trigger local cleanup (collector
        # bookkeeping + UP chunk table); finished_req_ids is the
        # engine's own liveness signal.
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)

        lwd_get_data_plane().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] Per-step exec_model timing on the cloud: total = the
        # whole forward pass (sample/collect are itemized separately in
        # sample_tokens)
        lwd_logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return output

    def _lwd_up_post_recvs(self, scheduler_output: SchedulerOutput) -> None:
        """Bind the UP recv for an incoming LWD_EMBED batch.

        The UP two-phase communication splits into two instants per the
        demo_br pattern:
          * post time (this function): only the endpoint rank posts the
            channel P2P recv; non-endpoints just register the meta;
          * consume time (the runner's inject side): every rank launches
            the in-group TP broadcast *on its own compute stream* followed
            immediately by work.wait() — under torch_npu, cross-stream
            ordering between two HCCL ops cannot be trusted (broadcasts
            launched at post time were measured racing ahead of the irecv
            and delivering truncated data); a launch+wait pair in the same
            stream context is the demo-verified reliable shape.

        Early recv: the endpoint's irecv has already been pre-posted by the
        comm thread on the engine's hint (posted as soon as the RangeNotify
        arrives, not waiting for this SO); this function only takes that
        future. The gate (``_lwd_pre_recv_ready``) guarantees it is posted
        before release. Should a hint never arrive, the same information
        re-posts it here (see ``get_or_post_early_recv``) — no missed recv.

        All control info rides the SchedulerOutput (edge -> cloud
        control plane fills it in): batch.seqno is the edge-side dispatch
        number (the cross-node pairing key); the recv size covers the
        embeddings plus optional Hash expert IDs."""
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None or not meta.req_ids:
            return
        num_tokens = sum(len(t) for t in meta.token_ids)
        if num_tokens <= 0:
            return
        # mrope positions moved to the control plane
        # (LwdRangeItem.mrope_positions arrives with the meta,
        # wire_version>=2); the UP recv posts only the main embeds frame
        # and aux_rows is always 0 — a stale nonzero would leave the
        # pre-posted aux irecv forever unmatched and stall the watermark.
        # The aux frame mechanism stays in the channel layer for deepstack
        # and other future payloads.
        if self._lwd_early_recv_endpoint:
            future = self.get_or_post_early_recv(batch.seqno, num_tokens, 0)
        else:
            # Non-endpoint: never touches the cross-node channel; the
            # broadcast receive is deferred to inject time. None marks
            # "broadcast on consumption".
            future = None
        self._lwd_up_recv_futures[batch.seqno] = (future, meta)
        if future is None:
            # Seven INFO lines per batch is pure noise; demoted to DEBUG
            # (with VLLM_LOGGING_LEVEL=DEBUG the per-rank forward entry
            # times/spread are still visible)
            lwd_logger.debug(
                "[Lwd][pre-recv] POST seqno=%d reqs=%d tokens=%d endpoint=False ts=%.3f",
                batch.seqno,
                len(meta.req_ids),
                num_tokens,
                time.monotonic(),
            )
            return
        # One memory reading per batch (only the endpoint rank holds recv
        # buffers): a climbing peak = something is leaking; mem_free
        # approaching 0 = OOM precursor.
        free_bytes, _total = torch.npu.mem_get_info()
        lwd_logger.info(
            "[Lwd][pre-recv] POST seqno=%d reqs=%d tokens=%d "
            "endpoint=True ts=%.3f mem_alloc=%.2fGiB mem_peak=%.2fGiB "
            "mem_free=%.2fGiB",
            batch.seqno,
            len(meta.req_ids),
            num_tokens,
            time.monotonic(),
            torch.npu.memory_allocated() / 2**30,
            torch.npu.max_memory_allocated() / 2**30,
            free_bytes / 2**30,
        )

    # ------------------------------------------------------------------ #
    # Early receive: comm thread (hint -> post recv -> report watermark) #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """The early-recv ledger (same keys as the engine-issued hints and
        the reported watermarks).

        Key = (channel, seqno). Entry lifecycle: a hint submits the recv
        and registers the entry; the consume point
        ``get_or_post_early_recv`` only *takes and marks consumed* (no
        pop: the reporter thread keeps polling it to advance the
        watermark); removal happens only once both reported and consumed.

        **In-flight cap = 3**: each posted recv holds a whole-batch-sized
        receive buffer (aux frames included), and this 3 shares its source
        with the edge-side pending gate — ``LwdEdgeScheduler`` caps edge
        lead by unconsumed chunks at ``_LWD_MAX_PENDING_CHUNKS = 3``, so
        the cloud never needs more than those 3. Hints beyond the cap
        queue first (``_early_recv_pending``, arrival-ordered, fields only
        — no device memory) and post as soon as a slot frees when an
        earlier batch is consumed with its SO. The slot must be returned
        on "consumed with the SO" (= dispatched), NOT on "received":
        the gate releases batch N once the watermark >= N, and
        execute_model immediately takes that future; if slots were
        returned only on completion, batch N+1's recv might not be posted
        yet and the gate would run ahead of the recv.
        """
        self._early_recv_max_inflight = 3
        self._early_recv_futures: dict[tuple[LwdChannelType, int], LwdTransferFuture] = {}
        self._early_recv_lock = threading.Lock()
        # Keys already taken by the consume point: prevents the comm thread from posting a duplicate recv on a late hint
        self._early_recv_consumed: set[tuple[LwdChannelType, int]] = set()
        # Keys whose completion has been reported
        self._early_recv_reported: set[tuple[LwdChannelType, int]] = set()
        # Hints queued while the in-flight cap is full (key -> hint, insertion order = arrival order)
        self._early_recv_pending: dict[tuple[LwdChannelType, int], dict[str, Any]] = {}
        # Rate-limit timestamp for the stuck-queue diagnostic
        self._early_recv_stuck_log_at = 0.0
        # Downstream hint MQ (reader) / upstream completion MQ (writer); the
        # engine attaches the done reader through the handle returned by
        # lwd_early_recv_done_handle (fetched over collective_rpc).
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        self._lwd_early_recv_thread: threading.Thread | None = None

    def _lwd_build_up_recv_request(self, seqno: int, num_tokens: int, aux_rows: int) -> LwdTransferRequest:
        """The single construction point for UP recv requests (the hint path
        and the consume-point re-post share the same shape).

        HCCL P2P requires both ends to match numel exactly: the main frame
        = the hash payload formula (embeds + optional expert-ID bytes).
        The aux frame mechanism is kept for deepstack and other future
        payloads (mrope positions moved to the control plane in
        wire_version 2; callers always pass aux_rows=0). The size formula
        is computed here exactly once so the two paths cannot drift.
        """
        return LwdTransferRequest(
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=hash_payload_numel(
                num_tokens,
                self._lwd_up_hidden_size,
                hash_layer_count(self.model_config.hf_config),
                getattr(self.model_config.hf_config, "num_experts_per_tok", 0),
            ),
            seqno=int(seqno),
            # The aux second payload shares the main frame's seqno; always
            # 0 since mrope retired (zero aux traffic); future payloads
            # (deepstack etc.) just pass their row count
            aux_num_elements=int(aux_rows) * 3,
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """Rebuild the hint MQ reader, create the done MQ writer, and start
        the early-recv comm thread.

        One log line before and after the thread start: "starting" before
        (proof this code ran) and "up" from inside the thread body after
        setting the device (proof the thread is alive). Only the first
        line = the thread died during device setup (traceback on stderr);
        both lines but no SUBMIT = no hints arrived (hint MQ not attached;
        the log will say attached/none).
        """
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        self._irecv_hint_mq = attach_hint_mq()
        self._irecv_done_mq = create_report_mq()
        # The engine fetches this MQ's handle through
        # lwd_early_recv_done_handle over collective_rpc once the executor
        # is up, so the vllm executor stays free of LWD plumbing.
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def lwd_early_recv_done_handle(self) -> Handle | None:
        """Handle of the early-recv done MQ, for the engine to attach its
        reader; None on ranks that do not report completions (the
        non-endpoint ranks hold no cross-node channel)."""
        if self._irecv_done_mq is None:
            return None
        return self._irecv_done_mq.export_handle()

    def _lwd_early_recv_loop(self) -> None:
        """Comm thread body: take hints -> pre-post recvs; poll futures ->
        report completions.

        It only does four featherweight things: ``dequeue`` /
        ``submit_recv`` / ``future.done()`` / ``enqueue`` — it never
        ``wait()``s any future and never touches the model, so it cannot
        conflict with the busy loop's HCCL usage (an irecv on one side of
        a channel plus a cross-thread wait on the other is disallowed by
        HCCL).
        """
        # The current device is thread-local state; the comm thread must
        # set it once itself (or the irecv lands on card 0)
        torch.npu.set_device(self.device)
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQ=%s",
            self.rank,
            self.device,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        hint_mq = self._irecv_hint_mq
        while True:
            if hint_mq is not None:
                try:
                    hint = hint_mq.dequeue(timeout=_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S)
                except TimeoutError:
                    pass
                except Exception:
                    # Ring cancelled at shutdown etc.: log once and sleep a
                    # little; never hot-spin (daemon threads exit with the
                    # process)
                    lwd_logger.exception("[Lwd][pre-recv] hint MQ dequeue failed")
                    time.sleep(_LWD_EARLY_RECV_ERROR_SLEEP_S)
                else:
                    self.start_early_irecv(hint)
            self._report_irecv_completions()
            self._warn_if_early_recv_stuck()
            # The ring's read side sched_yield-spins within the
            # just-read window (busy_loop_s=1s), so a 0.5ms timeout alone
            # does not cap the CPU — this sleep does; the cost is ~2ms of
            # added visibility latency on hints and watermarks each.
            time.sleep(_LWD_EARLY_RECV_LOOP_SLEEP_S)

    _LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S = 5.0

    def _warn_if_early_recv_stuck(self) -> None:
        """Warn every 5s while the queue cannot drain: hints are queued but
        every in-flight slot is held.

        Held slots = posted recvs neither consumed nor released (so the
        watermark stops advancing and the gate stops releasing prefills).
        The log names exactly which seqnos hold the slots — no more
        "prefill stalled and nobody knows why".
        """
        with self._early_recv_lock:
            if not self._early_recv_pending or len(self._early_recv_futures) < self._early_recv_max_inflight:
                return
            now = time.monotonic()
            if now - self._early_recv_stuck_log_at < self._LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S:
                return
            self._early_recv_stuck_log_at = now
            oldest = next(iter(self._early_recv_pending))[1]
            holders = sorted(key[1] for key in self._early_recv_futures)
        lwd_logger.warning(
            "[Lwd][pre-recv] queue stuck: oldest queued seqno=%d, %d slot(s) "
            "held by seqno=%s (slots held = those batches posted recv but "
            "were never consumed; downstream dispatch is the pacer)",
            oldest,
            len(holders),
            holders,
        )

    def start_early_irecv(self, hint: dict[str, Any]) -> None:
        """Take one engine hint; post the recv immediately while slots
        remain, otherwise queue it.

        Idempotent: duplicate hints for the same (channel, seqno), or ones
        the consume point already handled, are ignored — two irecvs on one
        channel would race the sender's single isend.
        """
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        fields = hint_fields(hint)
        if fields is None:
            lwd_logger.warning("[Lwd][pre-recv] malformed hint %s, skipped", hint)
            return
        seqno, _num_tokens, _aux_rows = fields
        key = (hint["channel"], seqno)
        with self._early_recv_lock:
            if key in self._early_recv_futures or key in self._early_recv_consumed or key in self._early_recv_pending:
                return
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _early_recv_pump(self) -> None:
        """Post queued hints as recvs in arrival order until the in-flight
        cap is full.

        Slots = entries whose recv is posted but not yet consumed by an SO
        (each holds a whole-batch receive buffer), so this cap is a memory
        cap; ``get_or_post_early_recv`` calls this after consuming one to
        post the next, so the hint queue never drops anything.
        """
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        while True:
            with self._early_recv_lock:
                if not self._early_recv_pending or len(self._early_recv_futures) >= self._early_recv_max_inflight:
                    return
                key, hint = next(iter(self._early_recv_pending.items()))
                del self._early_recv_pending[key]
                seqno, num_tokens, aux_rows = hint_fields(hint)
                future = lwd_get_data_plane().submit_recv(self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows))
                self._early_recv_futures[key] = future
                inflight = len(self._early_recv_futures)
                queued = len(self._early_recv_pending)
            lwd_logger.info(
                "[Lwd][pre-recv] SUBMIT channel=%s seqno=%d num_tokens=%d aux_rows=%d inflight=%d queued=%d ts=%.3f",
                key[0].value,
                seqno,
                num_tokens,
                aux_rows,
                inflight,
                queued,
                time.monotonic(),
            )

    def _report_irecv_completions(self) -> None:
        """Report completed recvs as (channel, seqno) tuples to the done MQ.

        Per-channel completion order = seqno order, so one max watermark
        per channel decides readiness exactly. Entries already taken by
        the consume point keep being polled: consumption means the forward
        is using the data, so ``done()`` flips soon and the watermark
        still advances; removal only once both reported and consumed.
        """
        done_mq = self._irecv_done_mq
        if done_mq is None:
            return
        with self._early_recv_lock:
            pending = [
                (key, future)
                for key, future in self._early_recv_futures.items()
                if key not in self._early_recv_reported
            ]
        for key, future in pending:
            if not future.done():
                continue
            # ① Completed: this recv's data (main frame + aux frame) has fully landed
            lwd_logger.info(
                "[Lwd][pre-recv] DONE channel=%s seqno=%d ts=%.3f",
                key[0].value,
                key[1],
                time.monotonic(),
            )
            # ② Report: write (channel, seqno) into the done MQ; the engine drains it each step to advance the watermark
            done_mq.enqueue(key)
            lwd_logger.info(
                "[Lwd][pre-recv] REPORT channel=%s seqno=%d ts=%.3f",
                key[0].value,
                key[1],
                time.monotonic(),
            )
            with self._early_recv_lock:
                self._early_recv_reported.add(key)
                if key in self._early_recv_consumed:
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)
                    released = True
                else:
                    released = False
            if released:
                self._early_recv_pump()

    def get_or_post_early_recv(self, seqno: int, num_tokens: int, aux_rows: int) -> LwdTransferFuture | None:
        """Take the comm thread's pre-posted UP recv future; if absent,
        post one on the spot and register it.

        Non-endpoint ranks hold no cross-node channel (their data arrives
        via the TP broadcast at inject time) and get None. Entries are not
        removed here: the reporter thread keeps polling them to advance
        the watermark; removal only once both reported and consumed, so
        the watermark never gaps on consumption.

        This is the **slot-return point** (same role as the old
        `note_scheduled`): once the SO arrives, this entry's buffer goes
        straight to the forward, and the slot passes to a queued hint
        immediately (a still-queued hint is posted in place here — the
        gate/forward can never overtake the recv).
        """
        if not self._lwd_early_recv_endpoint:
            return None

        key = (LwdChannelType.UP, seqno)
        with self._early_recv_lock:
            self._early_recv_consumed.add(key)
            future = self._early_recv_futures.get(key)
            if future is None:
                # The hint was still queued while slots were full: this batch is dispatched, so post it now
                hint = self._early_recv_pending.pop(key, None)
                if hint is not None:
                    future = lwd_get_data_plane().submit_recv(
                        self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows)
                    )
                    self._early_recv_futures[key] = future
            if future is not None:
                if key in self._early_recv_reported:
                    # Completion already reported: consumption is the entry's end (the reporter will not touch it again)
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)
            else:
                # The hint truly never arrived (or lands later than this
                # SO): re-post from the same information so the report chain
                # has no gaps and the watermark can keep advancing
                lwd_logger.info("[Lwd][pre-recv] submit at consume seqno=%d", seqno)
                future = lwd_get_data_plane().submit_recv(self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows))
                self._early_recv_futures[key] = future
            inflight = len(self._early_recv_futures)
            queued = len(self._early_recv_pending)
        # Slot returned: post the next queued hint
        self._early_recv_pump()
        lwd_logger.info(
            "[Lwd][pre-recv] stats seqno=%d inflight=%d queued=%d",
            seqno,
            inflight,
            queued,
        )
        return future

    # Note: take_lwd_up_embeds (host-blocking harvest) is not revived with
    # the port — UP consumption lives in the runner on this line
    # (_lwd_inject_remote_embeds: endpoint gate + in-place TP broadcast);
    # the worker layer keeps only the post side. aux (mrope) frames are
    # consumed on the same path.

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN: send the hidden packet (channel-async,
        # in-stream ordered); the step meta stays on device until the
        # fenced harvest below brings it to the host with a device-side
        # stream dependency (dedicated stream wait_event on the ready
        # event, completion event recorded on the copy stream itself).
        payload = self.model_runner.take_lwd_pending_down_payload()
        if payload is not None and output is not None:
            (hidden, meta_dev, seg_lens_host, ready_event, req_ids) = payload
            seqno = self._lwd_next_down_seqno()
            lwd_logger.info(
                "[Lwd][cloud-worker] DOWN send seqno=%d numel=%d",
                seqno,
                hidden.numel(),
            )
            lwd_get_data_plane().submit_send(
                LwdTransferRequest(
                    channel=LwdChannelType.DOWN,
                    op="send",
                    num_elements=hidden.numel(),
                    tensor=hidden,
                    seqno=seqno,
                )
            )
            # Fenced harvest: device-side dependency guarantees the copy
            # starts only after the main-stream work that produced
            # meta_dev; done.synchronize() waits for this copy only
            # (microseconds), not the whole GPU. This replaces the retired
            # pinned D2H + shared-event scheme, which could not fence the
            # side-stream copy on this platform (the engine could read a
            # previous generation's meta image — wrong accepted counts).
            _t_harvest = time.monotonic()
            stream = self._lwd_get_harvest_stream()
            n_dev = meta_dev.numel()
            n_seg = len(seg_lens_host) if seg_lens_host is not None else 0
            buf = self._lwd_get_harvest_pinned(max(n_dev + n_seg, 4096))
            with torch.npu.stream(stream):
                stream.wait_event(ready_event)
                buf[:n_dev].copy_(meta_dev, non_blocking=True)
                done = torch.npu.Event()
                done.record()
            done.synchronize()
            if n_seg:
                buf.numpy()[n_dev : n_dev + n_seg] = seg_lens_host
            # Attach the finished 4-tuple (wire layout [meta|seg_lens]
            # unchanged); the engine parse path needs no changes.
            target = getattr(output, "_model_runner_output", output)
            target.lwd_down_carrier = (
                buf[: n_dev + n_seg],
                req_ids,
                hidden.numel(),
                seqno,
            )
            # [Lwd][perf] Per-step cloud timing: sample = sampling +
            # collection (rank computation included); send = the DOWN
            # submit (snapshot clone + isend + bridge wait); harvest =
            # the fenced meta copy; total = the whole step
            lwd_logger.info(
                "[Lwd][perf] cloud-step seqno=%d sample=%.2f send=%.2f harvest=%.2f total=%.2fms rows=%d",
                seqno,
                (_t_sample - _t0) * 1000,
                (time.monotonic() - _t_sample) * 1000,
                (time.monotonic() - _t_harvest) * 1000,
                (time.monotonic() - _t0) * 1000,
                hidden.shape[0] if hidden is not None else 0,
            )
        return output

    def _lwd_get_harvest_stream(self) -> torch.npu.Stream:
        """Lazily create the dedicated harvest stream (one per worker)."""
        if not hasattr(self, "_lwd_harvest_stream"):
            self._lwd_harvest_stream = torch.npu.Stream()
        return self._lwd_harvest_stream

    def _lwd_get_harvest_pinned(self, min_capacity: int) -> torch.Tensor:
        """Lazily allocate (or grow) the pinned harvest buffer."""
        buf = getattr(self, "_lwd_harvest_pinned", None)
        if buf is None or buf.numel() < min_capacity:
            buf = self._lwd_harvest_pinned = torch.empty(
                min_capacity,
                dtype=torch.int32,
                pin_memory=True,
            )
        return buf

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def _lwd_cloud_flush_finished(self, finished_req_ids: list[str]) -> None:
        """Cloud side (streaming): a finished request's DOWN stream simply
        STOPS (its last step packet already went out with that step).
        No FIN packet -- request termination is signaled by the control
        plane (v2.6).  Bookkeeping release lives on the runner
        (``lwd_release_finished``, the single owner of the release path);
        finished_req_ids is the engine's own liveness signal, so this
        cleanup does not depend on the control plane."""
        lwd_logger.info("[Lwd][cloud-worker] flush finished reqs=%s", list(finished_req_ids))
        self.model_runner.lwd_release_finished(finished_req_ids)

    def _register_lwd_prompt_embeds_provider(self) -> None:
        """Wire the draft proposer's first-pass prompt-embeds provider.

        The provider resolves the request's prompt embeds from the
        runner's ``input_batch.req_prompt_embeds`` (already injected by
        ``_lwd_inject_remote_embeds`` during prefill); a missing entry
        (not scheduled / not LWD) returns None and the proposer falls
        back to the token-id path.
        """
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        runner = self.model_runner

        def _lwd_prompt_embeds_provider(req_id: str) -> torch.Tensor | None:
            idx = runner.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                return None
            return runner.input_batch.req_prompt_embeds.get(idx)

        AscendSpecDecodeBaseProposer.set_lwd_prompt_embeds_provider(_lwd_prompt_embeds_provider)

    def _lwd_next_down_seqno(self) -> int:
        seqno = self._lwd_down_next_seqno
        self._lwd_down_next_seqno += 1
        return seqno
