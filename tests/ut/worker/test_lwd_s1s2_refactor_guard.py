# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""S1/S2 refactor guards (pure source assertions; no torch, must pass in
any environment).

Anchors the key equivalence and structural commitments of
S1/S2 in docs/refactor/lwd_worker_packages_refactor_plan.md against
regressions:
- R1: the dead compute_top_id_th stays deleted;
- R2: the three hand-rolled pinned rings stay consolidated into
  LwdPinnedStaging (slot-index variables stay zeroed);
- R3: the endpoint predicate stays single-sourced (the runner never
  recomputes edge_npu_count);
- F6a/R4: the harvest stays split per-seqno (_lwd_stage_up_batch ->
  _lwd_harvest_tensors_one -> _lwd_check_up_windows) with an event order
  matching the pre-split single loop point for point, wholly before
  super()._prepare_inputs (the boundary of deferred item D1);
- R4: the inject reuses the batch positions carried by the harvest
  instead of re-querying req_id_to_index;
- the three fail-fast messages (the silent-corruption defense) stay
  verbatim;
- F6b: after the DOWN-collect split, the runner records the event via
  worker.note_lwd_meta_ready instead of writing the worker's private
  attribute;
- R9: finished-request cleanup stays single-owner
  (runner.lwd_release_finished is the only sink; the worker only
  forwards).

Known benign delta of the timing split (correspondence): the meta-stage
check precedes the tensor-stage host gate, so on failure paths the
exception message is verbatim-identical and only its position relative
to the wait differs; on success paths the device order is unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = (
    _REPO_ROOT
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_model_runner.py"
)
_WORKER = (
    _REPO_ROOT
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_worker.py"
)
_EDGE = (
    _REPO_ROOT / "vllm_ascend" / "worker" / "lwd_edge_model" / "lwd_edge_worker.py"
)


def _method(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found in {path.name}")


def _segment(path: Path, name: str) -> str:
    fn = _method(path, name)
    seg = ast.get_source_segment(path.read_text(encoding="utf-8"), fn)
    assert seg is not None
    return seg


def _call_names(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    """(lineno, called name) for attribute calls, in source order."""
    calls = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append((node.lineno, node.func.attr))
    return sorted(calls)


def test_dead_code_removed() -> None:
    src = _EDGE.read_text(encoding="utf-8")
    assert "compute_top_id_th" not in src, "dead helper must not come back"


def test_no_is_lwd_gates_inside_lwd_runner() -> None:
    """Team norm: LWD classes/functions never re-check "is this LWD".

    Entry invariant: platform.py selects LwdCloudWorker only when
    lwd_config.enabled, and LwdCloudModelRunner's sole construction site
    is its init_device — any enabled / is-not-None gate inside the class
    is an invariantly-true dead check (mode-dimension branches are not
    "is this LWD" and are allowed).
    """
    src = _RUNNER.read_text(encoding="utf-8")
    assert "_lwd_enabled" not in src
    assert "lwd_config.enabled" not in src
    assert 'getattr(self.vllm_config, "lwd_config", None)' not in src


def test_draft_cache_weave_removed() -> None:
    """Guard for the LwdDraftCache removal (same argument as the base
    spec-recovery-chain removal).

    The restore path has zero consumers on the current pdmix line:
    decode requests are never evicted (whole prefill notifies defer),
    the native max_tokens guard is depart-and-finish (the request
    necessarily finishes; no returning step), and the V2 cadence guard
    is dead on the cloud (V1 enforced). With the removal, the cloud
    runner's speculative path is fully native (_update_states /
    _prepare_input_ids stay unoverridden). If the phase arm
    (phase_scheduler) is ever rewired, a departure-safe draft keeper
    must return with it.
    """
    runner_src = _RUNNER.read_text(encoding="utf-8")
    for symbol in ("LwdDraftCache", "_lwd_draft_cache", "_lwd_restore_drafts"):
        assert symbol not in runner_src, f"draft-cache weave leaked back: {symbol}"
    # _prepare_input_ids must stay unoverridden (the draft-refill chain
    # was removed wholesale); _update_states now carries the
    # zero-weave placeholder rebuild wrap (prompt_embeds seam), a
    # legitimate override whose shape is locked by
    # test_lwd_prompt_embeds_seam.
    assert "def _prepare_input_ids(" not in runner_src
    assert not (
        _REPO_ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_draft_cache.py"
    ).exists(), "deleted module must not come back"


def test_pinned_staging_single_implementation() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    edge = _EDGE.read_text(encoding="utf-8")
    base = (_REPO_ROOT / "vllm_ascend/worker/lwd_base_model/lwd_base_model_runner.py").read_text(encoding="utf-8")
    assert base.count("LwdPinnedStaging(") == 1  # the named lazy factory is the sole constructor
    assert runner.count("LwdPinnedStaging(") == 0  # the cloud delegates via _lwd_staging
    assert runner.count("self._lwd_staging(") == 2  # down_meta + down_index
    assert edge.count("LwdPinnedStaging(") == 0  # the edge worker is pure dispatch
    edge_runner = (_REPO_ROOT / "vllm_ascend/worker/lwd_edge_model/lwd_edge_model_runner.py").read_text(encoding="utf-8")
    assert edge_runner.count("self._lwd_staging(") == 1  # ids staging goes through the factory
    for stale in ("_lwd_pinned_ring_idx", "_lwd_h2d_ring_idx"):
        assert stale not in runner, f"hand-rolled rotation leaked: {stale}"
    assert "_lwd_ids_ring_idx" not in edge


def test_endpoint_single_source() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    assert "edge_npu_count" not in runner, (
        "runner must reuse worker._lwd_early_recv_endpoint, not recompute"
    )
    assert "_lwd_early_recv_endpoint" in runner


def test_harvest_split_and_pre_super_order() -> None:
    prepare = _method(_RUNNER, "_prepare_inputs")
    harvest_lineno = None
    super_lineno = None
    for lineno, name in _call_names(prepare):
        if name == "_lwd_harvest_up_batches" and harvest_lineno is None:
            harvest_lineno = lineno
        if name == "_prepare_inputs" and super_lineno is None:
            super_lineno = lineno
    assert harvest_lineno is not None and super_lineno is not None
    assert harvest_lineno < super_lineno, (
        "harvest (both phases) must stay before super()._prepare_inputs; "
        "moving the tensor phase is deferred item D1, not this refactor"
    )

    orch = _segment(_RUNNER, "_lwd_harvest_up_batches")
    stage_pos = orch.index("_lwd_stage_up_batch")
    tensor_pos = orch.index("_lwd_harvest_tensors_one")
    window_pos = orch.index("_lwd_check_up_windows")
    assert stage_pos < tensor_pos < window_pos, (
        "per-seqno order must mirror the pre-split loop: length check, "
        "host gate + broadcast, window check + mrope (log order included)"
    )
    for helper in ("_lwd_stage_up_batch", "_lwd_harvest_tensors_one",
                   "_lwd_check_up_windows"):
        assert f"def {helper}(" in _RUNNER.read_text(encoding="utf-8")


def test_harvest_failfast_messages_preserved() -> None:
    src = _RUNNER.read_text(encoding="utf-8")
    for fragment in (
        "req not in input_batch",
        "INJECT window mismatch",
        "prompt chunk offset mismatch",
        "mrope payload missing",
        "LWD requires one prompt offset and mrope flag per request",
    ):
        assert fragment in src, f"fail-fast line dropped: {fragment}"


def test_inject_reuses_carried_index() -> None:
    seg = _segment(_RUNNER, "_lwd_inject_remote_embeds")
    assert "req_id_to_index" not in seg, (
        "inject must reuse the index carried by the harvest meta phase"
    )
    assert "offsets[idx]" in seg


def test_meta_ready_via_worker_api() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    worker = _WORKER.read_text(encoding="utf-8")
    assert "note_lwd_meta_ready()" in runner
    # The runner must not touch the worker's private attribute; the
    # executor still reads it, so the writer stays on the worker.
    assert "_lwd_meta_ready_event" not in runner
    assert "def note_lwd_meta_ready" in worker
    assert "_lwd_meta_ready_event" in _segment(_WORKER, "note_lwd_meta_ready")


def test_collect_split_and_payload_protocol() -> None:
    src = _RUNNER.read_text(encoding="utf-8")
    for helper in ("_lwd_pack_plain_down_rows", "_lwd_pack_spec_down_rows",
                   "_lwd_stage_down_meta"):
        assert f"def {helper}(" in src
    orch = _segment(_RUNNER, "_lwd_collect_down_payload")
    assert "_lwd_stage_down_meta(" in orch
    # The 5-tuple DOWN payload protocol (hidden, pinned, None, req_ids,
    # None) with req_ids filtered by the captured snapshot mask.
    assert "[r for i, r in enumerate(batch_req_ids) if valid[i]]" in orch


def test_finished_release_single_owner() -> None:
    worker_src = _WORKER.read_text(encoding="utf-8")
    runner_src = _RUNNER.read_text(encoding="utf-8")
    assert "def lwd_release_finished" in runner_src
    flush = _segment(_WORKER, "_lwd_cloud_flush_finished")
    assert "lwd_release_finished" in flush
    assert "hash_state.discard" not in worker_src, (
        "release logic must live on the runner only"
    )

def test_layer_trace_rehomed_and_installed() -> None:
    """The layer-trace parity instrumentation is homed in lwd_debug with its
    install points restored.

    c22c04b53 removed the install call from NPUModelRunner.load_model
    while clearing probes from base classes and nobody rehomed it,
    orphaning ~120 lines of hook/checksum machinery. Now rehomed: edge/cloud
    install from their own load_model (env-gated); centralized is manual opt-in.
    """
    edge = _EDGE.read_text(encoding="utf-8")
    worker = _WORKER.read_text(encoding="utf-8")
    edge_runner_src = (_REPO_ROOT / "vllm_ascend/worker/lwd_edge_model/lwd_edge_model_runner.py").read_text(encoding="utf-8")
    assert "install_lwd_layer_trace(self.get_model())" in edge_runner_src
    assert "install_lwd_layer_trace(self.model_runner.get_model())" in worker
    assert "vllm_ascend.worker.lwd_debug.lwd_layer_trace" in edge_runner_src
    assert (_REPO_ROOT / "vllm_ascend/worker/lwd_debug/lwd_layer_trace.py").exists()
    assert not (_REPO_ROOT / "vllm_ascend/worker/lwd_layer_trace.py").exists()

def test_v4_hash_wiring_on_main() -> None:
    """Guards for the V4 Hash expert-routing wiring on main new structure (B<->A).

    B side (ww branch) = the deepseek_v4.py:460 construction-time runner wiring +
    the experts_selector merge block + the fused_moe forward injection.
    main object graph changed (the runner is held indirectly; no .runner attr),
      ported as: (1) single-point context injection at the model forward entry
         (hash layer = index, ordinary layer = None); the forwarding layers
         never inject separately, so an inner layer cannot clobber the outer;
      (2) the merge block in router/fused_topk_router sqrtsoftplus branch
         (merged table + row indices as token ids; reuses native TP/EP kernels);
      (3) the B-side post-TP/EP row-count fail-fast is kept.
    """
    model_src = (_REPO_ROOT / "vllm_ascend/models/deepseek_v4/model.py").read_text(encoding="utf-8")
    assert "get_forward_context().lwd_hash_layer_idx = self.layer_idx if self.hash else None" in model_src
    for path in ("vllm_ascend/ops/fused_moe/fused_moe.py",
                 "vllm_ascend/ops/fused_moe/routed_experts.py"):
        assert "lwd_hash_layer_idx" not in (_REPO_ROOT / path).read_text(encoding="utf-8"), path
    router_src = (
        _REPO_ROOT / "vllm_ascend/ops/fused_moe/router/fused_topk_router.py"
    ).read_text(encoding="utf-8")
    assert "merge_hash_expert_ids(" in router_src
    assert 'getattr(_EXTRA_CTX, "lwd_hash_expert_ids", None)' in router_src
    assert "routing rows do not match router logits" in router_src

