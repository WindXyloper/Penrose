#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Penrose v1.0 & Legacy .bkp — 100% 真实自然语言语料第一性原理全面重验脚本
"""

import gc
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bkp_manager import BKPPackage, BKPManager, BKPMountMode
from penrose_pmc import (
    D_BKP,
    PAGE_SIZE,
    JET_NORMAL_ALPHA_STAR,
    JET_TANGENT_NORM_STAR,
    PMCMountMode,
    PMCPackage,
    PMCManager,
)
from penrose_mpu import PenroseMPUConfig, PenroseMPU
from penrose_harness import HolographicWavepacketTransducer, PenroseHarness


def collect_real_utf8_corpus() -> Dict[str, List[str]]:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus: Dict[str, List[str]] = {
        "wiki_sci": [],
        "math_cot": [],
        "code": [],
        "raw_jsonl_dirty": [],
    }

    remote_sample = os.path.join(root_dir, "workspace", "real_raw_corpus_sample.jsonl")
    if os.path.exists(remote_sample):
        with open(remote_sample, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    dom = obj.get("domain", "")
                    raw_line = obj.get("raw_line", "")
                    if dom == "wiki_sci":
                        corpus["wiki_sci"].append(raw_line)
                    elif dom in ("math", "cot_distill"):
                        corpus["math_cot"].append(raw_line)
                    elif dom == "code":
                        corpus["code"].append(raw_line)
                except Exception:
                    pass

    doc_files = [
        os.path.join(root_dir, "学术发现.md"),
        os.path.join(root_dir, "洞见.md"),
        os.path.join(root_dir, "升级思路.md"),
        os.path.join(root_dir, "README.md"),
    ]
    for df in doc_files:
        if os.path.exists(df):
            with open(df, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 60]
            for p in paras:
                if any(k in p for k in ["定理", "公式", "推导", "因为", "所以", "如果", "那么", "\\frac", "\\sum", "="]):
                    corpus["math_cot"].append(p[:400])
                else:
                    corpus["wiki_sci"].append(p[:400])

    py_files = [
        os.path.join(root_dir, "penrose_mpu.py"),
        os.path.join(root_dir, "penrose_pmc.py"),
        os.path.join(root_dir, "penrose_harness.py"),
        os.path.join(root_dir, "bkp_manager.py"),
    ]
    for pf in py_files:
        if os.path.exists(pf):
            with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            chunk = []
            for ln in lines:
                chunk.append(ln)
                if len("".join(chunk)) >= 240:
                    s = "".join(chunk).strip()
                    if "def " in s or "if " in s or "for " in s or "return " in s:
                        corpus["code"].append(s[:360])
                    chunk = []

    for s in corpus["wiki_sci"][:30] + corpus["math_cot"][:30]:
        clean_s = s.replace("\n", " ").replace('"', "'")[:200]
        corpus["raw_jsonl_dirty"].append(f'{{"text": "{clean_s}"}}')

    return corpus


def audit_legacy_bkp_manifolds(device: torch.device) -> Dict[str, Any]:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bkp_paths = sorted(
        glob.glob(os.path.join(root_dir, "workspace", "old_bicamera", "workspace", "bkp_store", "micro_125m_autoforest_v36", "*.bkp"))
        + glob.glob(os.path.join(root_dir, "workspace", "bkp_store_forest", "micro_125m_mitosis64k_v42", "*.bkp"))
    )
    results = {}
    for bp in bkp_paths:
        meta_p = bp + ".meta.pt"
        if not os.path.exists(meta_p):
            continue
        pkg_name = os.path.basename(os.path.dirname(bp)) + "/" + os.path.basename(bp)
        pkg = BKPPackage.load_package(bp, mount_mode=BKPMountMode.RO_SYS, device=device)
        k = pkg.grid_k
        num_e = pkg.num_experts

        scale_v = pkg.scales[:, 1].float()
        act_cnt = pkg.activation_counts.float()
        purity = pkg.purity_r.float()
        active_mask = (scale_v > 1e-6) & (act_cnt > 0)
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        n_active = int(active_indices.numel())
        if n_active == 0:
            continue

        u_raw, v_raw = pkg.read_experts_from_mmap_pages(active_indices)
        u_hat = F.normalize(u_raw.float(), dim=-1)
        v_raw = v_raw.float()
        v_norms = v_raw.norm(dim=-1)
        v_hat = F.normalize(v_raw, dim=-1)

        uv_cos = (u_hat * v_hat).sum(dim=-1)
        eta_div_point = uv_cos.pow(2)
        eta_curl_point = 1.0 - eta_div_point

        act_sub = act_cnt[active_indices]
        pur_sub = purity[active_indices]
        med_act = float(act_sub.median().item())
        med_pur = float(pur_sub.median().item())

        ctrl_like_mask = (act_sub >= med_act) & (pur_sub <= med_pur)
        data_like_mask = (pur_sub > med_pur)

        results[pkg_name] = {
            "grid_k": k,
            "total_experts": num_e,
            "active_experts": n_active,
            "mean_single_point_uv_cos": round(float(uv_cos.mean().item()), 6),
            "mean_single_point_abs_uv_cos": round(float(uv_cos.abs().mean().item()), 6),
            "mean_single_point_eta_div": round(float(eta_div_point.mean().item()), 6),
            "mean_single_point_eta_curl": round(float(eta_curl_point.mean().item()), 6),
            "max_single_point_eta_div": round(float(eta_div_point.max().item()), 6),
            "mean_v_norm": round(float(v_norms.mean().item()), 4),
            "mean_activation_count": round(float(act_sub.mean().item()), 2),
            "max_activation_count": round(float(act_sub.max().item()), 2),
            "mean_context_purity": round(float(pur_sub.mean().item()), 4),
            "ctrl_like_experts_count": int(ctrl_like_mask.sum().item()),
            "ctrl_like_mean_v_norm": round(float(v_norms[ctrl_like_mask].mean().item()), 4) if ctrl_like_mask.any() else 0.0,
            "ctrl_like_mean_purity": round(float(pur_sub[ctrl_like_mask].mean().item()), 4) if ctrl_like_mask.any() else 0.0,
            "ctrl_like_mean_act": round(float(act_sub[ctrl_like_mask].mean().item()), 2) if ctrl_like_mask.any() else 0.0,
            "data_like_experts_count": int(data_like_mask.sum().item()),
            "data_like_mean_v_norm": round(float(v_norms[data_like_mask].mean().item()), 4) if data_like_mask.any() else 0.0,
            "data_like_mean_purity": round(float(pur_sub[data_like_mask].mean().item()), 4) if data_like_mask.any() else 0.0,
            "data_like_mean_act": round(float(act_sub[data_like_mask].mean().item()), 2) if data_like_mask.any() else 0.0,
        }
        del pkg
        if len(results) >= 4:
            break
    print(f"[TRACK-A] Audited {len(results)} real legacy .bkp manifolds.", flush=True)
    return results


def is_instruction_or_control_chunk(chunk_str: str) -> bool:
    s = chunk_str.strip()
    if not s:
        return True
    ctrl_keywords = {
        "def", "if", "else", "elif", "for", "while", "return", "import", "from",
        "class", "in", "and", "or", "not", "assert", "with", "try", "except",
        "=", "==", "!=", "<=", ">=", "+", "-", "*", "/", "(", ")", "[", "]", "{", "}",
        ":", ",", ".", ";", "->", '":', '{"', '"}',
        "因为", "所以", "如果", "那么", "推导", "证明", "求解", "可得", "即", "则", "当", "时",
        "的", "了", "在", "是", "与", "或", "及", "将", "通过", "由", "为",
    }
    if s in ctrl_keywords:
        return True
    if all(not ("\u4e00" <= c <= "\u9fff" or c.isalpha()) for c in s):
        return True
    return False


def train_and_probe_penrose_pmc_on_real_corpus(
    corpus: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cfg = PenroseMPUConfig.preset_penrose_agile_12l()
    mpu = PenroseMPU(cfg).to(device)
    pmc_manager = PMCManager(device=device)
    base_pkg = PMCPackage(
        pkg_id="Base.pmc",
        grid_k=64,
        mount_mode=PMCMountMode.RW_SHARED,
        is_base_pkg=True,
        device=device,
    )
    pmc_manager.mount_package(base_pkg, pin_to_l1=True)
    harness = PenroseHarness(mpu=mpu, pmc_manager=pmc_manager, device=device)

    train_texts: List[Tuple[str, str]] = []
    for t in corpus["wiki_sci"][:8]:
        train_texts.append(("wiki_sci", t[:160]))
    for t in corpus["math_cot"][:8]:
        train_texts.append(("math_cot", t[:160]))
    for t in corpus["code"][:8]:
        train_texts.append(("code", t[:160]))
    for t in corpus["raw_jsonl_dirty"][:8]:
        train_texts.append(("raw_jsonl", t[:160]))

    expert_surface_hits: Dict[int, List[str]] = defaultdict(list)
    expert_incoming_states: Dict[int, List[torch.Tensor]] = defaultdict(list)
    expert_target_states: Dict[int, List[torch.Tensor]] = defaultdict(list)

    mpu_opt = torch.optim.AdamW(harness.mpu.parameters(), lr=2e-3, weight_decay=0.01)
    t0 = time.perf_counter()
    losses = []
    for step_idx, (dom, raw_txt) in enumerate(train_texts):
        q_seq, chunks = harness.transducer.segment_text_to_wavepackets(
            raw_txt, pmc_manager=None
        )
        if q_seq.shape[0] < 3:
            continue
        stat = harness.train_manifold_trajectory(
            raw_text_or_wavepackets=q_seq,
            snap_rate=0.85,
            enable_auto_mitosis=True,
            mpu_optimizer=mpu_opt,
            domain_hint=dom,
            auto_evolve=False,
        )
        losses.append(stat["loss"])
        if (step_idx + 1) % 8 == 0:
            print(f"[TRACK-B] step {step_idx + 1}/{len(train_texts)} loss={stat['loss']:.4f}", flush=True)

        with torch.no_grad():
            q_in = q_seq[:-1]
            q_tgt = q_seq[1:]
            q1, q2 = harness.pmc_manager._standardize_subqueries(q_in, update_center=False)
            s1, s2, _, _ = harness.pmc_manager._precompute_pkg_subkey_scores(
                base_pkg, q1, q2
            )
            r1 = s1.amax(dim=1).argmax(dim=-1).reshape(-1)
            r2 = s2.amax(dim=1).argmax(dim=-1).reshape(-1)
            eids = (r1 * base_pkg.grid_k + r2).tolist()
            for pos_i, eid in enumerate(eids):
                chunk_s = chunks[pos_i] if pos_i < len(chunks) else ""
                expert_surface_hits[eid].append(chunk_s)
                if len(expert_incoming_states[eid]) < 16:
                    expert_incoming_states[eid].append(q_in[pos_i].detach().cpu())
                    expert_target_states[eid].append(q_tgt[pos_i].detach().cpu())

    train_wall_sec = time.perf_counter() - t0
    del mpu_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _summarize_pmc_expert_geometry(
        harness, base_pkg, train_texts, losses, train_wall_sec,
        expert_surface_hits, expert_incoming_states, expert_target_states, device
    )


def _summarize_pmc_expert_geometry(
    harness: PenroseHarness,
    base_pkg: PMCPackage,
    train_texts: List[Tuple[str, str]],
    losses: List[float],
    train_wall_sec: float,
    expert_surface_hits: Dict[int, List[str]],
    expert_incoming_states: Dict[int, List[torch.Tensor]],
    expert_target_states: Dict[int, List[torch.Tensor]],
    device: torch.device,
) -> Dict[str, Any]:
    active_mask = base_pkg.scales[:, 1] > 1e-6
    active_eids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    n_active = int(active_eids.numel())

    u_raw, v_raw = base_pkg.read_experts_from_mmap_pages(active_eids)
    u_hat = F.normalize(u_raw.float(), dim=-1)
    v_raw = v_raw.float()
    v_norms = v_raw.norm(dim=-1)
    v_hat = F.normalize(v_raw, dim=-1)

    uv_cos = (u_hat * v_hat).sum(dim=-1)
    eta_div_point = uv_cos.pow(2)
    eta_curl_point = 1.0 - eta_div_point

    out_degrees = (base_pkg.stream_next_ptr >= 0).sum(dim=-1).float()
    in_degrees = torch.zeros(base_pkg.num_experts, dtype=torch.float32, device=device)
    valid_ptrs = base_pkg.stream_next_ptr[base_pkg.stream_next_ptr >= 0]
    if valid_ptrs.numel() > 0:
        ones = torch.ones_like(valid_ptrs, dtype=torch.float32)
        in_degrees.scatter_add_(0, valid_ptrs.long(), ones)

    ctrl_experts_stats = []
    data_experts_stats = []

    for idx_in_act, eid_t in enumerate(active_eids):
        eid = int(eid_t.item())
        hits = expert_surface_hits.get(eid, [])
        if len(hits) == 0:
            continue
        ctrl_ratio = sum(1 for h in hits if is_instruction_or_control_chunk(h)) / float(len(hits))
        uv_c = float(uv_cos[idx_in_act].item())
        e_curl = float(eta_curl_point[idx_in_act].item())
        v_n = float(v_norms[idx_in_act].item())
        act_c = float(base_pkg.activation_counts[eid].item())
        pur_c = float(base_pkg.purity_r[eid].item())
        out_d = float(out_degrees[eid].item())
        in_d = float(in_degrees[eid].item())

        in_list = expert_incoming_states.get(eid, [])
        tgt_list = expert_target_states.get(eid, [])
        if len(in_list) >= 2:
            q_in_mat = F.normalize(torch.stack(in_list, dim=0).to(device), dim=-1)
            q_tgt_mat = F.normalize(torch.stack(tgt_list, dim=0).to(device), dim=-1)
            v_vec = v_raw[idx_in_act].unsqueeze(0)
            q_pushed_mat = F.normalize(q_in_mat + v_vec, dim=-1)
            var_in = 1.0 - float(q_in_mat.mean(dim=0).norm().item())
            var_pushed = 1.0 - float(q_pushed_mat.mean(dim=0).norm().item())
            var_tgt = 1.0 - float(q_tgt_mat.mean(dim=0).norm().item())
            div_bundle = var_pushed - var_in
        else:
            var_in = 0.0
            var_pushed = 0.0
            var_tgt = 0.0
            div_bundle = 0.0

        entry = {
            "eid": eid,
            "sample_chunks": hits[:6],
            "hit_count": len(hits),
            "ctrl_ratio": round(ctrl_ratio, 4),
            "single_point_uv_cos": round(uv_c, 6),
            "single_point_eta_curl": round(e_curl, 6),
            "v_norm": round(v_n, 4),
            "activation_count": round(act_c, 2),
            "context_purity": round(pur_c, 4),
            "stream_out_degree": out_d,
            "stream_in_degree": in_d,
            "bundle_var_in": round(var_in, 6),
            "bundle_var_pushed": round(var_pushed, 6),
            "bundle_var_tgt": round(var_tgt, 6),
            "dynamical_div_bundle": round(div_bundle, 6),
        }
        if ctrl_ratio >= 0.6:
            ctrl_experts_stats.append(entry)
        else:
            data_experts_stats.append(entry)
    audit_b = _format_pmc_audit_result(
        train_texts, losses, train_wall_sec, n_active,
        uv_cos, eta_div_point, eta_curl_point, ctrl_experts_stats, data_experts_stats
    )

    # Track C: Real-corpus Spectral Biclustering Split & Direct-Sum / Triadic Merge (NO artificial codebook injection!)
    with torch.no_grad():
        q_eval_sample, _ = harness.transducer.segment_text_to_wavepackets(
            train_texts[0][1], pmc_manager=None
        )
        diag_before_split = harness.pmc_manager.query_with_helmholtz_diagnostics(q_eval_sample.unsqueeze(0))
        v_norm_before_split = float(diag_before_split.delta_pmc.norm(dim=-1).mean().item())

        split_res = base_pkg.bkp_split_spectral(cross_ratio_threshold=0.99, return_bridge=True)
        assert split_res is not None
        sub_a, sub_b, bridge_ab = split_res
        active_a = int((sub_a.scales[:, 1] > 1e-6).sum().item())
        active_b = int((sub_b.scales[:, 1] > 1e-6).sum().item())
        active_bridge = int((bridge_ab.scales[:, 1] > 1e-6).sum().item()) if bridge_ab is not None else 0
        sub_a_k, sub_b_k = sub_a.grid_k, sub_b.grid_k
        stream_edges_before = int((base_pkg.stream_next_ptr >= 0).sum().item())

        merged_ds = PMCPackage.bkp_merge_direct_sum(sub_a, sub_b, "RealMergedDS.pmc")
        merged_ds_k = merged_ds.grid_k
        stream_edges_merged = int((merged_ds.stream_next_ptr >= 0).sum().item())
        del merged_ds
        if device.type == "cuda":
            torch.cuda.empty_cache()

        merged_tri = PMCPackage.bkp_merge_triadic_bundle(sub_a, sub_b, "RealMergedTri.pmc")
        merged_tri_k = merged_tri.grid_k
        del merged_tri, sub_a, sub_b, bridge_ab
        if device.type == "cuda":
            torch.cuda.empty_cache()

    track_c = {
        "parent_grid_k": base_pkg.grid_k,
        "parent_active_experts": n_active,
        "sub_a_grid_k": sub_a_k,
        "sub_a_active_experts": active_a,
        "sub_b_grid_k": sub_b_k,
        "sub_b_active_experts": active_b,
        "bridge_ab_active_experts": active_bridge,
        "merged_direct_sum_grid_k": merged_ds_k,
        "merged_triadic_grid_k": merged_tri_k,
        "stream_edges_parent": stream_edges_before,
        "stream_edges_merged_direct_sum": stream_edges_merged,
        "v_norm_before_split": round(v_norm_before_split, 6),
    }

    # Track D: Real-corpus Closed-form Inverse Kaczmarz Unlearning & Apoptosis
    target_erase_doc = train_texts[2][1]
    retained_doc = train_texts[10][1]
    q_erase, _ = harness.transducer.segment_text_to_wavepackets(target_erase_doc, pmc_manager=None)
    q_retain, _ = harness.transducer.segment_text_to_wavepackets(retained_doc, pmc_manager=None)

    with torch.no_grad():
        d_erase_pre = harness.pmc_manager.query_with_helmholtz_diagnostics(q_erase[:-1].unsqueeze(0))
        d_retain_pre = harness.pmc_manager.query_with_helmholtz_diagnostics(q_retain[:-1].unsqueeze(0))
        v_erase_pre = float(d_erase_pre.delta_pmc.norm(dim=-1).mean().item())
        v_retain_pre = float(d_retain_pre.delta_pmc.norm(dim=-1).mean().item())

    erase_res = harness.drag_and_drop_erase(
        q_erase,
        erase_rate=1.0,
        eval_before_after=True,
    )

    with torch.no_grad():
        d_erase_post = harness.pmc_manager.query_with_helmholtz_diagnostics(q_erase[:-1].unsqueeze(0))
        d_retain_post = harness.pmc_manager.query_with_helmholtz_diagnostics(q_retain[:-1].unsqueeze(0))
        v_erase_post = float(d_erase_post.delta_pmc.norm(dim=-1).mean().item())
        v_retain_post = float(d_retain_post.delta_pmc.norm(dim=-1).mean().item())

    track_d = {
        "erased_doc_preview": target_erase_doc[:60],
        "retained_doc_preview": retained_doc[:60],
        "v_erase_pre": round(v_erase_pre, 6),
        "v_erase_post": round(v_erase_post, 6),
        "erase_attenuation_ratio": round(1.0 - (v_erase_post / max(1e-8, v_erase_pre)), 6),
        "v_retain_pre": round(v_retain_pre, 6),
        "v_retain_post": round(v_retain_post, 6),
        "retain_preservation_ratio": round(v_retain_post / max(1e-8, v_retain_pre), 6),
        "erase_stats": erase_res.get("erase_stats", {}),
    }

    # Track E: Real-corpus Dirty JSONL Noise Immunity & Spontaneous Composite Word Nucleation
    clean_real_para = train_texts[0][1][:120]
    dirty_wrapped_para = f'{{"crawl_id": 99812, "status": "uncleaned_raw", "payload": "{clean_real_para.replace(chr(10), " ")}"}}'
    q_clean_p, chunks_clean_p = harness.transducer.segment_text_to_wavepackets(clean_real_para, pmc_manager=None)
    q_dirty_p, chunks_dirty_p = harness.transducer.segment_text_to_wavepackets(dirty_wrapped_para, pmc_manager=None)
    # Match wavepackets corresponding to the clean payload inside dirty JSONL
    sim_matrix = torch.matmul(q_clean_p, q_dirty_p.t())
    payload_match_cos = float(sim_matrix.max(dim=-1).values.mean().item())

    real_word_test = "流形处理器在超球面上沿着测地线流线推进"
    _, chunks_before_nuc = harness.transducer.segment_text_to_wavepackets(real_word_test, pmc_manager=harness.pmc_manager)
    for w1, w2 in [("流形", "处理器"), ("超球", "面上"), ("测地", "线")]:
        harness.nucleate_composite_word_sink(w1, w2)
    _, chunks_after_nuc = harness.transducer.segment_text_to_wavepackets(real_word_test, pmc_manager=harness.pmc_manager)

    track_e = {
        "dirty_jsonl_payload_recovery_cosine": round(payload_match_cos, 6),
        "clean_wavepacket_count": int(q_clean_p.shape[0]),
        "dirty_wavepacket_count": int(q_dirty_p.shape[0]),
        "word_nucleation_text": real_word_test,
        "chunks_before_nucleation": chunks_before_nuc,
        "chunks_after_nucleation": chunks_after_nuc,
        "wavepacket_count_before": len(chunks_before_nuc),
        "wavepacket_count_after": len(chunks_after_nuc),
    }

    audit_b["track_C_real_spectral_split_and_merge"] = track_c
    audit_b["track_D_real_inverse_kaczmarz_unlearning"] = track_d
    audit_b["track_E_real_dirty_jsonl_and_nucleation"] = track_e
    return audit_b


def _format_pmc_audit_result(
    train_texts: List[Tuple[str, str]],
    losses: List[float],
    train_wall_sec: float,
    n_active: int,
    uv_cos: torch.Tensor,
    eta_div_point: torch.Tensor,
    eta_curl_point: torch.Tensor,
    ctrl_experts_stats: List[Dict[str, Any]],
    data_experts_stats: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _agg(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not entries:
            return {}
        multi_entries = [e for e in entries if e["hit_count"] >= 2]
        return {
            "count": len(entries),
            "multi_hit_count": len(multi_entries),
            "mean_single_point_uv_cos": round(sum(e["single_point_uv_cos"] for e in entries) / len(entries), 6),
            "mean_single_point_abs_uv_cos": round(sum(abs(e["single_point_uv_cos"]) for e in entries) / len(entries), 6),
            "mean_single_point_eta_curl": round(sum(e["single_point_eta_curl"] for e in entries) / len(entries), 6),
            "mean_v_norm": round(sum(e["v_norm"] for e in entries) / len(entries), 4),
            "mean_activation_count": round(sum(e["activation_count"] for e in entries) / len(entries), 2),
            "mean_context_purity": round(sum(e["context_purity"] for e in entries) / len(entries), 4),
            "mean_stream_out_degree": round(sum(e["stream_out_degree"] for e in entries) / len(entries), 4),
            "mean_stream_in_degree": round(sum(e["stream_in_degree"] for e in entries) / len(entries), 4),
            "mean_bundle_var_in": round(sum(e["bundle_var_in"] for e in multi_entries) / max(1, len(multi_entries)), 6),
            "mean_bundle_var_pushed": round(sum(e["bundle_var_pushed"] for e in multi_entries) / max(1, len(multi_entries)), 6),
            "mean_bundle_var_tgt": round(sum(e["bundle_var_tgt"] for e in multi_entries) / max(1, len(multi_entries)), 6),
            "mean_dynamical_div_bundle": round(sum(e["dynamical_div_bundle"] for e in multi_entries) / max(1, len(multi_entries)), 6),
            "top_examples": sorted(entries, key=lambda x: x["hit_count"], reverse=True)[:5],
        }

    return {
        "train_docs": len(train_texts),
        "train_wall_sec": round(train_wall_sec, 3),
        "loss_start": round(losses[0], 6) if losses else 0.0,
        "loss_end": round(losses[-1], 6) if losses else 0.0,
        "loss_mean_first10": round(sum(losses[:10]) / max(1, len(losses[:10])), 6),
        "loss_mean_last10": round(sum(losses[-10:]) / max(1, len(losses[-10:])), 6),
        "active_experts": n_active,
        "mean_single_point_uv_cos": round(float(uv_cos.mean().item()), 6),
        "mean_single_point_abs_uv_cos": round(float(uv_cos.abs().mean().item()), 6),
        "mean_single_point_eta_div": round(float(eta_div_point.mean().item()), 6),
        "mean_single_point_eta_curl": round(float(eta_curl_point.mean().item()), 6),
        "instruction_control_experts_summary": _agg(ctrl_experts_stats),
        "data_entity_experts_summary": _agg(data_experts_stats),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PROBE] Running on device: {device}")
    corpus = collect_real_utf8_corpus()
    print(f"[CORPUS] wiki_sci={len(corpus['wiki_sci'])}, math_cot={len(corpus['math_cot'])}, code={len(corpus['code'])}, raw_jsonl={len(corpus['raw_jsonl_dirty'])}")
    bkp_audit = audit_legacy_bkp_manifolds(device)
    pmc_audit = train_and_probe_penrose_pmc_on_real_corpus(corpus, device)
    out = {
        "device": str(device),
        "corpus_counts": {k: len(v) for k, v in corpus.items()},
        "track_A_legacy_bkp_audit": bkp_audit,
        "track_B_penrose_pmc_real_nl_audit": pmc_audit,
    }
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(root_dir, "workspace", "_real_first_principles_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved to {out_path}")




