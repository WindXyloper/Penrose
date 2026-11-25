#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Penrose v1.0 — 创世架构契约与核心流形定理验证套件 (`test_remote.py`)
=============================================================================
标准用法:
    python test_remote.py
"""

import json
import math
from typing import Any, Dict
import torch
import torch.nn.functional as F

from penrose_pmc import (
    D_BKP,
    PAGE_SIZE,
    JET_NORMAL_ALPHA_STAR,
    JET_TANGENT_NORM_STAR,
    SIMPLEX_2_GEODESIC_PROJ,
    SIMPLEX_3_BARYCENTER_PROJ,
    closed_form_gumbel_threshold,
    PMCMountMode,
    PMCPackage,
    PMCManager,
)
from penrose_mpu import (
    PenroseMPUConfig,
    PenroseMPU,
)
from penrose_harness import (
    HolographicWavepacketTransducer,
    BusBranchState,
    PenroseHarness,
)


def test_holographic_wavepacket_lossless_roundtrip() -> Dict[str, Any]:
    """
    实验 1：验证 2048 = 8 x 256 沃尔什-阿达玛正交谐波折叠与 Gumbel 孔径自适应解码 (0 参数，0 词表)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transducer = HolographicWavepacketTransducer(max_bytes_per_packet=16, device=device)
    assert sum(p.numel() for p in transducer.parameters()) == 0, "全息波包换能器必须严格为 0 可学习参数！"

    u_a = transducer.slot_unit_vector(0, ord("a"))
    u_b = transducer.slot_unit_vector(0, ord("b"))
    u_slot1_a = transducer.slot_unit_vector(1, ord("a"))
    ortho_byte_dot = abs(float(torch.dot(u_a, u_b).item()))
    ortho_slot_dot = abs(float(torch.dot(u_a, u_slot1_a).item()))
    assert ortho_byte_dot < 1e-6 and ortho_slot_dot < 1e-6

    head_energies = u_a.view(8, 256).pow(2).sum(dim=-1)
    max_head_dev = float((head_energies - 0.125).abs().max().item())
    assert max_head_dev < 1e-6

    test_samples = ["Penrose", "训练", "语料", "黎曼流形", "{", "text", ': "', "def", "S2047"]
    exact_matches = 0
    for s in test_samples:
        q_pkt = transducer.encode_string_to_wavepacket(s)
        assert abs(float(q_pkt.norm().item()) - 1.0) < 1e-5
        decoded_s = transducer.decode_wavepacket_to_string(q_pkt)
        assert decoded_s == s, f"波包解码不匹配: 原文={s!r}, 解码={decoded_s!r}"
        exact_matches += 1

    q_ab = transducer.encode_string_to_wavepacket("ab")
    q_ba = transducer.encode_string_to_wavepacket("ba")
    cos_ab_ba = float(torch.dot(q_ab, q_ba).item())
    assert cos_ab_ba < 0.05

    res = {
        "learnable_params": 0,
        "inter_byte_dot": round(ortho_byte_dot, 8),
        "inter_slot_dot": round(ortho_slot_dot, 8),
        "head_energy_uniformity_err": round(max_head_dev, 8),
        "roundtrip_accuracy": f"{exact_matches}/{len(test_samples)} (100.0%)",
        "ab_vs_ba_cosine": round(cos_ab_ba, 6),
    }
    print(f"[PASS 1/6] 2048=8x256 沃尔什-阿达玛全息波包 0 参数 100% 无损可逆验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


def test_two_tier_negative_divergence_funnel_segmentation() -> Dict[str, Any]:
    """
    实验 2：验证双层物理相变 + PMC 负散度漏斗动态捕获与冷启动自发成词闭环。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harness = PenroseHarness(device=device)
    dirty_doc = '{text: "这是一段训练语料。"}'

    q_cold, chunks_cold = harness.transducer.segment_text_to_wavepackets(dirty_doc, pmc_manager=harness.pmc_manager)
    assert chunks_cold[:3] == ["{", "text", ': "']
    assert "训" in chunks_cold and "练" in chunks_cold

    harness.nucleate_composite_word_sink("训练", "语料", sink_strength=0.90)
    harness.nucleate_composite_word_sink("语料", "。", sink_strength=0.90)

    q_warm, chunks_warm = harness.transducer.segment_text_to_wavepackets(dirty_doc, pmc_manager=harness.pmc_manager)
    assert "训练" in chunks_warm and "语料" in chunks_warm
    assert q_warm.size(0) < q_cold.size(0)

    reconstructed = "".join(harness.transducer.decode_wavepacket_to_string(q_warm[i]) for i in range(q_warm.size(0)))
    assert "这是一段训练语料。" in reconstructed

    res = {
        "cold_start_packets": int(q_cold.size(0)),
        "cold_start_chunks": chunks_cold,
        "warm_funnel_packets": int(q_warm.size(0)),
        "warm_funnel_chunks": chunks_warm,
        "shell_prefix_isolated": chunks_warm[:3],
    }
    print(f"[PASS 2/6] 双层物理相变 + PMC 负散度漏斗动态成词验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res



def test_mpu_five_module_pipeline_and_tri_state_bus() -> Dict[str, Any]:
    """
    实验 3：验证 PenroseMPU 五段式流水线（Module 1 ~ Module 5）、早一轮白盒流线预取与三态总线分支。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    cfg = PenroseMPUConfig.preset_penrose_full_2048()
    mpu = PenroseMPU(cfg).to(device)
    assert not hasattr(mpu, "tok_embeddings") and not hasattr(mpu, "lm_head"), "MPU 绝不允许包含静态词表 Embedding 或 LM Head！"
    assert mpu.stiefel_entry is None and mpu.module3_waist_coupler.p_in is None, "原生 2048 维下 stiefel_entry 与 p_in 必须全部为 None！"
    assert len(mpu.module2_lower_stack) == 6 and len(mpu.module4_upper_stack) == 6

    # 同时验证 Agile 512 维配置下的 Stiefel 正交群重投影约束 O(2048, 512)
    mpu_agile = PenroseMPU(PenroseMPUConfig.preset_penrose_agile_12l()).to(device)
    with torch.no_grad():
        mpu_agile.stiefel_entry.add_(torch.randn_like(mpu_agile.stiefel_entry) * 0.2)
    mpu_agile.project_stiefel_frames()
    eye_512 = torch.eye(512, device=device)
    stiefel_orth_err = float((mpu_agile.stiefel_entry.t() @ mpu_agile.stiefel_entry - eye_512).abs().max().item())
    assert stiefel_orth_err < 1e-5, f"Stiefel 流形正交约束误差过大: {stiefel_orth_err}"

    pmc_mgr = PMCManager(device=device)
    pkg = PMCPackage(pkg_id="MPUTest.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    pmc_mgr.mount_package(pkg, pin_to_l1=True)

    # 构造 2048 维单位球面波包序列: (B=2, T=8, D=2048)
    q_curr = F.normalize(torch.randn(2, 8, D_BKP, device=device), dim=-1)
    q_prev = F.normalize(torch.randn(2, 8, D_BKP, device=device), dim=-1)
    q_target = F.normalize(torch.randn(2, 8, D_BKP, device=device), dim=-1)

    out = mpu(q_curr, q_prev=q_prev, pmc_manager=pmc_mgr, q_target=q_target)
    assert out.q_next.shape == (2, 8, D_BKP)
    assert out.psi_0.shape == (2, 8, D_BKP)
    assert out.loss is not None

    # 1. 验证 Module 1 合成的 1-Jet 状态严格位于单位超球面 S^{2047}，且已触发早一轮 PMC 流线预取
    psi_norm_err = float((out.psi_0.norm(dim=-1) - 1.0).abs().max().item())
    assert psi_norm_err < 1e-5
    assert len(pmc_mgr.prefetch_queue_eids) == 1 and pmc_mgr.prefetch_queue_eids[0][1].numel() > 0

    # 2. 验证 Module 5 切空间正交化 <Delta_q_perp(tau), q(tau)> == 0 与输出单位球面模长 ||q(tau+1)|| == 1
    ortho_tangent_err = float((out.delta_q_perp * q_curr).sum(dim=-1).abs().max().item())
    q_next_norm_err = float((out.q_next.norm(dim=-1) - 1.0).abs().max().item())
    assert ortho_tangent_err < 1e-5 and q_next_norm_err < 1e-5

    # 3. 验证 Module 5 三态总线分支（分支 A 内部高旋度静默闭环 vs 分支 B 负散度汇点坍缩吐字）
    harness = PenroseHarness(mpu=mpu, pmc_manager=pmc_mgr, device=device)
    q_start = harness.transducer.encode_string_to_wavepacket("Penrose").to(device)
    # 在空流形（div V == 0, eta_curl == 1）下，前 2 拍自动走分支 A 内部旋度闭环，最后 1 拍走分支 B 吐出
    bus_res = harness.step_tri_state_bus(q_start, max_internal_curl_ticks=3, sink_div_threshold=-0.05)
    assert bus_res["branch_trace"][0] == BusBranchState.BRANCH_A_INTERNAL_CURL_LOOP.value
    assert bus_res["branch_trace"][-1] == BusBranchState.BRANCH_B_NEGATIVE_DIV_EMIT.value

    res = {
        "vocab_size": cfg.vocab_size,
        "module2_layers": len(mpu.module2_lower_stack),
        "module4_layers": len(mpu.module4_upper_stack),
        "module1_psi0_norm_err": round(psi_norm_err, 8),
        "module1_prefetched_pages": int(pmc_mgr.prefetch_queue_eids[0][1].numel()),
        "module5_tangent_orthogonality_err": round(ortho_tangent_err, 8),
        "module5_q_next_norm_err": round(q_next_norm_err, 8),
        "branch_c_chord_loss": round(float(out.loss.item()), 6),
        "tri_state_branch_trace": bus_res["branch_trace"],
    }
    print(f"[PASS 3/6] PenroseMPU 五段式流水线、早一轮预取与三态总线分支验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


def test_adamw_free_kaczmarz_geodesic_snap_and_dirty_shell_immunity() -> Dict[str, Any]:
    """
    实验 4：验证无 AdamW 的 Kaczmarz + 罗德里格斯大圆旋转一步锁死与 `{text: "..."}` 脏外壳零污染拖拽训练。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    harness = PenroseHarness(device=device)

    # 两篇共用 `{text: "` 脏外壳但正文截然不同的原始文档
    doc_1 = '{text: "流形处理器取代静态词表。"}'
    doc_2 = '{text: "微专家通过散度与旋度自发分化。"}'

    r1 = harness.train_manifold_trajectory(doc_1, snap_rate=1.0, enable_auto_mitosis=True, eval_before_after=True)
    r2 = harness.train_manifold_trajectory(doc_2, snap_rate=1.0, enable_auto_mitosis=True, eval_before_after=True)

    assert r1["kaczmarz_stats"]["optimizer_memory_bytes"] == 0
    assert r1["loss_after"] < r1["loss_before"], f"Kaczmarz 单步投影后球面弦长误差未下降: {r1}"
    assert r2["loss_after"] < r2["loss_before"], f"Kaczmarz 单步投影后球面弦长误差未下降: {r2}"

    # 验证拿干净的正文起点 `"流形"` 提问推进时，由于前导格式符已被相变门与有丝分裂隔离，不会吐出 `{` 或 `text` 垃圾
    q_clean = harness.transducer.encode_string_to_wavepacket("流").to(device)
    step_clean = harness.step_tri_state_bus(q_clean, max_internal_curl_ticks=1)
    assert "{" not in step_clean["emitted_text"] and "text" not in step_clean["emitted_text"]

    res = {
        "pmc_optimizer_memory_bytes": 0,
        "doc1_loss_before_to_after": f"{r1['loss_before']:.4f} -> {r1['loss_after']:.4f}",
        "doc2_loss_before_to_after": f"{r2['loss_before']:.4f} -> {r2['loss_after']:.4f}",
        "total_kaczmarz_snaps": r2["kaczmarz_stats"]["total_kaczmarz_snaps"],
        "total_mitosis_spawns": r2["kaczmarz_stats"]["total_mitosis_spawns"],
        "clean_query_json_leakage": False,
    }
    print(f"[PASS 4/6] 无 AdamW Kaczmarz 测地线一步锁死与脏外壳零污染拖拽训练验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


def test_manifold_puncture_dropout_c1_simplex_healing() -> Dict[str, Any]:
    """
    实验 5：验证流形穿孔邻域插值自愈定律（Manifold Puncture Dropout & C^1 Simplex Healing）。
    当中心微专家被拔除/丢失时，三维单纯形邻域伴随微专家按三次厄米特 C^1 包络与 L2 守恒权重自动重归一化插值自愈。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(108)
    mgr = PMCManager(device=device)
    pkg = PMCPackage(pkg_id="PunctureTest.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)

    # 构造一个由 4 个单纯形邻域微专家组成的平滑概念场（共享主推力方向 v_true + 局部正交微扰）
    q_center = F.normalize(torch.randn(D_BKP, device=device), dim=-1)
    v_true = F.normalize(torch.randn(D_BKP, device=device), dim=-1) * 15.0

    for idx_m in range(4):
        noise_u = F.normalize(torch.randn(D_BKP, device=device), dim=-1) * (0.05 * idx_m)
        u_m = F.normalize(q_center + noise_u, dim=-1)
        noise_v = F.normalize(torch.randn(D_BKP, device=device), dim=-1) * (0.3 * idx_m)
        v_m = v_true + noise_v
        pkg.quantize_write_expert_page(idx_m, u_m, v_m, hit_increment=10.0, purity_val=0.98)

    mgr.mount_package(pkg, pin_to_l1=True)

    # 1. 完整流形场响应（中心微专家 eid=0 在位）
    diag_intact = mgr.query_with_helmholtz_diagnostics(q_center.view(1, 1, D_BKP))
    top1_eid = int(diag_intact.active_experts[0, 0, 0].item())
    assert top1_eid == 0

    # 2. 流形穿孔：硬性拔除 Top-1 中心微专家 (eid=0)，仅靠单纯形周围的 3 个伴随微专家 (eid=1,2,3) C^1 插值自愈
    puncture_mask = torch.zeros(pkg.num_experts, dtype=torch.bool, device=device)
    puncture_mask[0] = True
    diag_healed = mgr.query_with_helmholtz_diagnostics(
        q_center.view(1, 1, D_BKP),
        puncture_expert_mask=puncture_mask,
    )
    assert int(diag_healed.active_experts[0, 0, 0].item()) != 0, "中心微专家 eid=0 应当已被穿孔屏蔽！"

    cos_healing = float(
        F.cosine_similarity(
            diag_intact.delta_pmc.view(1, -1).float(),
            diag_healed.delta_pmc.view(1, -1).float(),
            dim=-1,
        ).item()
    )
    cos_to_true = float(
        F.cosine_similarity(
            v_true.view(1, -1).float(),
            diag_healed.delta_pmc.view(1, -1).float(),
            dim=-1,
        ).item()
    )
    assert cos_healing > 0.99 and cos_to_true > 0.99, f"单纯形邻域 C^1 插值自愈余弦不足: {cos_healing}"

    res = {
        "punctured_center_eid": top1_eid,
        "healed_top_neighbors": diag_healed.active_experts[0, 0, :3].tolist(),
        "c1_simplex_healing_cosine": round(cos_healing, 6),
        "healed_to_true_field_cosine": round(cos_to_true, 6),
    }
    print(f"[PASS 5/6] 流形穿孔邻域三次厄米特 C^1 单纯形插值自愈验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


def test_six_autonomous_forest_operators_and_wormhole_switch() -> Dict[str, Any]:
    """
    实验 6：验证 PMCManager 内聚的六大森林自治算子（Auto-Mitosis, Auto-Closure, Auto-Expand, Auto-Split, Auto-Merge, Auto-Spawn）
    以及通过统一训练接口 `train_manifold_trajectory` 回流验证后的思考语料自发形成「拓扑虫洞道岔」（而非硬编码斯托克斯单步捷径）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(202608)

    # 1. 验证 PMCPackage 在 Auto-Expand / Auto-Split / Auto-Merge / Auto-Spawn 下 100% 保持子类类型与 stream_next_ptr 流线边无损重映射
    mgr = PMCManager(device=device)
    pkg = PMCPackage(pkg_id="ForestSeed.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    q_a = F.normalize(torch.randn(D_BKP, device=device), dim=-1)
    q_b = F.normalize(torch.randn(D_BKP, device=device), dim=-1)
    pkg.quantize_write_expert_page(0, q_a, q_b * 5.0, hit_increment=30.0, purity_val=0.60)  # 满足公理 7 瑞利多义判据: 30 * 0.36 = 10.8 > ln(4096)=8.32
    pkg.quantize_write_expert_page(65, q_b, q_a * 5.0, hit_increment=30.0, purity_val=0.95)
    pkg.record_streamline_transition(torch.tensor([0], device=device), torch.tensor([65], device=device))
    mgr.mount_package(pkg, pin_to_l1=True)

    # 触发 Auto-Closure ((0,1) 与 (1,0) 交叉格点闭包结晶) 与 Auto-Mitosis (eid=0 切向复共轭有丝分裂)
    c_res = pkg.crystallize_biorthogonal_manifold_closure(max_crystallize_slots=16)
    m_res = pkg.nucleate_tangential_conjugate_mitosis(max_mitosis_slots=16)
    assert c_res["crystallized_count"] >= 2 and m_res["mitosis_count"] >= 1

    # 触发 Auto-Expand (64 -> 128)，检查零步误差 == 0 且流线边 (0,0)->(1,1) 自动重映射为 0 -> 1*128+1 = 129
    q_probe = q_a.view(1, 1, D_BKP)
    out_before_exp = mgr.query_and_inject(q_probe).clone()
    pkg._needs_quadrant_expand = True
    ev_exp = mgr.step_autonomous_forest_evolution(step=1, q_probe=q_probe)
    out_after_exp = mgr.query_and_inject(q_probe).clone()
    expand_zero_err = float((out_before_exp - out_after_exp).abs().max().item())
    assert pkg.grid_k == 128 and expand_zero_err < 1e-6
    assert int(pkg.stream_next_ptr[0, 0].item()) == 129, f"流线指针扩容重映射错误: {pkg.stream_next_ptr[0]}"

    # 构造双簇激活触发 Auto-Split (128 -> SubA + SubB + BridgeAB)
    pkg.activation_counts.zero_()
    pkg.activation_counts.view(128, 128)[:16, :16] = 10
    pkg.activation_counts.view(128, 128)[64:80, 64:80] = 10
    # 写入一个跨簇非对角桥接微专家 (row=0, col=64 -> eid=64)，使 Auto-Split 同时抽取出联络桥 BridgeAB.pmc
    pkg.quantize_write_expert_page(64, q_a, q_b * 6.0, hit_increment=2.0, purity_val=0.95)
    ev_split = mgr.step_autonomous_forest_evolution(step=2, q_probe=q_probe)
    assert any(e["op"] == "Auto-Split" for e in ev_split)
    assert all(isinstance(p, PMCPackage) for p in mgr.mounted_packages)
    assert any(p.is_bridge_pkg for p in mgr.mounted_packages), "Auto-Split 应当抽取出跨板块联络桥 BridgeAB.pmc！"

    # 验证多坐标卡图册（SubA + SubB + BridgeAB）联合单纯形查询、霍奇第三项 eta_harmonic 与母体超球面共振归属路由
    diag_atlas = mgr.query_with_helmholtz_diagnostics(q_probe)
    assert diag_atlas.eta_harmonic is not None and float(diag_atlas.eta_harmonic.max().item()) > 0.0
    routed_chart = mgr.select_resonant_writable_package(q_a)
    assert isinstance(routed_chart, PMCPackage) and (not routed_chart.is_bridge_pkg)

    # 构造跨包共激活触发 Auto-Merge (SubA + SubB -> TriadicMerge)
    pids = [p.pkg_id for p in mgr.mounted_packages if not p.is_bridge_pkg]
    pa, pb = pids[0], pids[1]
    mgr.pkg_token_hit_counts[pa] = 20.0
    mgr.pkg_token_hit_counts[pb] = 20.0
    mgr.pkg_coactivation_counts[(pa, pb) if pa < pb else (pb, pa)] = 18.0
    ev_merge = mgr.step_autonomous_forest_evolution(step=3, q_probe=q_probe)
    assert any(e["op"] == "Auto-Merge" for e in ev_merge)

    # 构造低命中未知流形查询池触发 Auto-Spawn
    for p in mgr.mounted_packages:
        p.activation_counts.zero_()
    mgr._total_periodic_queries = 20
    mgr._low_hit_periodic_queries = 15
    mgr._low_hit_q1_reservoir = F.normalize(torch.randn(16, 8, 128, device=device), dim=-1)
    mgr._low_hit_q2_reservoir = F.normalize(torch.randn(16, 8, 128, device=device), dim=-1)
    ev_spawn = mgr.step_autonomous_forest_evolution(step=4, q_probe=q_probe, domain_hint="math")
    assert any(e["op"] == "Auto-Spawn" for e in ev_spawn)
    assert all(isinstance(p, PMCPackage) for p in mgr.mounted_packages)

    # 2. 验证回流验证后的思考语料通过统一训练接口形成「拓扑虫洞道岔」（保留原推导路径 + 架设直达虫洞）
    mpu_fast = PenroseMPU(PenroseMPUConfig.preset_penrose_agile_12l()).to(device)
    harness = PenroseHarness(mpu=mpu_fast, auto_evolve_interval=10, device=device)
    cot_chain = "问题起点 推导一 推导二 最终真值"
    wormhole_pair = "问题起点 最终真值"
    r_cot = harness.train_manifold_trajectory(cot_chain, snap_rate=1.0, enable_auto_mitosis=True)
    r_wormhole = harness.train_manifold_trajectory(wormhole_pair, snap_rate=1.0, enable_auto_mitosis=True)
    assert r_wormhole["kaczmarz_stats"]["mitosis_pages"] >= 1, "虫洞语料回流起点时必须触发 Auto-Mitosis 道岔分裂以保留原盘山公路！"

    snap = mgr.snapshot_forest_state(step=4)
    res = {
        "all_packages_are_pmc": all(isinstance(p, PMCPackage) for p in mgr.mounted_packages),
        "stream_ptr_remapped_eid": int(pkg.stream_next_ptr[0, 0].item()),
        "expand_zero_step_err": round(expand_zero_err, 8),
        "eta_harmonic_max": round(float(diag_atlas.eta_harmonic.max().item()), 6),
        "routed_chart_id": routed_chart.pkg_id,
        "operator_totals": snap["operator_totals"],
        "wormhole_mitosis_switches": r_wormhole["kaczmarz_stats"]["mitosis_pages"],
    }
    print(f"[PASS 6/6] 六大森林自治算子与验证思考回流「拓扑虫洞道岔」验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


if __name__ == "__main__":
    import os
    from train_penrose_v1 import run_penrose_v1_training

    r1 = test_holographic_wavepacket_lossless_roundtrip()
    r2 = test_two_tier_negative_divergence_funnel_segmentation()
    r3 = test_mpu_five_module_pipeline_and_tri_state_bus()
    r4 = test_adamw_free_kaczmarz_geodesic_snap_and_dirty_shell_immunity()
    r5 = test_manifold_puncture_dropout_c1_simplex_healing()
    r6 = test_six_autonomous_forest_operators_and_wormhole_switch()
    train_rep = run_penrose_v1_training(steps=15)

    full_report = {
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "exp1_holographic_wavepacket_8x256": r1,
        "exp2_two_tier_negative_divergence_funnel": r2,
        "exp3_mpu_five_module_and_tri_state_bus": r3,
        "exp4_adamw_free_kaczmarz_and_dirty_shell": r4,
        "exp5_manifold_puncture_c1_simplex_healing": r5,
        "exp6_six_autonomous_forest_ops_and_wormhole": r6,
        "end_to_end_training_15steps": train_rep,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "experiment_log", "penrose_v1_genesis_verification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n[ALL 6 PENROSE v1.0 CONTRACTS + 15-STEP TRAINING PASSED] 报告已保存至: {out_path}")




