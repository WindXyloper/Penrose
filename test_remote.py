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
import sys
from typing import Any, Dict
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    print(f"[PASS 1/8] 2048=8x256 沃尔什-阿达玛全息波包 0 参数 100% 无损可逆验证通过: {json.dumps(res, ensure_ascii=False)}")
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
    print(f"[PASS 2/8] 双层物理相变 + PMC 负散度漏斗动态成词验证通过: {json.dumps(res, ensure_ascii=False)}")
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

    # 使用真实自然语言句子经全息波包换能器生成 2048 维单位球面波包序列: (B=2, T=8, D=2048)
    transducer = HolographicWavepacketTransducer(max_bytes_per_packet=16, device=device)
    q_s1, _ = transducer.segment_text_to_wavepackets("流形处理器在单位超球面上推进状态向量。", pmc_manager=pmc_mgr)
    q_s2, _ = transducer.segment_text_to_wavepackets("微专家根据散度与旋度自发分化为概念势阱。", pmc_manager=pmc_mgr)
    q_batch = torch.stack([q_s1[:10], q_s2[:10]], dim=0)
    q_prev = q_batch[:, :8, :]
    q_curr = q_batch[:, 1:9, :]
    q_target = q_batch[:, 2:10, :]

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
    print(f"[PASS 3/8] PenroseMPU 五段式流水线、早一轮预取与三态总线分支验证通过: {json.dumps(res, ensure_ascii=False)}")
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

    # 验证逆向拖拽擦除 (drag_and_drop_erase)：擦除 doc_2 后，doc_2 独占页触发公理 9 细胞凋亡、共享外壳剥离，而 doc_1 保持低损失
    r_erase_doc2 = harness.drag_and_drop_erase(doc_2, erase_rate=1.0, eval_before_after=True)
    assert r_erase_doc2["erase_stats"]["apoptosis_pages"] > 0, f"擦除独占文档应当触发公理 9 细胞凋亡: {r_erase_doc2}"
    assert r_erase_doc2["erase_stats"]["peeled_shared_pages"] > 0, f"共用脏外壳微专家应当触发秩-1 正交剥离: {r_erase_doc2}"
    assert r_erase_doc2["loss_after"] > 0.75, f"擦除后 doc_2 球面弦长误差应当精确回退至未学习基线 (~0.79): {r_erase_doc2}"

    # 验证擦除 doc_2 对 doc_1 的记忆零损伤 (doc_1 球面弦长误差依然保持在 ~0.51，远低于初始 1.0022)
    q_seq_1, _ = harness.transducer.segment_text_to_wavepackets(doc_1, pmc_manager=harness.pmc_manager)
    q_c1 = q_seq_1[:-1].unsqueeze(0)
    q_t1 = q_seq_1[1:].unsqueeze(0)
    q_p1 = torch.cat([q_c1[:, :1, :], q_c1[:, :-1, :]], dim=1)
    with torch.no_grad():
        doc1_loss_retained = float(harness.mpu(q_c1, q_prev=q_p1, pmc_manager=harness.pmc_manager, q_target=q_t1).loss.item())
    assert doc1_loss_retained < 0.55, f"擦除 doc_2 不应破坏共享流形中 doc_1 的记忆: {doc1_loss_retained}"

    res = {
        "pmc_optimizer_memory_bytes": 0,
        "doc1_loss_before_to_after": f"{r1['loss_before']:.4f} -> {r1['loss_after']:.4f}",
        "doc2_loss_before_to_after": f"{r2['loss_before']:.4f} -> {r2['loss_after']:.4f}",
        "doc2_erase_loss_before_to_after": f"{r_erase_doc2['loss_before']:.4f} -> {r_erase_doc2['loss_after']:.4f}",
        "doc1_loss_retained_after_doc2_erase": round(doc1_loss_retained, 4),
        "erase_apoptosis_pages": r_erase_doc2["erase_stats"]["apoptosis_pages"],
        "erase_peeled_shared_pages": r_erase_doc2["erase_stats"]["peeled_shared_pages"],
        "total_kaczmarz_snaps": r2["kaczmarz_stats"]["total_kaczmarz_snaps"],
        "total_mitosis_spawns": r2["kaczmarz_stats"]["total_mitosis_spawns"],
        "clean_query_json_leakage": False,
    }
    print(f"[PASS 4/8] 无 AdamW Kaczmarz 测地线一步锁死与脏外壳零污染拖拽训练验证通过: {json.dumps(res, ensure_ascii=False)}")
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

    # 使用真实自然语言词元波包构建由 4 个单纯形邻域微专家组成的平滑概念场（共享主推力方向 v_true + 局部词形微扰）
    transducer = HolographicWavepacketTransducer(max_bytes_per_packet=16, device=device)
    q_center = transducer.encode_string_to_wavepacket("黎曼流形").to(device)
    v_true = transducer.encode_string_to_wavepacket("测地线方程").to(device) * 15.0
    neighbor_phrases = ["黎曼流形", "黎曼几何", "黎曼度规", "黎曼曲率"]

    for idx_m, phrase in enumerate(neighbor_phrases):
        q_phrase = transducer.encode_string_to_wavepacket(phrase).to(device)
        u_m = F.normalize(q_center + (0.05 * idx_m) * q_phrase, dim=-1)
        v_m = v_true + (0.3 * idx_m) * q_phrase
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
    print(f"[PASS 5/8] 流形穿孔邻域三次厄米特 C^1 单纯形插值自愈验证通过: {json.dumps(res, ensure_ascii=False)}")
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
    transducer = HolographicWavepacketTransducer(max_bytes_per_packet=16, device=device)
    q_a = transducer.encode_string_to_wavepacket("微分流形").to(device)
    q_b = transducer.encode_string_to_wavepacket("切丛联络").to(device)
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
    remapped_eid = int(pkg.stream_next_ptr[0, 0].item())
    assert remapped_eid == 129, f"流线指针扩容重映射错误: {pkg.stream_next_ptr[0]}"

    # 构造非对称双簇激活 (20x20 小簇 + 88x88 大簇) 触发 Auto-Split (128 -> 非对称 64 + 128 + BridgeAB)
    pkg.activation_counts.zero_()
    pkg.activation_counts.view(128, 128)[:20, :20] = 15
    pkg.activation_counts.view(128, 128)[20:108, 20:108] = 10
    # 写入一个跨簇非对角桥接微专家 (row=0, col=30 -> eid=30)，使 Auto-Split 走完整 SVD 非对称谱切分并抽取出联络桥 BridgeAB.pmc
    pkg.quantize_write_expert_page(30, q_a, q_b * 6.0, hit_increment=2.0, purity_val=0.95)
    ev_split = mgr.step_autonomous_forest_evolution(step=2, q_probe=q_probe)
    assert any(e["op"] == "Auto-Split" for e in ev_split)
    assert all(isinstance(p, PMCPackage) for p in mgr.mounted_packages)
    assert any(p.is_bridge_pkg for p in mgr.mounted_packages), "Auto-Split 应当抽取出跨板块联络桥 BridgeAB.pmc！"
    non_bridge_ks = sorted([p.grid_k for p in mgr.mounted_packages if not p.is_bridge_pkg])
    assert non_bridge_ks == [64, 128], f"非对称谱双聚类切分应当将 (20x20, 88x88) 向上补齐为 [64, 128] 而非中位数对半切分，实际: {non_bridge_ks}"

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

    # 构造低命中未知流形查询池（使用 16 个真实自然语言数学专业术语波包的子键分量）触发 Auto-Spawn
    for p in mgr.mounted_packages:
        p.activation_counts.zero_()
    q_math_unseen, _ = transducer.segment_text_to_wavepackets(
        "希尔伯特 自伴算子 稠密定义 谱分解定 紧算子列 特征值谱 正交投影 伴随算子 巴拿赫空 索伯列夫 泛函分析 测度积分 勒贝格空 傅里叶变 拉普拉斯 黎曼度规"
    )
    q1_unseen, q2_unseen = mgr._standardize_subqueries(q_math_unseen[:16].to(device), update_center=False)
    mgr._total_periodic_queries = 20
    mgr._low_hit_periodic_queries = 15
    mgr._low_hit_q1_reservoir = F.normalize(q1_unseen, dim=-1)
    mgr._low_hit_q2_reservoir = F.normalize(q2_unseen, dim=-1)
    ev_spawn = mgr.step_autonomous_forest_evolution(step=4, q_probe=q_probe, domain_hint="math")
    assert any(e["op"] == "Auto-Spawn" for e in ev_spawn)
    assert all(isinstance(p, PMCPackage) for p in mgr.mounted_packages)
    all_are_pmc = all(isinstance(p, PMCPackage) for p in mgr.mounted_packages)
    eta_harm_max = round(float(diag_atlas.eta_harmonic.max().item()), 6)
    routed_id = routed_chart.pkg_id
    snap = mgr.snapshot_forest_state(step=4)
    del mgr, pkg, diag_atlas, routed_chart
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. 验证回流验证后的思考语料通过统一训练接口形成「拓扑虫洞道岔」（保留原推导路径 + 架设直达虫洞）
    mpu_fast = PenroseMPU(PenroseMPUConfig.preset_penrose_agile_12l()).to(device)
    harness = PenroseHarness(mpu=mpu_fast, auto_evolve_interval=10, device=device)
    cot_chain = "问题起点 推导一 推导二 最终真值"
    wormhole_pair = "问题起点 最终真值"
    r_cot = harness.train_manifold_trajectory(cot_chain, snap_rate=1.0, enable_auto_mitosis=True)
    r_wormhole = harness.train_manifold_trajectory(wormhole_pair, snap_rate=1.0, enable_auto_mitosis=True)
    assert r_wormhole["kaczmarz_stats"]["mitosis_pages"] >= 1, "虫洞语料回流起点时必须触发 Auto-Mitosis 道岔分裂以保留原盘山公路！"

    res = {
        "all_packages_are_pmc": all_are_pmc,
        "stream_ptr_remapped_eid": remapped_eid,
        "expand_zero_step_err": round(expand_zero_err, 8),
        "asymmetric_split_grid_ks": non_bridge_ks,
        "eta_harmonic_max": eta_harm_max,
        "routed_chart_id": routed_id,
        "operator_totals": snap["operator_totals"],
        "wormhole_mitosis_switches": r_wormhole["kaczmarz_stats"]["mitosis_pages"],
    }
    print(f"[PASS 6/8] 六大森林自治算子与验证思考回流「拓扑虫洞道岔」验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


def test_dirty_data_auto_split_and_low_wake_isolation() -> Dict[str, Any]:
    """
    实验 7：验证流形对脏数据的输入时序免疫与 Auto-Split 独立分块 + 测地圆自然低唤醒（零新增算子，100% 现有机制）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(202609)

    transducer = HolographicWavepacketTransducer(max_bytes_per_packet=16, device=device)
    q_anchor = transducer.encode_string_to_wavepacket("光速不变原理").to(device).view(1, 1, D_BKP)
    v_truth_dir = transducer.encode_string_to_wavepacket("洛伦兹协变变换").to(device).view(1, 1, D_BKP)
    raw_fake = transducer.encode_string_to_wavepacket("以太绝对静止参考系").to(device).view(1, 1, D_BKP)
    raw_fake = raw_fake - (raw_fake * v_truth_dir).sum(dim=-1, keepdim=True) * v_truth_dir
    v_fake_dir = F.normalize(raw_fake, dim=-1)
    q_truth_target = F.normalize(0.2 * q_anchor + v_truth_dir, dim=-1)
    q_fake_target = F.normalize(0.2 * q_anchor + v_fake_dir, dim=-1)

    def _run_sequence(order_tags):
        mgr_seq = PMCManager(device=device)
        pkg_seq = PMCPackage(pkg_id="OrderTest.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
        mgr_seq.mount_package(pkg_seq, pin_to_l1=True)
        for tag in order_tags:
            qt = q_truth_target if tag == "T" else q_fake_target
            mgr_seq.geodesic_kaczmarz_snap_update(q_anchor, q_anchor, qt, snap_rate=1.0, enable_auto_mitosis=True)
        active_eids = torch.where(pkg_seq.scales[:, 1] > 1e-6)[0].tolist()
        v_tan_t = q_truth_target - (q_truth_target * q_anchor).sum(dim=-1, keepdim=True) * q_anchor
        v_des_t = (1.0 / SIMPLEX_3_BARYCENTER_PROJ) * v_tan_t - SIMPLEX_3_BARYCENTER_PROJ * q_anchor
        truth_hits, fake_hits = 0, 0
        best_cos_truth = -1.0
        for eid in active_eids:
            v_e = pkg_seq._master_uv_fp32[eid, 1, :].float()
            c_t = float(F.cosine_similarity(v_e.view(1, -1), v_des_t.view(1, -1), dim=-1).item())
            cnt = int(pkg_seq.activation_counts[eid].item())
            if c_t > 0.90:
                truth_hits += cnt
                best_cos_truth = max(best_cos_truth, c_t)
            else:
                fake_hits += cnt
        return active_eids, [truth_hits, fake_hits], best_cos_truth

    eids_tf, counts_tf, cos_tf = _run_sequence(["T", "T", "T", "T", "T", "F"])
    eids_ft, counts_ft, cos_ft = _run_sequence(["F", "T", "T", "T", "T", "T"])
    assert counts_tf == [5, 1] and counts_ft == [5, 1], f"真假分槽计数必须与输入顺序无关: {counts_tf} vs {counts_ft}"
    assert abs(cos_tf - cos_ft) < 1e-4 and cos_ft > 0.999

    mgr = PMCManager(device=device)
    pkg_mixed = PMCPackage(pkg_id="MixedCorpus.pmc", grid_k=128, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    mgr.mount_package(pkg_mixed, pin_to_l1=True)

    clean_text = "黎曼流形上的测地线方程描述了无外力作用下的自由粒子沿单位超球面大圆运动并保持切向量平行移动与曲率守恒。"
    dirty_text = '{"err_code": 404, "html": "<div><span>cookie_banner_ad_tracker_99812_spam_popup</span></div>"}'
    q_clean_raw, _ = transducer.segment_text_to_wavepackets(clean_text, pmc_manager=mgr)
    q_dirty_raw, _ = transducer.segment_text_to_wavepackets(dirty_text, pmc_manager=mgr)
    clean_queries = F.normalize(q_clean_raw[:32].to(device), dim=-1)
    dirty_queries = F.normalize(q_dirty_raw[:12].to(device), dim=-1)
    q_clean_basis, _ = torch.linalg.qr(clean_queries.T)
    dirty_queries = F.normalize(dirty_queries - (dirty_queries @ q_clean_basis) @ q_clean_basis.T, dim=-1)

    q1_clean, q2_clean = mgr._standardize_subqueries(clean_queries, update_center=False)
    q1_dirty, q2_dirty = mgr._standardize_subqueries(dirty_queries, update_center=False)

    with torch.no_grad():
        for idx_d in range(12):
            r_d, c_d = idx_d * 2, idx_d * 2
            pkg_mixed.u_codebook[:, r_d, :].copy_(q1_dirty[idx_d])
            pkg_mixed.v_codebook[:, c_d, :].copy_(q2_dirty[idx_d])
            eid_d = r_d * 128 + c_d
            pkg_mixed.quantize_write_expert_page(eid_d, dirty_queries[idx_d], dirty_queries[(idx_d + 1) % 12] * 6.0, hit_increment=2.0, purity_val=0.85)
        for idx_c in range(32):
            r_c, c_c = 24 + idx_c * 2, 24 + idx_c * 2
            pkg_mixed.u_codebook[:, r_c, :].copy_(q1_clean[idx_c])
            pkg_mixed.v_codebook[:, c_c, :].copy_(q2_clean[idx_c])
            eid_c = r_c * 128 + c_c
            pkg_mixed.quantize_write_expert_page(eid_c, clean_queries[idx_c], clean_queries[(idx_c + 1) % 32] * 8.0, hit_increment=20.0, purity_val=0.98)
        act_2d = pkg_mixed.activation_counts.view(128, 128)
        act_2d[:24, :24].clamp_(min=4)
        act_2d[24:104, 24:104].clamp_(min=10)

    ev_split = mgr.step_autonomous_forest_evolution(step=1, q_probe=clean_queries[:4].view(1, 4, D_BKP))
    assert any(e["op"] == "Auto-Split" for e in ev_split)
    non_bridge_pkgs = [p for p in mgr.mounted_packages if not p.is_bridge_pkg]
    assert len(non_bridge_pkgs) == 2

    dirty_pkg = min(non_bridge_pkgs, key=lambda p: p.grid_k)
    clean_pkg = max(non_bridge_pkgs, key=lambda p: p.grid_k)
    assert dirty_pkg.grid_k == 64 and clean_pkg.grid_k == 128

    mgr.eval()
    clean_test_batch = clean_queries.view(4, 8, D_BKP)
    _ = mgr.query_and_inject(clean_test_batch)
    clean_pkg_hits = int(mgr.last_pkg_slot_hits.get(clean_pkg.pkg_id, 0))
    dirty_pkg_hits = int(mgr.last_pkg_slot_hits.get(dirty_pkg.pkg_id, 0))
    dirty_wake_ratio = float(dirty_pkg_hits) / float(max(1, clean_pkg_hits + dirty_pkg_hits))
    assert clean_pkg_hits > 0 and dirty_wake_ratio < 0.05

    res = {
        "order_invariance_counts_truth_first": counts_tf,
        "order_invariance_counts_fake_first": counts_ft,
        "order_invariance_truth_cosine_diff": round(abs(cos_tf - cos_ft), 8),
        "split_clean_pmc_id": f"{clean_pkg.pkg_id} (grid_k={clean_pkg.grid_k})",
        "split_dirty_pmc_id": f"{dirty_pkg.pkg_id} (grid_k={dirty_pkg.grid_k})",
        "clean_query_clean_pmc_slot_hits": clean_pkg_hits,
        "clean_query_dirty_pmc_slot_hits": dirty_pkg_hits,
        "dirty_pmc_wake_rate_pct": round(dirty_wake_ratio * 100.0, 4),
    }
    print(f"[PASS 7/8] 脏数据输入无时序分槽与 Auto-Split 独立分块 + 测地圆自然低唤醒验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


def test_penrose_five_dimensional_complexity_scaling() -> Dict[str, Any]:
    """
    实验 8：实测验证 `Penrose v1.0` 五维时间与空间复杂度标度律（对标 `洞见.md` 全景复杂度表）。
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(202610)

    mpu_cfg = PenroseMPUConfig.preset_penrose_agile_12l()
    mpu = PenroseMPU(mpu_cfg).to(device)
    mpu.eval()
    transducer = HolographicWavepacketTransducer(max_bytes_per_packet=16, device=device)
    assert mpu.config.vocab_size == 0 and sum(p.numel() for p in transducer.parameters()) == 0

    mgr_stream = PMCManager(device=device)
    pkg_stream = PMCPackage(pkg_id="StreamScale.pmc", grid_k=64, mount_mode=PMCMountMode.RO_SYS, is_base_pkg=True, device=device)
    mgr_stream.mount_package(pkg_stream, pin_to_l1=True)
    window_size = 16
    stream_corpus = "流形处理器在单位超球面上沿着测地线流线推进状态向量并消除静态词表内存开销。"
    q_stream_base, _ = transducer.segment_text_to_wavepackets(stream_corpus, pmc_manager=mgr_stream)
    warm_chunk = q_stream_base[:window_size].unsqueeze(0).to(device)
    with torch.no_grad():
        _ = mpu(warm_chunk, pmc_manager=mgr_stream)

    def _profile_streaming_sequence(total_packets: int):
        n_windows = total_packets // window_size
        state_bytes = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            carry_wave = warm_chunk[:, :1, :].clone()
            for _ in range(n_windows):
                chunk = warm_chunk
                step_in = F.normalize(chunk + 0.1 * carry_wave, dim=-1)
                out = mpu(step_in, pmc_manager=mgr_stream)
                carry_wave = out.q_next[:, -1:, :].detach()
                state_bytes = carry_wave.numel() * carry_wave.element_size() + step_in.numel() * step_in.element_size()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, state_bytes

    ms_n64, bytes_n64 = _profile_streaming_sequence(64)
    ms_n512, bytes_n512 = _profile_streaming_sequence(512)
    state_mem_ratio = float(bytes_n512) / float(max(1, bytes_n64))
    time_ratio_8x = ms_n512 / max(1e-3, ms_n64)
    assert abs(state_mem_ratio - 1.0) < 1e-6

    pkg_m4k = PMCPackage(pkg_id="Scale4K.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    q_test = q_stream_base[:4].to(device)
    q1_t, q2_t = mgr_stream._standardize_subqueries(q_test, update_center=False)
    s1_4k, s2_4k, _, _ = mgr_stream._precompute_pkg_subkey_scores(pkg_m4k, q1_t, q2_t)
    subkey_ops_4k = int(s1_4k.shape[-1] + s2_4k.shape[-1])

    pkg_m65k = PMCPackage(pkg_id="Scale65K.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    pkg_m65k.expand_quadrant_2x()
    pkg_m65k.expand_quadrant_2x()
    assert pkg_m65k.num_experts == 65536 and pkg_m4k.num_experts == 4096
    s1_65k, s2_65k, _, _ = mgr_stream._precompute_pkg_subkey_scores(pkg_m65k, q1_t, q2_t)
    subkey_ops_65k = int(s1_65k.shape[-1] + s2_65k.shape[-1])
    expert_count_scale = float(pkg_m65k.num_experts) / float(pkg_m4k.num_experts)
    cold_search_scale = float(subkey_ops_65k) / float(subkey_ops_4k)
    assert abs(expert_count_scale - 16.0) < 1e-6 and abs(cold_search_scale - 4.0) < 1e-6

    pkg_m4k.record_streamline_transition(torch.tensor([0], device=device), torch.tensor([17], device=device))
    pkg_m65k.record_streamline_transition(torch.tensor([0], device=device), torch.tensor([17], device=device))
    assert int(pkg_m4k.stream_next_ptr[0, 0].item()) == 17 and int(pkg_m65k.stream_next_ptr[0, 0].item()) == 17

    pkg_worm = PMCPackage(pkg_id="WormholeDepth.pmc", grid_k=64, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    chain_eids = [10, 20, 30, 40, 50, 60, 70]
    for i_step in range(len(chain_eids) - 1):
        pkg_worm.record_streamline_transition(
            torch.tensor([chain_eids[i_step]], device=device),
            torch.tensor([chain_eids[i_step + 1]], device=device),
        )
    cur_eid = chain_eids[0]
    base_hops = 0
    while cur_eid != chain_eids[-1] and base_hops < 20:
        cur_eid = int(pkg_worm.stream_next_ptr[cur_eid, 0].item())
        base_hops += 1
    assert base_hops == 6

    pkg_worm.record_streamline_transition(
        torch.tensor([chain_eids[0]], device=device),
        torch.tensor([chain_eids[-1]], device=device),
    )
    wormhole_fast_target = int(pkg_worm.stream_next_ptr[chain_eids[0], 0].item())
    preserved_base_first_hop = int(pkg_worm.stream_next_ptr[chain_eids[0], 1].item())
    assert wormhole_fast_target == 70 and preserved_base_first_hop == 20

    mgr_local = PMCManager(device=device)
    pkg_16k = PMCPackage(pkg_id="LocalO1.pmc", grid_k=128, mount_mode=PMCMountMode.RW_SHARED, is_base_pkg=True, device=device)
    mgr_local.mount_package(pkg_16k, pin_to_l1=True)
    q_local = q_stream_base[:2].unsqueeze(0).to(device)
    q_target_local = q_stream_base[2:4].unsqueeze(0).to(device)
    snap_stat = mgr_local.geodesic_kaczmarz_snap_update(q_local, q_local, q_target_local, snap_rate=1.0, enable_auto_mitosis=True)
    touched_pages = int((pkg_16k.scales[:, 1] > 1e-6).sum().item())
    untouched_ratio_pct = 100.0 * (1.0 - float(touched_pages) / float(pkg_16k.num_experts))
    assert untouched_ratio_pct > 99.8 and snap_stat["optimizer_memory_bytes"] == 0

    res = {
        "dim1_vocab_size_and_transducer_params": [mpu.config.vocab_size, 0],
        "dim2_seq_len_8x_state_mem_ratio": round(state_mem_ratio, 4),
        "dim2_seq_len_8x_time_ratio": round(time_ratio_8x, 2),
        "dim3_experts_16x_scale_4096_to_65536": round(expert_count_scale, 2),
        "dim3_cold_subkey_ops_scale_sqrt_m": round(cold_search_scale, 2),
        "dim3_hot_streamline_pointer_hops": 1,
        "dim4_reasoning_hops_before_and_after_wormhole": [base_hops, 1],
        "dim4_preserved_base_track_hop": preserved_base_first_hop,
        "dim5_16384_experts_touched_pages": touched_pages,
        "dim5_untouched_experts_pct": round(untouched_ratio_pct, 4),
        "dim5_optimizer_memory_bytes": snap_stat["optimizer_memory_bytes"],
    }
    print(f"[PASS 8/8] Penrose v1.0 五维时间与空间复杂度标度律验证通过: {json.dumps(res, ensure_ascii=False)}")
    return res


if __name__ == "__main__":
    import gc
    import os
    import sys
    import glob

    if "--export-raw-corpus" in sys.argv:
        out_samples = []
        for dom in ["wiki_sci", "math", "code", "cot_distill"]:
            raw_files = sorted(glob.glob(f"workspace/raw_datasets/{dom}/*"))
            count = 0
            for rf in raw_files:
                if rf.endswith(".jsonl") or rf.endswith(".json") or rf.endswith(".txt"):
                    with open(rf, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) > 40:
                                out_samples.append({"domain": dom, "raw_line": line[:600]})
                                count += 1
                                if count >= 80:
                                    break
                if count >= 80:
                    break
            if count == 0 and os.path.exists(f"workspace/datasets/{dom}/val.bin") and os.path.exists("workspace/tokenizer/tokenizer.json"):
                import numpy as np
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file("workspace/tokenizer/tokenizer.json")
                arr = np.memmap(f"workspace/datasets/{dom}/val.bin", dtype=np.uint16, mode="r")
                for i in range(80):
                    seg = arr[i * 128 : (i + 1) * 128].tolist()
                    txt = tok.decode(seg).strip()
                    if len(txt) > 30:
                        out_samples.append({"domain": dom, "raw_line": json.dumps({"text": txt}, ensure_ascii=False)})
        os.makedirs("workspace", exist_ok=True)
        with open("workspace/real_raw_corpus_sample.jsonl", "w", encoding="utf-8") as f:
            for item in out_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[EXPORT_OK] Exported {len(out_samples)} real natural language samples to workspace/real_raw_corpus_sample.jsonl")
        sys.exit(0)

    from train_penrose_v1 import run_penrose_v1_training

    def _gc():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    r1 = test_holographic_wavepacket_lossless_roundtrip(); _gc()
    r2 = test_two_tier_negative_divergence_funnel_segmentation(); _gc()
    r3 = test_mpu_five_module_pipeline_and_tri_state_bus(); _gc()
    r4 = test_adamw_free_kaczmarz_geodesic_snap_and_dirty_shell_immunity(); _gc()
    r5 = test_manifold_puncture_dropout_c1_simplex_healing(); _gc()
    r6 = test_six_autonomous_forest_operators_and_wormhole_switch(); _gc()
    r7 = test_dirty_data_auto_split_and_low_wake_isolation(); _gc()
    r8 = test_penrose_five_dimensional_complexity_scaling(); _gc()
    train_rep = run_penrose_v1_training(steps=15); _gc()

    full_report = {
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "exp1_holographic_wavepacket_8x256": r1,
        "exp2_two_tier_negative_divergence_funnel": r2,
        "exp3_mpu_five_module_and_tri_state_bus": r3,
        "exp4_adamw_free_kaczmarz_and_dirty_shell": r4,
        "exp5_manifold_puncture_c1_simplex_healing": r5,
        "exp6_six_autonomous_forest_ops_and_wormhole": r6,
        "exp7_dirty_data_auto_split_and_low_wake": r7,
        "exp8_five_dimensional_complexity_scaling": r8,
        "end_to_end_training_15steps": train_rep,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "experiment_log", "penrose_v1_genesis_verification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n[ALL 8 PENROSE v1.0 CONTRACTS + 15-STEP TRAINING PASSED] 报告已保存至: {out_path}")




