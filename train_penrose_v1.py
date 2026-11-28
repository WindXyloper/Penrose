"""
Penrose v1.0 — 免清洗全语料流形训练与拖拽式蒸馏主程序 (`train_penrose_v1.py`)
=============================================================================
彻底抛弃静态词表 (Tokenizer / Embedding / LM Head) 与 PMC 的 AdamW 优化器：
1. 外围总线通过 `HolographicWavepacketTransducer` (2048 = 8 x 256 阿达玛正交谐波 + 双层负散度漏斗)
   将原始文本流直接转化为 2048 维单位球面波包序列 `q(1), q(2), ..., q(T) in S^{2047}`；
2. `PenroseMPU` (Module 1 ~ Module 5) 在球面上执行全序列并行 ResNet 残差推进，
   直接在 Module 5 出口（分支 C）计算球面弦长损失 `L_Penrose = 1 - <q(tau+1), q*(tau+1)>`；
3. `PMC` 四叉树流形块通过无 AdamW 的 `geodesic_kaczmarz_snap_update`（Kaczmarz 单纯形正交投影 +
   罗德里格斯大圆旋转 + Auto-Mitosis 切向有丝分裂）以 0 字节优化器显存一步吸附目标流形。
"""

import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional
import torch

from penrose_pmc import PMCMountMode, PMCPackage, PMCManager
from penrose_mpu import PenroseMPUConfig, PenroseMPU
from penrose_harness import PenroseHarness


DEFAULT_RAW_CORPUS: List[str] = [
    '{text: "这是一段训练语料。流形处理器直接在单位超球面上推进状态向量。"}',
    '{text: "微专家在流形演化中根据散度与旋度自发分化为数据势阱与指令旋度流。"}',
    '{text: "def penrose_step(q_curr, q_prev): return normalize(q_curr + delta_q_perp)"}',
    '{text: "The Manifold Processing Unit replaces token embeddings with 2048D holographic wavepackets."}',
]


def run_penrose_v1_training(
    steps: int = 20,
    lr_mpu: float = 1e-3,
    device_str: Optional[str] = None,
    use_full_2048: bool = True,
) -> Dict[str, Any]:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    cfg = (
        PenroseMPUConfig.preset_penrose_full_2048()
        if use_full_2048
        else PenroseMPUConfig.preset_penrose_agile_12l()
    )
    mpu = PenroseMPU(cfg).to(device)
    pmc_manager = PMCManager(device=device)
    base_pmc = PMCPackage(
        pkg_id="Base_v1.pmc",
        grid_k=128,
        mount_mode=PMCMountMode.RW_SHARED,
        is_base_pkg=True,
        device=device,
    )
    pmc_manager.mount_package(base_pmc, pin_to_l1=True)
    harness = PenroseHarness(mpu=mpu, pmc_manager=pmc_manager, device=device)

    # 注意：无需任何人工预先注册词表，高频共现短波包在 train_manifold_trajectory 中满闭式阈值 ceil(ln(8))=3 次即自发结晶复合词负散度汇点！
    # 仅 MPU 的 12 层 8 头元算子使用轻量优化器，外置 PMC 流形块 100% 使用 0 字节显存的 Kaczmarz 测地线直投！
    mpu_opt = torch.optim.AdamW(mpu.parameters(), lr=lr_mpu, weight_decay=0.01)

    t0 = time.time()
    history: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        raw_doc = DEFAULT_RAW_CORPUS[(step - 1) % len(DEFAULT_RAW_CORPUS)]
        res = harness.train_manifold_trajectory(
            raw_text_or_wavepackets=raw_doc,
            snap_rate=0.85,
            enable_auto_mitosis=True,
            mpu_optimizer=mpu_opt,
        )
        if step % 5 == 0 or step == 1 or step == steps:
            snap = harness.pmc_manager.snapshot_forest_state(step=step)
            active_pages = sum(p["active_scale_v_experts"] for p in snap["packages"])
            history.append({
                "step": step,
                "num_wavepackets": res["num_wavepackets"],
                "loss": res["loss"],
                "active_pages": active_pages,
                "mitosis_spawns": snap["operator_totals"]["mitosis_spawns"],
                "closure_count": snap["operator_totals"]["closure_count"],
                "forest_events": res["forest_events"],
            })
            print(
                f"[Penrose v1.0 | Step {step:03d}/{steps:03d}] "
                f"Packets={res['num_wavepackets']:02d} | "
                f"ChordLoss={res['loss']:.4f} | "
                f"ActivePMC={active_pages} | Mitosis={snap['operator_totals']['mitosis_spawns']} | "
                f"Closure={snap['operator_totals']['closure_count']}"
            )

    elapsed = time.time() - t0
    return {
        "architecture": "Penrose-v1.0 (MPU-2048 + PMC + Harness)",
        "d_model": cfg.d_model,
        "vocab_size": 0,
        "pmc_optimizer_memory_bytes": 0,
        "steps": steps,
        "elapsed_sec": round(elapsed, 3),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Penrose v1.0 Manifold Training")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    report = run_penrose_v1_training(steps=args.steps, lr_mpu=args.lr)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
