import os
import sys
import time
import math
import json
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from bicamera_kernel_draft import BicameraConfig, BicameraKernel
from bkp_manager import (
    BKPPackage,
    BKPManager,
    BKPMountMode,
    BKPRoutingConfig,
    D_BKP,
    closed_form_gumbel_threshold,
)
from eval_bicamera_benchmarks import (
    run_full_benchmark_suite,
    USER_ID,
    THINK_BOS_ID,
    THINK_EOS_ID,
)


# ==============================================================================
# 1. 多源混合流 + 连续跨块长 CoT 数据加载器 + 三段式双相 Loss Mask 生成器
# ==============================================================================
class MultiStreamCoTDataset:
    """
    同时管理 5 路 uint16 内存映射语料流:
    - general (通用双语底座)
    - wiki_sci (高密度百科科学)
    - math (数学推导)
    - code (代码工程)
    - cot_distill (不定长多跳 <|think|> ... <|/think|> 思维链蒸馏流)

    支持两种采样模式:
    1. 随机窗口采样 (Random Window Batch)
    2. 跨窗口连续轨迹采样 (Sequential Chunk Continuation):
       针对跨越多个窗口的不定长超长 CoT，保持游标连续推进，配合 BKPManager.stash_microbatch_trace()
       跨块累积整条长思考链在各个时间块唤醒的所有不同微专家梯度，一次联合更新！
    """

    def __init__(self, workspace: Path, seq_len: int = 512):
        self.workspace = workspace
        self.seq_len = seq_len
        self.streams: Dict[str, Dict[str, np.memmap]] = {}
        self.seq_cursors: Dict[str, np.ndarray] = {}
        self.doc_starts: Dict[str, Dict[str, np.ndarray]] = {}

        stream_dirs = {
            "general": workspace / "tokenized_bins",
            "wiki_sci": workspace / "datasets" / "wiki_sci",
            "math": workspace / "datasets" / "math",
            "code": workspace / "datasets" / "code",
            "cot_distill": workspace / "datasets" / "cot_distill",
        }
        for name, d_path in stream_dirs.items():
            tr_p = d_path / "train.bin"
            va_p = d_path / "val.bin"
            if tr_p.is_file() and va_p.is_file():
                self.streams[name] = {
                    "train": np.memmap(str(tr_p), dtype=np.uint16, mode="r"),
                    "val": np.memmap(str(va_p), dtype=np.uint16, mode="r"),
                }
                if name == "cot_distill":
                    self.doc_starts[name] = {}
                    for sp_k, mm in self.streams[name].items():
                        max_s = len(mm) - self.seq_len - 2
                        if max_s > 0:
                            bos_locs = np.flatnonzero(mm[:max_s] == 1)
                            if bos_locs.size > 0:
                                self.doc_starts[name][sp_k] = bos_locs.astype(np.int64)

        if "general" not in self.streams:
            raise FileNotFoundError(f"未找到基础数据集: {workspace / 'tokenized_bins'}")

        # 构建全量语料池无放回遍历索引 (Full-Corpus Natural Traversal Pool):
        # 将五大领域的所有训练窗口 (domain_id, start_offset) 汇总为统一池，抛弃人工 domain_schedule 配比，
        # 确保每个 Epoch 内全量语料中的每一个窗口都不偏不倚、不重不漏地恰好遍历 1 次！
        self.domain_names: List[str] = list(self.streams.keys())
        dom_id_chunks: List[np.ndarray] = []
        start_chunks: List[np.ndarray] = []
        self.domain_token_counts: Dict[str, int] = {}
        for d_idx, d_name in enumerate(self.domain_names):
            tr_mm = self.streams[d_name]["train"]
            self.domain_token_counts[d_name] = int(len(tr_mm))
            max_s = len(tr_mm) - self.seq_len - 2
            if max_s <= 0:
                continue
            win_starts = np.arange(0, max_s, self.seq_len, dtype=np.int64)
            dom_id_chunks.append(np.full(win_starts.shape, d_idx, dtype=np.uint8))
            start_chunks.append(win_starts)

        self.pool_dom_ids = np.concatenate(dom_id_chunks) if dom_id_chunks else np.zeros((1,), dtype=np.uint8)
        self.pool_starts = np.concatenate(start_chunks) if start_chunks else np.zeros((1,), dtype=np.int64)
        self.pool_perm = np.random.permutation(len(self.pool_starts))
        self.pool_cursor: int = 0
        self.completed_epochs: int = 0

    def available_domains(self) -> List[str]:
        return list(self.streams.keys())

    def total_train_tokens(self) -> int:
        return sum(self.domain_token_counts.values())

    def total_windows_per_epoch(self) -> int:
        return int(len(self.pool_starts))

    def steps_per_epoch(self, batch_size: int, grad_accum: int) -> int:
        windows_per_step = max(1, batch_size * grad_accum)
        return max(1, int(math.ceil(len(self.pool_starts) / windows_per_step)))

    def sample_full_pass_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, float]:
        """从全量语料池无放回洗牌游标中顺序抽取下一个 Batch，返回 (x, y, loss_mask, 主导领域名, 当前已遍历 Epoch 进度)"""
        n_total = len(self.pool_starts)
        if self.pool_cursor + batch_size > n_total:
            self.completed_epochs += 1
            self.pool_perm = np.random.permutation(n_total)
            self.pool_cursor = 0

        idx_slice = self.pool_perm[self.pool_cursor : self.pool_cursor + batch_size]
        self.pool_cursor += batch_size

        d_ids = self.pool_dom_ids[idx_slice]
        starts = self.pool_starts[idx_slice]
        rows = []
        for d_i, st in zip(d_ids.tolist(), starts.tolist()):
            d_name = self.domain_names[int(d_i)]
            mm = self.streams[d_name]["train"]
            rows.append(mm[int(st) : int(st) + self.seq_len + 1])

        buf = np.stack(rows).astype(np.int64)
        tensor = torch.from_numpy(buf).to(device, non_blocking=True)
        x = tensor[:, :-1].contiguous()
        y = tensor[:, 1:].contiguous()
        loss_mask = self.build_cot_loss_mask(x, target_ids=y)

        vals, counts = np.unique(d_ids, return_counts=True)
        dom_lead = self.domain_names[int(vals[int(np.argmax(counts))])]
        epoch_progress = float(self.completed_epochs) + float(self.pool_cursor) / max(1, n_total)
        return x, y, loss_mask, dom_lead, epoch_progress

    @staticmethod
    def build_cot_loss_mask(
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        prompt_weight: float = 0.02,
        think_weight: float = 1.1,
        answer_weight: float = 1.6,
    ) -> torch.Tensor:
        """
        构造纯自然语言三段式 CoT 损失掩码:
        - 若序列不含 <|user|> / <|think|> / <|/think|> (即普通非 CoT 自然长文章、书籍、代码文档)，直接返回全 1.0 均匀掩码；
        - 若序列含对话或思维链标记，对提问前缀赋予低权重 (prompt_weight)，对思考段与最终回答赋予高权重。
        """
        B, T = input_ids.shape
        dev = input_ids.device
        mask = torch.ones((B, T), dtype=torch.float32, device=dev)
        has_cot = (
            (input_ids == USER_ID)
            | (input_ids == THINK_BOS_ID)
            | (input_ids == THINK_EOS_ID)
        )
        if target_ids is not None:
            has_cot = has_cot | (
                (target_ids == USER_ID)
                | (target_ids == THINK_BOS_ID)
                | (target_ids == THINK_EOS_ID)
            )
        if not bool(has_cot.any().item()):
            return mask

        pos_idx = torch.arange(1, T + 1, device=dev, dtype=torch.int64).unsqueeze(0).expand(B, T)
        u_pos = torch.where(input_ids == USER_ID, pos_idx, torch.zeros_like(pos_idx)).cummax(dim=1).values
        t_pos = torch.where(input_ids == THINK_BOS_ID, pos_idx, torch.zeros_like(pos_idx)).cummax(dim=1).values
        a_pos = torch.where(input_ids == THINK_EOS_ID, pos_idx, torch.zeros_like(pos_idx)).cummax(dim=1).values
        e_pos = torch.where(
            (input_ids == 1) | (input_ids == 2), pos_idx, torch.zeros_like(pos_idx)
        ).cummax(dim=1).values

        in_prompt = (u_pos > e_pos) & (u_pos >= t_pos) & (u_pos >= a_pos)
        in_think = (t_pos > e_pos) & (t_pos > u_pos) & (t_pos >= a_pos)
        in_answer = (a_pos > e_pos) & (a_pos >= t_pos) & (a_pos > u_pos)

        mask = torch.where(in_prompt, torch.full_like(mask, prompt_weight), mask)
        mask = torch.where(in_think, torch.full_like(mask, think_weight), mask)
        mask = torch.where(in_answer, torch.full_like(mask, answer_weight), mask)

        # 对齐自回归目标位 target_ids: 当 input_ids 位于提问末尾而下一步需预测并发射 <|think|> 时，
        # 赋予完整 think_weight (1.1) 而非 prompt_weight (0.02)，避免 <|think|> 触发信号被压制 55 倍；
        # 同理，预测 <|/think|> 的状态切换位赋予完整 answer_weight (1.6)
        if target_ids is not None:
            mask = torch.where(target_ids == USER_ID, torch.full_like(mask, prompt_weight), mask)
            mask = torch.where(target_ids == THINK_BOS_ID, torch.full_like(mask, think_weight), mask)
            mask = torch.where(target_ids == THINK_EOS_ID, torch.full_like(mask, answer_weight), mask)
            mask = torch.where(in_answer & (target_ids == 2), torch.full_like(mask, answer_weight), mask)
        return mask

    def sample_batch(
        self,
        domain: str,
        split: str,
        batch_size: int,
        device: torch.device,
        sequential_continuation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dom = domain if domain in self.streams else "general"
        data = self.streams[dom][split]
        max_start = len(data) - self.seq_len - 2

        doc_anchors = self.doc_starts.get(dom, {}).get(split)
        if doc_anchors is not None and doc_anchors.size > 0:
            # 对 cot_distill 优先从完整文档起点 (<|bos|>) 对齐采样 (85% 概率)，
            # 确保 <|user|>问题 -> <|think|>思考 -> <|/think|>回答 因果链完整闭合，杜绝无头半截思考链污染
            picked_idx = np.random.randint(0, doc_anchors.size, size=(batch_size,))
            starts = doc_anchors[picked_idx].copy()
            rand_mask = np.random.rand(batch_size) < 0.15
            if np.any(rand_mask):
                starts[rand_mask] = np.random.randint(0, max_start, size=(int(np.sum(rand_mask)),))
        elif sequential_continuation and split == "train":
            cur_key = f"{dom}_{batch_size}"
            if cur_key not in self.seq_cursors:
                self.seq_cursors[cur_key] = np.random.randint(0, max_start, size=(batch_size,))
            starts = self.seq_cursors[cur_key]
            # 推进游标至下一个相邻窗口以模拟跨块超长 CoT 连续轨迹
            next_starts = starts + self.seq_len
            wrap_mask = next_starts >= max_start
            if np.any(wrap_mask):
                next_starts[wrap_mask] = np.random.randint(0, max_start, size=(int(np.sum(wrap_mask)),))
            self.seq_cursors[cur_key] = next_starts
        else:
            starts = np.random.randint(0, max_start, size=(batch_size,))

        buf = np.stack([data[i : i + self.seq_len + 1] for i in starts]).astype(np.int64)
        tensor = torch.from_numpy(buf).to(device, non_blocking=True)
        x = tensor[:, :-1].contiguous()
        y = tensor[:, 1:].contiguous()
        loss_mask = self.build_cot_loss_mask(x, target_ids=y)
        return x, y, loss_mask


# ==============================================================================
# 2. 零人工干预的 .bkp 森林自治生命周期控制器 (AutoBKPForestController)
# ==============================================================================
class AutoBKPForestController:
    """
    从唯一一颗 64x64 种子知识包 (Seed_0.bkp) 起步，根据训练期间的实时数学遥测自主触发四大拓扑算子:
    1. Auto-Expand (K_p -> 2*K_p): 唤醒率超阈值时原位四象限无损扩容 (误差 < 1e-7)
    2. Auto-Spawn: 从低命中未知球面查询流形孵化新 64x64 专家种子包
    3. Auto-Split: 二分图拉普拉斯谱双聚类检测出双连通分量时自发裂变为两颗独立子包
    4. Auto-Merge: 跨包同 Token 共激活率超阈值时沿主对角线合并并开辟 2ab 交叉耦合象限
    """

    def __init__(
        self,
        bkp_mgr: BKPManager,
        forest_dir: Path,
        max_grid_k: int = 128,
        expand_woken_thresh: Optional[float] = None,
        split_cross_thresh: Optional[float] = None,
        merge_coact_thresh: Optional[float] = None,
    ):
        self.bkp_mgr = bkp_mgr
        self.forest_dir = forest_dir
        self.forest_dir.mkdir(parents=True, exist_ok=True)
        self.max_grid_k = max_grid_k
        # 零超参数闭式球面几何阈值:
        # 1. Auto-Expand 唤醒饱和比: 1 - exp(-1/sqrt(2)) ≈ 0.5069 (或由瑞利多义干涉判据触发)
        # 2. Auto-Split 谱双聚类跨簇比: 1 - 1/sqrt(2) ≈ 0.2929 (2-测地线正交补集残差界)
        # 3. Auto-Merge 跨包共激活比: sqrt(2) - 1 ≈ 0.4142 (白银分割共现阈值)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.expand_woken_thresh = (
            float(expand_woken_thresh) if expand_woken_thresh is not None else (1.0 - math.exp(-inv_sqrt2))
        )
        self.split_cross_thresh = (
            float(split_cross_thresh) if split_cross_thresh is not None else (1.0 - inv_sqrt2)
        )
        self.merge_coact_thresh = (
            float(merge_coact_thresh) if merge_coact_thresh is not None else (math.sqrt(2.0) - 1.0)
        )

        self.lineage_events: List[Dict] = []
        self.spawn_counter: int = 0
        self.split_counter: int = 0
        self.merge_counter: int = 0
        self.closure_counter: int = 0
        self._already_split_children: set = set()

    def _persist_pkg_to_forest(self, pkg: BKPPackage, filename: str) -> Path:
        """通过 BKPPackage.save_package 统一封装落盘 (对上层隐藏底层 .meta.pt 实现细节)"""
        bkp_path = self.forest_dir / filename
        pkg.save_package(str(bkp_path))
        return bkp_path

    def snapshot_forest_state(self, step: int) -> List[Dict]:
        """采集当前挂载森林中每颗 .bkp 的拓扑与健康度快照"""
        snap = []
        for pid in list(self.bkp_mgr.l1_active_pkg_ids):
            mod_k = self.bkp_mgr._mod_key(pid)
            if mod_k not in self.bkp_mgr.packages:
                continue
            pkg = cast(BKPPackage, self.bkp_mgr.packages[mod_k])
            woken_mask = (pkg.activation_counts > 0) & (pkg.scales[:, 1] > 1e-4)
            n_woken = int(woken_mask.sum().item())
            woken_ratio = n_woken / max(1, pkg.num_experts)
            mean_purity = (
                float(pkg.purity_r[woken_mask].mean().item()) if n_woken > 0 else float(pkg.purity_r.mean().item())
            )
            snap.append({
                "step": step,
                "pkg_id": pkg.pkg_id,
                "grid_k": pkg.grid_k,
                "num_experts": pkg.num_experts,
                "size_mb": round(pkg.num_experts * 4096 / (1024 * 1024), 1),
                "woken_experts": n_woken,
                "woken_ratio_pct": round(100.0 * woken_ratio, 2),
                "mean_purity_r": round(mean_purity, 4),
                "quality_q": round(float(pkg.quality_q.item()), 4),
                "is_base": pkg.is_base_pkg,
            })
        return snap

    @torch.no_grad()
    def step_autonomous_evolution(
        self,
        step: int,
        active_domain_hint: str,
        model: BicameraKernel,
        device: torch.device,
        probe_x: torch.Tensor,
    ) -> List[Dict]:
        """在拓扑巡检步自动评估并执行 Split / Expand / Merge / Spawn，同时在 FP32 下精确测量零步前向误差"""
        events_this_step: List[Dict] = []
        model.eval()
        self.bkp_mgr.eval()
        # 在 FP32 下提取固定探针的腰部查询向量 q_probe，消除 BF16 TensorCore Tile 切换的 1-ULP 舍入伪差
        with torch.autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=False):
            h_probe = model.tok_embeddings(probe_x)
            for l_idx in range(model.config.bkp_bus_layer):
                h_probe, _, _ = model.layers[l_idx](h_probe, rope=model.rope)
            q_probe = model.bkp_bus.project_to_bkp_space(h_probe.float())

        # 法则 1: Auto-Split (二分图拉普拉斯谱双聚类 + 2ab 跨簇突触无损桥接裂变 P -> P_A + P_B + Bridge_AB，无包数量上限!)
        if len(events_this_step) == 0:
            for pid in list(self.bkp_mgr.l1_active_pkg_ids):
                if pid in self._already_split_children:
                    continue
                pkg = cast(BKPPackage, self.bkp_mgr.packages[self.bkp_mgr._mod_key(pid)])
                if pkg.is_bridge_pkg or pkg.grid_k < 128:
                    continue
                k_cur = pkg.grid_k
                woken_2d = (pkg.activation_counts.view(k_cur, k_cur) > 0) & (pkg.scales[:, 1].view(k_cur, k_cur) > 2e-5)
                woken_tl = int(woken_2d[: k_cur // 2, : k_cur // 2].sum().item())
                woken_br = int(woken_2d[k_cur // 2 :, k_cur // 2 :].sum().item())
                if woken_tl < 128 or woken_br < 128:
                    continue
                self.bkp_mgr.vram_cache.clear()
                z_before = self.bkp_mgr.query_and_inject(q_probe).float()
                split_res = pkg.bkp_split_spectral(cross_ratio_threshold=self.split_cross_thresh, return_bridge=True)
                if split_res is not None:
                    sub_a, sub_b, sub_bridge = split_res
                    self.split_counter += 1
                    base_stem = pkg.pkg_id.replace(".bkp", "")
                    sub_a.pkg_id = f"{base_stem}_SubA{self.split_counter}.bkp"
                    sub_b.pkg_id = f"{base_stem}_SubB{self.split_counter}.bkp"
                    self._already_split_children.add(sub_a.pkg_id)
                    self._already_split_children.add(sub_b.pkg_id)

                    self.bkp_mgr.unmount_package(pid)
                    self.bkp_mgr.mount_package(sub_a, pin_to_l1=True)
                    self.bkp_mgr.mount_package(sub_b, pin_to_l1=True)
                    child_names = [sub_a.pkg_id, sub_b.pkg_id]
                    child_grids = [f"{sub_a.grid_k}x{sub_a.grid_k}", f"{sub_b.grid_k}x{sub_b.grid_k}"]
                    if sub_bridge is not None:
                        sub_bridge.pkg_id = f"{base_stem}_BridgeAB{self.split_counter}.bkp"
                        self._already_split_children.add(sub_bridge.pkg_id)
                        self.bkp_mgr.mount_package(sub_bridge, pin_to_l1=True)
                        child_names.append(sub_bridge.pkg_id)
                        child_grids.append(f"{sub_bridge.grid_k}x{sub_bridge.grid_k}(Bridge)")

                    self.bkp_mgr.vram_cache.clear()
                    z_after = self.bkp_mgr.query_and_inject(q_probe).float()
                    zero_err = float((z_before - z_after).abs().max().item())

                    unlocked_a = sub_a.enable_empty_pages_for_training(init_scale_v=1e-5, freeze_existing_pages=False)
                    unlocked_b = sub_b.enable_empty_pages_for_training(init_scale_v=1e-5, freeze_existing_pages=False)
                    unlocked_list = [unlocked_a, unlocked_b]
                    self._persist_pkg_to_forest(sub_a, sub_a.pkg_id)
                    self._persist_pkg_to_forest(sub_b, sub_b.pkg_id)
                    if sub_bridge is not None:
                        unlocked_br = sub_bridge.enable_empty_pages_for_training(
                            init_scale_v=1e-5, freeze_existing_pages=False
                        )
                        unlocked_list.append(unlocked_br)
                        self._persist_pkg_to_forest(sub_bridge, sub_bridge.pkg_id)

                    ev = {
                        "step": step,
                        "op": "Auto-Split",
                        "parent_pkg": pid,
                        "children": child_names,
                        "child_grids": child_grids,
                        "unlocked_pages": unlocked_list,
                        "zero_step_max_diff": zero_err,
                    }
                    self.lineage_events.append(ev)
                    events_this_step.append(ev)
                    br_str = f" + {sub_bridge.pkg_id} (64^2 跨簇桥接)" if sub_bridge is not None else ""
                    print(
                        f"  [AutoForest @ Step {step}] Auto-Split: 混杂包 {pid} 无损桥接裂变 -> "
                        f"{sub_a.pkg_id} ({sub_a.grid_k}^2) + {sub_b.pkg_id} ({sub_b.grid_k}^2){br_str} | "
                        f"零步误差={zero_err:.3e}",
                        flush=True,
                    )
                    break

        # 法则 2: Auto-Expand (四象限翻倍扩容 K_p -> 2*K_p，由闭式饱和率或瑞利多义干涉判据触发)
        if len(events_this_step) == 0:
            for pid in list(self.bkp_mgr.l1_active_pkg_ids):
                mod_k = self.bkp_mgr._mod_key(pid)
                if mod_k not in self.bkp_mgr.packages:
                    continue
                pkg = cast(BKPPackage, self.bkp_mgr.packages[mod_k])
                if pkg.is_bridge_pkg:
                    continue
                woken_mask = (pkg.activation_counts > 0) & (pkg.scales[:, 1] > 2e-5)
                woken_ratio = float(woken_mask.float().mean().item())
                polysemy_mask = pkg.detect_polysemy_interference_mask()
                polysemy_count = int(polysemy_mask.sum().item())
                polysemy_trigger = polysemy_count >= max(1, pkg.grid_k)
                if pkg.grid_k < self.max_grid_k and (woken_ratio >= self.expand_woken_thresh or polysemy_trigger):
                    old_k = pkg.grid_k
                    z_before = self.bkp_mgr.query_and_inject(q_probe).float()
                    pkg.expand_quadrant_2x()
                    self.bkp_mgr.vram_cache.clear()
                    z_after = self.bkp_mgr.query_and_inject(q_probe).float()
                    zero_err = float((z_before - z_after).abs().max().item())

                    unlocked = pkg.enable_empty_pages_for_training(
                        init_scale_v=1e-5, freeze_existing_pages=False, sparse_cross_bridge_only=True
                    )
                    self._persist_pkg_to_forest(pkg, pkg.pkg_id)
                    self._already_split_children.discard(pkg.pkg_id)
                    ev = {
                        "step": step,
                        "op": "Auto-Expand",
                        "pkg_id": pkg.pkg_id,
                        "transition": f"{old_k}x{old_k} -> {pkg.grid_k}x{pkg.grid_k}",
                        "trigger_woken_ratio_pct": round(100.0 * woken_ratio, 2),
                        "unlocked_new_pages": unlocked,
                        "zero_step_max_diff": zero_err,
                    }
                    self.lineage_events.append(ev)
                    events_this_step.append(ev)
                    print(
                        f"  [AutoForest @ Step {step}] Auto-Expand: {pkg.pkg_id} ({old_k}^2 -> {pkg.grid_k}^2) | "
                        f"触发唤醒率={woken_ratio*100:.1f}% | 解锁新页={unlocked:,} | 零步误差={zero_err:.3e}",
                        flush=True,
                    )
                    break

        # 法则 3: Auto-Merge (高共现主对角线聚变 P_A + P_B -> P_AB，所有包平权参与)
        if len(events_this_step) == 0 and len(self.bkp_mgr.l1_active_pkg_ids) >= 3:
            best_pair = None
            best_coact_ratio = 0.0
            for (pa, pb), co_cnt in self.bkp_mgr.pkg_coactivation_counts.items():
                if pa not in self.bkp_mgr.l1_active_pkg_ids or pb not in self.bkp_mgr.l1_active_pkg_ids:
                    continue
                pkg_a = cast(BKPPackage, self.bkp_mgr.packages[self.bkp_mgr._mod_key(pa)])
                pkg_b = cast(BKPPackage, self.bkp_mgr.packages[self.bkp_mgr._mod_key(pb)])
                if pkg_a.is_bridge_pkg or pkg_b.is_bridge_pkg:
                    continue
                if pkg_a.grid_k + pkg_b.grid_k > self.max_grid_k:
                    continue
                denom = min(
                    self.bkp_mgr.pkg_token_hit_counts.get(pa, 1.0),
                    self.bkp_mgr.pkg_token_hit_counts.get(pb, 1.0),
                )
                if denom < 32.0:
                    continue
                ratio = co_cnt / max(1.0, denom)
                if ratio > best_coact_ratio:
                    best_coact_ratio = ratio
                    best_pair = (pkg_a, pkg_b)

            if best_pair is not None and best_coact_ratio >= self.merge_coact_thresh:
                pkg_a, pkg_b = best_pair
                self.merge_counter += 1
                merged_id = f"Syn_Merged_{self.merge_counter}.bkp"
                z_before = self.bkp_mgr.query_and_inject(q_probe).float()

                merged_pkg = BKPPackage.bkp_merge_triadic_bundle(pkg_a, pkg_b, merged_id=merged_id)
                self.bkp_mgr.unmount_package(pkg_a.pkg_id)
                self.bkp_mgr.unmount_package(pkg_b.pkg_id)
                self.bkp_mgr.mount_package(merged_pkg, pin_to_l1=True)
                self.bkp_mgr.vram_cache.clear()

                z_after = self.bkp_mgr.query_and_inject(q_probe).float()
                zero_err = float((z_before - z_after).abs().max().item())

                unlocked_2ab = merged_pkg.enable_empty_pages_for_training(
                    init_scale_v=1e-5, freeze_existing_pages=True, sparse_cross_bridge_only=True
                )
                self._persist_pkg_to_forest(merged_pkg, merged_id)
                ev = {
                    "step": step,
                    "op": "Auto-Merge",
                    "parents": [pkg_a.pkg_id, pkg_b.pkg_id],
                    "merged_pkg": merged_id,
                    "grid": f"{merged_pkg.grid_k}x{merged_pkg.grid_k}",
                    "trigger_coact_ratio_pct": round(100.0 * best_coact_ratio, 2),
                    "unlocked_2ab_pages": unlocked_2ab,
                    "zero_step_max_diff": zero_err,
                }
                self.lineage_events.append(ev)
                events_this_step.append(ev)
                print(
                    f"  [AutoForest @ Step {step}] Auto-Merge: {pkg_a.pkg_id} + {pkg_b.pkg_id} -> {merged_id} "
                    f"({merged_pkg.grid_k}^2) | 共激活率={best_coact_ratio*100:.1f}% | 解锁2ab交叉页={unlocked_2ab:,} | 零步误差={zero_err:.3e}",
                    flush=True,
                )

        # 法则 4: Auto-Spawn (球面未知流形孵化新 64x64 专家种子包，闭式冈贝尔极值尾概率触发)
        if len(events_this_step) == 0:
            periodic_total = max(1, self.bkp_mgr._total_periodic_queries)
            periodic_low_ratio = self.bkp_mgr._low_hit_periodic_queries / periodic_total
            tau_gumbel_64 = closed_form_gumbel_threshold(64)
            closed_form_tail_prob = 0.5 * math.erfc(tau_gumbel_64 / math.sqrt(2.0))
            has_low_hit_pool = (
                self.bkp_mgr._low_hit_q1_reservoir is not None
                and self.bkp_mgr._low_hit_q1_reservoir.shape[0] >= 64
                and periodic_low_ratio >= closed_form_tail_prob
            )
            if has_low_hit_pool or len(self.bkp_mgr.l1_active_pkg_ids) < 3:
                self.spawn_counter += 1
                new_id = f"Spec_{active_domain_hint.capitalize()}_{self.spawn_counter}.bkp"
                mmap_p = str(self.forest_dir / new_id)
                z_before = self.bkp_mgr.query_and_inject(q_probe).float()

                spawned_pkg = self.bkp_mgr.spawn_specialist_package_from_queries(
                    new_pkg_id=new_id,
                    grid_k=64,
                    mmap_file_path=mmap_p,
                    mount_mode=BKPMountMode.RW_SHARED,
                    init_scale_v=1e-5,
                )
                self._persist_pkg_to_forest(spawned_pkg, spawned_pkg.pkg_id)
                z_after = self.bkp_mgr.query_and_inject(q_probe).float()
                zero_err = float((z_before - z_after).abs().max().item())

                ev = {
                    "step": step,
                    "op": "Auto-Spawn",
                    "spawned_pkg": spawned_pkg.pkg_id,
                    "domain_hint": active_domain_hint,
                    "grid": "64x64",
                    "periodic_low_hit_ratio_pct": round(100.0 * periodic_low_ratio, 2),
                    "zero_step_max_diff": zero_err,
                }
                self.lineage_events.append(ev)
                events_this_step.append(ev)
                print(
                    f"  [AutoForest @ Step {step}] Auto-Spawn: 从球面查询流形孵化新种子包 {new_id} (64x64, 16MB) | "
                    f"周期低命中率={periodic_low_ratio*100:.2f}% | 零步挂载误差={zero_err:.3e}",
                    flush=True,
                )

        # 法则 5: Auto-Closure (双正交跨簇笛卡尔流形闭包成核 R^1024 (+) R^1024，由已知流形预合成未知交叉格点并在真实语境下自演进)
        total_closure_nucleated = 0
        closure_details = []
        for pid in list(self.bkp_mgr.l1_active_pkg_ids):
            mod_k = self.bkp_mgr._mod_key(pid)
            if mod_k not in self.bkp_mgr.packages:
                continue
            pkg = cast(BKPPackage, self.bkp_mgr.packages[mod_k])
            if pkg.is_bridge_pkg or pkg.mount_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
                continue
            res_cl = pkg.crystallize_biorthogonal_manifold_closure(
                min_mature_scale_v=5e-5,
                closure_scale_ratio=0.25,
                max_crystallize_slots=256,
            )
            n_nuc = int(res_cl.get("crystallized_count", 0))
            if n_nuc > 0:
                self._persist_pkg_to_forest(pkg, pkg.pkg_id)
                total_closure_nucleated += n_nuc
                closure_details.append({
                    "pkg_id": pkg.pkg_id,
                    "nucleated_pages": n_nuc,
                    "mature_before": int(res_cl.get("mature_before", 0)),
                    "mature_after": int(res_cl.get("mature_after", 0)),
                    "mean_operator_offdiag_cos": round(float(res_cl.get("mean_operator_offdiag_cos", 0.0)), 4),
                })

        if total_closure_nucleated > 0:
            self.closure_counter += 1
            self.bkp_mgr.vram_cache.clear()
            ev_cl = {
                "step": step,
                "op": "Auto-Closure",
                "total_nucleated_pages": total_closure_nucleated,
                "packages": closure_details,
            }
            self.lineage_events.append(ev_cl)
            events_this_step.append(ev_cl)
            pkg_summary = ", ".join(
                f"{d['pkg_id']}(+{d['nucleated_pages']}p, offDiagCos={d['mean_operator_offdiag_cos']:.3f})"
                for d in closure_details
            )
            print(
                f"  [AutoForest @ Step {step}] Auto-Closure: 双正交笛卡尔闭包成核 +{total_closure_nucleated:,} 页 | {pkg_summary}",
                flush=True,
            )

        # 法则 6: Auto-Mitosis (切向复共轭叶状分裂: 将瑞利多义冲突微专家沿 ±u_perp 复共轭分裂至同行伴随空槽，零正交梯度衰减!)
        total_mitosis_pages = 0
        mitosis_details = []
        for pid in list(self.bkp_mgr.l1_active_pkg_ids):
            mod_k = self.bkp_mgr._mod_key(pid)
            if mod_k not in self.bkp_mgr.packages:
                continue
            pkg = cast(BKPPackage, self.bkp_mgr.packages[mod_k])
            if pkg.is_bridge_pkg or pkg.mount_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
                continue
            res_mit = pkg.nucleate_tangential_conjugate_mitosis(max_mitosis_slots=256)
            n_mit = int(res_mit.get("mitosis_count", 0))
            if n_mit > 0:
                self._persist_pkg_to_forest(pkg, pkg.pkg_id)
                total_mitosis_pages += n_mit
                mitosis_details.append({
                    "pkg_id": pkg.pkg_id,
                    "mitosis_pages": n_mit,
                    "mean_conjugate_cos": round(float(res_mit.get("mean_conjugate_cos", 0.0)), 4),
                })

        if total_mitosis_pages > 0:
            self.bkp_mgr.vram_cache.clear()
            ev_mit = {
                "step": step,
                "op": "Auto-Mitosis",
                "total_mitosis_pages": total_mitosis_pages,
                "packages": mitosis_details,
            }
            self.lineage_events.append(ev_mit)
            events_this_step.append(ev_mit)
            mit_summary = ", ".join(
                f"{d['pkg_id']}(+{d['mitosis_pages']}p, conjCos={d['mean_conjugate_cos']:.2f})"
                for d in mitosis_details
            )
            print(
                f"  [AutoForest @ Step {step}] Auto-Mitosis: 切向复共轭叶状分裂 +{total_mitosis_pages:,} 页 | {mit_summary}",
                flush=True,
            )

        self.bkp_mgr.reset_coactivation_stats()
        self.bkp_mgr.train()
        model.train()
        return events_this_step


# ==============================================================================
# 3. 领域验证集困惑度与路由遥测评估器
# ==============================================================================
@torch.no_grad()
def eval_domain_ppl(
    model: BicameraKernel,
    bkp_mgr: Optional[BKPManager],
    dataset: MultiStreamCoTDataset,
    domain: str,
    batch_size: int,
    device: torch.device,
    n_batches: int = 8,
    seed: int = 2026,
) -> Dict:
    model.eval()
    if bkp_mgr is not None:
        bkp_mgr.eval()
    state = np.random.get_state()
    np.random.seed(seed)

    total_nll = 0.0
    total_slots = 0.0
    total_uniq_seq = 0.0
    total_hop_jacc = 0.0
    pkg_weights: Dict[str, float] = {}

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    for _ in range(n_batches):
        x, y, loss_mask = dataset.sample_batch(domain, "val", batch_size, device, sequential_continuation=False)
        with amp_ctx:
            nll, _, _ = model(
                x, bkp_manager=bkp_mgr, bkp_dropout=(bkp_mgr is None), target_ids=y, loss_mask=loss_mask
            )
        total_nll += float(nll.item())
        if bkp_mgr is not None:
            total_slots += bkp_mgr.last_avg_active_slots
            total_uniq_seq += bkp_mgr.last_unique_experts_per_seq
            total_hop_jacc += bkp_mgr.last_hop_jaccard_dist
            for pid, w_val in bkp_mgr.last_pkg_weight_sum.items():
                pkg_weights[pid] = pkg_weights.get(pid, 0.0) + w_val

    np.random.set_state(state)
    avg_nll = total_nll / max(1, n_batches)
    avg_slots = total_slots / max(1, n_batches)
    avg_uniq = total_uniq_seq / max(1, n_batches)
    avg_jacc = total_hop_jacc / max(1, n_batches)
    w_tot = max(1e-9, sum(pkg_weights.values()))
    pkg_share_pct = {k: round(100.0 * v / w_tot, 2) for k, v in pkg_weights.items()}

    return {
        "nll": round(avg_nll, 4),
        "ppl": round(math.exp(min(15.0, avg_nll)), 2),
        "avg_active_slots": round(avg_slots, 2),
        "unique_experts_per_seq": round(avg_uniq, 1),
        "hop_jaccard_dist": round(avg_jacc, 4),
        "pkg_share_pct": pkg_share_pct,
    }


@torch.no_grad()
def extract_waist_query_and_standardized(
    model: BicameraKernel,
    bkp_mgr: BKPManager,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    提取模型在腰部总线 (Layer 0..bkp_bus_layer-1) 生成的原始查询向量 q_raw (N, 2048)
    以及经参考流形球面化度规标准化后的子查询 q1, q2 (N, 8, 128)，全程保持 float32 精度。
    """
    was_training = model.training
    model.eval()
    B, T = input_ids.shape
    h = model.tok_embeddings(input_ids)
    for l_idx in range(model.config.bkp_bus_layer):
        h, _, _ = model.layers[l_idx](h, rope=model.rope)
    q_raw = model.bkp_bus.project_to_bkp_space(h).reshape(B * T, D_BKP).float()
    with torch.autocast(device_type=input_ids.device.type if input_ids.device.type == "cuda" else "cpu", enabled=False):
        q1, q2 = bkp_mgr._standardize_subqueries(q_raw, update_center=False)
    if was_training:
        model.train()
    return q_raw.float(), q1.float(), q2.float()


# ==============================================================================
# 4. 从 0 到 1 自治 .bkp 森林生长 + 多跳 CoT 蒸馏主入口 (单阶段端到端无人工干预训练)
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Bicamera 从 0 到 1 自治 .bkp 森林生长 + 多跳 CoT 蒸馏与经典基准评测")
    ap.add_argument("--workspace", type=str, default=str(Path(__file__).resolve().parent / "workspace"))
    ap.add_argument("--run-name", type=str, default="micro_125m_autoforest_v29")
    ap.add_argument("--total-steps", type=int, default=1200)
    ap.add_argument("--evolve-interval", type=int, default=120)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed-grid-k", type=int, default=64)
    ap.add_argument("--max-grid-k", type=int, default=128)
    ap.add_argument("--lr-kernel", type=float, default=6e-4)
    ap.add_argument("--lr-bkp", type=float, default=0.10)
    ap.add_argument("--fatigue-beta", type=float, default=0.20)
    ap.add_argument("--collinear-lock", type=float, default=0.35, help="成熟共激活微专家的共线主轴保护与前向残差分流系数")
    ap.add_argument("--closure-softness", type=float, default=0.72, help="双正交闭包微专家的未知语境自适应测地圆软化系数")
    ap.add_argument(
        "--num-epochs",
        type=float,
        default=float(os.environ.get("AF_EPOCHS", "0.0")),
        help="若 > 0 (或 total_steps <= 0)，则不硬约束步数，按全量语料池完整遍历指定遍数 (例如 2.0 遍) 自动计算步数并在收敛时停止",
    )
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ws = Path(args.workspace)
    log_dir = ws / "logs" / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    forest_dir = ws / "bkp_store_forest" / args.run_name
    if forest_dir.exists():
        shutil.rmtree(forest_dir)
    forest_dir.mkdir(parents=True, exist_ok=True)

    tok_path = ws / "tokenizer" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    dataset = MultiStreamCoTDataset(ws, seq_len=args.seq_len)
    avail_domains = dataset.available_domains()
    steps_per_ep = dataset.steps_per_epoch(args.batch_size, args.grad_accum)
    if args.num_epochs > 0.0 or args.total_steps <= 0:
        target_epochs = args.num_epochs if args.num_epochs > 0.0 else 2.0
        args.total_steps = max(1, int(round(steps_per_ep * target_epochs)))

    print("=" * 88)
    print(f"Bicamera v27 从 0 到 1 自治 .bkp 森林演化 + 全量语料池自然遍历引擎 | Device: {device}")
    print(
        f"  - Run Name: {args.run_name} | Total Steps: {args.total_steps} "
        f"(1 Epoch = {steps_per_ep:,} steps / {dataset.total_train_tokens():,} tokens) | SeqLen: {args.seq_len}"
    )
    print(f"  - 初始状态: 随机初始化 125M Micro Kernel + 单颗 {args.seed_grid_k}x{args.seed_grid_k} Seed_0.bkp (16MB)")
    print(f"  - 全量遍历流: {dataset.domain_token_counts} | 轨迹驻留疲劳系数 fatigue_beta={args.fatigue_beta}")
    print("=" * 88, flush=True)

    # 1. 从零随机初始化 125M Micro Kernel (bkp_window_k=1 无近视滑动窗口，全图直通 O(1) 差分注意力)
    cfg = BicameraConfig.preset_micro_125m(vocab_size=tok.get_vocab_size())
    model = BicameraKernel(cfg).to(device)
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. 初始化 BKPManager 并在空文件夹中种下唯一一颗 64x64 种子知识包 Seed_0.bkp (不设包数量上限)
    # 全部路由、测地闭包软化与共线锁定均采用 (d, N_grid, H_m, R_m) 零超参数闭式物理定律
    routing_cfg = BKPRoutingConfig(
        alpha_0=1.0,
        head_top_k=4,
        global_top_k_max=16,
        geodesic_closure_soft_gamma=1.0 / math.sqrt(2.0),
        geodesic_closure_damping=1.0 / math.sqrt(3.0),
        l1_tlb_max_packages=64,
        vram_cache_slots=65536 if device.type == "cuda" else 4096,
    )
    bkp_mgr = BKPManager(config=routing_cfg, device=device).to(device)
    bkp_mgr.trajectory_fatigue_beta = args.fatigue_beta
    # 置为 None 以激活球面 James-Stein 闭式最优逐专家共线锁定比 gamma_m^* = max(0, (H_m R_m^2 - 1)/(H_m - 1))
    bkp_mgr.collinear_core_lock_strength = None

    seed_bkp_path = forest_dir / "Seed_0.bkp"
    seed_pkg = BKPPackage(
        pkg_id="Seed_0.bkp",
        grid_k=args.seed_grid_k,
        mmap_file_path=str(seed_bkp_path),
        mount_mode=BKPMountMode.RW_SHARED,
        is_base_pkg=True,
        device=device,
    )
    with torch.no_grad():
        n_seed = seed_pkg.num_experts
        all_eids = torch.arange(n_seed, dtype=torch.long, device=device)
        u_init = F.normalize(torch.randn((n_seed, D_BKP), dtype=torch.float32, device=device), dim=-1) * math.sqrt(D_BKP)
        v_init = F.normalize(torch.randn((n_seed, D_BKP), dtype=torch.float32, device=device), dim=-1) * (
            0.025 * math.sqrt(D_BKP)
        )
        seed_pkg.write_experts_to_mmap_pages(all_eids, u_init, v_init)
    bkp_mgr.mount_package(seed_pkg, pin_to_l1=True)

    controller = AutoBKPForestController(
        bkp_mgr=bkp_mgr,
        forest_dir=forest_dir,
        max_grid_k=args.max_grid_k,
    )
    controller._persist_pkg_to_forest(seed_pkg, seed_pkg.pkg_id)
    controller.lineage_events.append({
        "step": 0,
        "op": "Genesis-Seed",
        "pkg_id": "Seed_0.bkp",
        "grid": f"{args.seed_grid_k}x{args.seed_grid_k}",
        "num_experts": seed_pkg.num_experts,
        "size_mb": round(seed_pkg.num_experts * 4096 / (1024 * 1024), 1),
    })

    # 固定一组验证探针用于测量每次自治拓扑算子触发瞬间的零步前向误差
    probe_x, _, _ = dataset.sample_batch("general", "val", min(4, args.batch_size), device)

    # 前段提问器 FFN 阻尼分组: 压制前段 Layer 0..5 的单输入 MoE 私藏静态百科，逼迫知识卸载至腰部 .bkp 与后段双线性元算子
    prefix_moe_params = []
    core_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_prefix_moe = any(
            name.startswith(f"layers.{l_i}.moe.") for l_i in range(model.config.bkp_bus_layer)
        )
        if is_prefix_moe:
            prefix_moe_params.append(p)
        else:
            core_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": core_params, "lr": args.lr_kernel, "lr_scale": 1.0},
            {"params": prefix_moe_params, "lr": args.lr_kernel * 0.35, "lr_scale": 0.35},
        ],
        lr=args.lr_kernel,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    forest_snapshots = [controller.snapshot_forest_state(0)]
    eval_history = []
    evolve_cutoff = int(args.total_steps * 0.70)
    effective_evolve_interval = max(1, max(args.evolve_interval, args.total_steps // 10))

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    t_start = time.time()
    tokens_processed = 0

    # 抛弃人工硬编码的 domain_schedule 配比:
    # 直接使用 dataset.sample_full_pass_batch 从五大领域的全局窗口池中无放回洗牌遍历，
    # 确保每一条语料都以完全均等的 1:1 概率被模型流形完整吸收！
    converged_streak = 0
    prev_mean_eval_nll: Optional[float] = None

    for step in range(1, args.total_steps + 1):
        progress = step / max(1, args.total_steps)
        cos_factor = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))
        cur_lr_bkp = args.lr_bkp * cos_factor
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_kernel * float(pg.get("lr_scale", 1.0)) * cos_factor

        optimizer.zero_grad(set_to_none=True)
        bkp_mgr.clear_grad_traces()

        cur_dom = "general"
        cur_epoch = 0.0
        step_nll_acc_t = torch.zeros((), dtype=torch.float32, device=device)
        for micro_i in range(args.grad_accum):
            x, y, loss_mask, cur_dom, cur_epoch = dataset.sample_full_pass_batch(
                args.batch_size, device
            )
            with amp_ctx:
                nll_loss, _, _ = model(
                    x,
                    bkp_manager=bkp_mgr,
                    bkp_dropout=False,
                    target_ids=y,
                    loss_mask=loss_mask,
                )
                total_loss = nll_loss / args.grad_accum
                if micro_i == args.grad_accum - 1:
                    iso_loss = model.bkp_bus.compute_isometric_loss()
                    ortho_loss = torch.stack(
                        [cast(Any, layer).moe.compute_orthogonality_loss() for layer in model.layers]
                    ).sum()
                    total_loss = total_loss + 0.02 * iso_loss + 0.01 * ortho_loss

            total_loss.backward()
            bkp_mgr.stash_microbatch_trace()
            step_nll_acc_t.add_(nll_loss.detach().float())
            tokens_processed += int(x.numel())

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        for layer in model.layers:
            cast(Any, layer).moe.update_expert_bias()
        bkp_mgr.step_sparse_bkp_update(
            lr=cur_lr_bkp,
            purity_drop_abort_threshold=0.0,
            grad_scale=1.0 / args.grad_accum,
            collinear_core_lock_strength=None,
        )

        if step == 1 or step % 25 == 0 or step == args.total_steps:
            step_nll_acc = float(step_nll_acc_t.item()) / args.grad_accum
            elap = max(1e-3, time.time() - t_start)
            tps = tokens_processed / elap
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            active_pids = list(bkp_mgr.l1_active_pkg_ids)
            print(
                f"Step {step:5d}/{args.total_steps} (Ep {cur_epoch:4.2f}) [{cur_dom:11s}] | NLL={step_nll_acc:.4f} | "
                f"Slots={bkp_mgr.last_avg_active_slots:4.1f}/16 | "
                f"UniqExp/Seq={bkp_mgr.last_unique_experts_per_seq:5.1f} | "
                f"HopJacc={bkp_mgr.last_hop_jaccard_dist:.3f} | "
                f"Forest={len(active_pids)} pkgs {active_pids} | "
                f"{tps:,.1f} tok/s | VRAM={vram_gb:.2f}GB",
                flush=True,
            )

        # 触发自治森林拓扑演化巡检 (自动执行 Expand / Split / Merge / Spawn / Closure)
        if step % effective_evolve_interval == 0 and step <= evolve_cutoff:
            controller.step_autonomous_evolution(
                step=step, active_domain_hint=cur_dom, model=model, device=device, probe_x=probe_x
            )
            forest_snapshots.append(controller.snapshot_forest_state(step))

        if step % args.eval_interval == 0 or step == args.total_steps:
            ev_gen = eval_domain_ppl(model, bkp_mgr, dataset, "general", args.batch_size, device, n_batches=4)
            ev_cot = eval_domain_ppl(
                model, bkp_mgr, dataset, "cot_distill" if "cot_distill" in avail_domains else "math",
                args.batch_size, device, n_batches=4
            )
            eval_history.append({
                "step": step,
                "epoch": round(cur_epoch, 4),
                "general": ev_gen,
                "cot_distill": ev_cot,
                "forest": controller.snapshot_forest_state(step),
            })
            print(
                f"  [Eval @ Step {step:5d} | Ep {cur_epoch:.2f}] General PPL={ev_gen['ppl']:.2f} (UniqExp={ev_gen['unique_experts_per_seq']:.1f}) | "
                f"CoT PPL={ev_cot['ppl']:.2f} (UniqExp={ev_cot['unique_experts_per_seq']:.1f}, HopJacc={ev_cot['hop_jaccard_dist']:.3f})",
                flush=True,
            )
            # 当全量语料已至少完整过完 1 遍 (cur_epoch >= 1.0) 且双正交流形闭合已完成 (step > evolve_cutoff) 时，
            # 检查验证集平均 NLL 是否已达流形不动点 (相对变化率 < 5e-4 连续 2 次)
            cur_mean_nll = 0.5 * (float(ev_gen["nll"]) + float(ev_cot["nll"]))
            if cur_epoch >= 1.0 and step > evolve_cutoff and prev_mean_eval_nll is not None:
                rel_delta = abs(prev_mean_eval_nll - cur_mean_nll) / max(1e-6, prev_mean_eval_nll)
                if rel_delta < 5e-4:
                    converged_streak += 1
                    if converged_streak >= 2:
                        print(
                            f"  [Manifold Convergence] Step {step} (Ep {cur_epoch:.2f}): "
                            f"验证集流形能量相对变化率 ({rel_delta:.2e}) 连续达标，自动宣告收敛停止！",
                            flush=True,
                        )
                        break
                else:
                    converged_streak = 0
            prev_mean_eval_nll = cur_mean_nll

    # ==========================================================================
    # 5. 收尾封印: 对自治森林中所有 .bkp 执行 seal_unwoken_empty_pages 并落盘切为 RO_SYS
    # ==========================================================================
    print("\n" + "=" * 88)
    print(">>> [收尾封印] 对自治生长出的全部 .bkp 执行未唤醒空页封印 (seal_unwoken_empty_pages) 与 RO_SYS 固化...")
    final_forest_inventory = []
    for pid in list(bkp_mgr.l1_active_pkg_ids):
        pkg = cast(BKPPackage, bkp_mgr.packages[bkp_mgr._mod_key(pid)])
        sealed_cnt = pkg.seal_unwoken_empty_pages(min_activations=2)
        saved_p = controller._persist_pkg_to_forest(pkg, pkg.pkg_id)
        pkg.mount_mode = BKPMountMode.RO_SYS
        woken_final = int((pkg.scales[:, 1] > 0).sum().item())
        info = {
            "pkg_id": pkg.pkg_id,
            "grid": f"{pkg.grid_k}x{pkg.grid_k}",
            "num_experts": pkg.num_experts,
            "woken_mature_experts": woken_final,
            "sealed_empty_pages": sealed_cnt,
            "size_mb": round(pkg.num_experts * 4096 / (1024 * 1024), 1),
            "quality_q": round(float(pkg.quality_q.item()), 4),
            "path": str(saved_p),
        }
        final_forest_inventory.append(info)
        print(
            f"  - {pkg.pkg_id:24s} | 网格={pkg.grid_k}x{pkg.grid_k} ({info['size_mb']} MB) | "
            f"成熟微专家={woken_final:,}/{pkg.num_experts:,} | 封印空页={sealed_cnt:,} | Q_p={info['quality_q']:.4f}",
            flush=True,
        )

    # 持久化保存完整 Kernel 权重检查点 (供第二阶段真伪判别与终身增量学习直接加载继承)
    ckpt_dir = ws / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "kernel_final.pt"
    torch.save(
        {
            "run_name": args.run_name,
            "total_steps": args.total_steps,
            "config": model.config.__dict__,
            "model_state_dict": model.state_dict(),
            "forest_dir": str(forest_dir),
            "active_packages": list(bkp_mgr.l1_active_pkg_ids),
        },
        ckpt_path,
    )
    print(f"  [基座固化] 已保存 BicameraKernel 检查点至: {ckpt_path}", flush=True)

    # ==========================================================================
    # 6. 五大领域完整自治 .bkp 森林 PPL 评估 (已废除冗余的拔除裸核消融重复评测)
    # ==========================================================================
    print("\n>>> [实验 1] 五大领域完整自治 .bkp 森林 PPL 评测...")
    domain_ablation = {}
    for dom in avail_domains:
        res_mounted = eval_domain_ppl(model, bkp_mgr, dataset, dom, args.batch_size, device, n_batches=8)
        domain_ablation[dom] = {
            "mounted_forest": res_mounted,
        }
        print(
            f"  [{dom:11s}] 挂载森林 PPL={res_mounted['ppl']:7.2f} "
            f"(NLL={res_mounted['nll']:.4f}, UniqExp/Seq={res_mounted['unique_experts_per_seq']:.1f}, "
            f"HopJacc={res_mounted['hop_jaccard_dist']:.3f})",
            flush=True,
        )

    # ==========================================================================
    # 7. 经典大模型验证集与跨领域自我纠错基准跑分 (仅评测完整自治森林)
    # ==========================================================================
    print("\n>>> [实验 2] 运行经典大模型验证集与狄拉克自我纠错评测 (ARC / PIQA / MMLU / GSM8K / Humanoid)...")
    bench_full_forest = run_full_benchmark_suite(model, bkp_mgr, tok, device, state_label="1_Full_Autonomous_Forest")

    gsm_f = bench_full_forest["GSM8K_MultiHop_CoT"]
    print(
        f"  [经典基准汇总 - 挂载自治森林] Overall_MC_Acc={bench_full_forest['overall_mc_acc_norm_pct']:.2f}% "
        f"(ARC={bench_full_forest['ARC_Science']['acc_norm']:.1f}%, PIQA={bench_full_forest['PIQA_Commonsense']['acc_norm']:.1f}%, "
        f"MMLU={bench_full_forest['MMLU_CrossDomain']['acc_norm']:.1f}%) | "
        f"GSM8K_CoT_Hit={gsm_f['greedy_answer_hit_rate_pct']:.1f}% | "
        f"CoT_Gain={gsm_f['cot_reasoning_gain_nll']:+.4f} NLL | "
        f"时间换空间放大={gsm_f['time_to_space_amplification_x']:.2f}x "
        f"(单步 {gsm_f['avg_slots_per_single_step']:.1f} 槽位 -> 单次思考累计唤醒 {gsm_f['avg_unique_experts_per_cot_seq']:.1f} 个独立微专家!)",
        flush=True,
    )

    lineage_file = log_dir / "forest_lineage.json"
    report_file = log_dir / "autonomous_forest_report.json"
    summary_file = log_dir / "summary.json"
    status_file = log_dir / "status.json"

    full_report = {
        "run_name": args.run_name,
        "kernel_params_m": round(n_params_m, 2),
        "total_steps": args.total_steps,
        "wall_time_sec": round(time.time() - t_start, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0,
        "autonomous_lineage_events": controller.lineage_events,
        "final_forest_inventory": final_forest_inventory,
        "domain_ablation": domain_ablation,
        "benchmarks": {
            "full_forest_mounted": bench_full_forest,
        },
        "eval_history": eval_history,
    }
    with open(lineage_file, "w", encoding="utf-8") as f:
        json.dump(controller.lineage_events, f, indent=2, ensure_ascii=False)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"state": "succeeded", "run_name": args.run_name, "report": str(report_file)}, f, indent=2)

    print(f"\n>>> [全部完成] 自治森林族谱已保存至: {lineage_file}")
    print(f">>> [全部完成] 评测与消融总报告已保存至: {report_file}", flush=True)
    try:
        from serve_penrose_api import run_intelligence_scorecard
        run_intelligence_scorecard(args.run_name)
    except Exception as e_sc:
        print(f"[WARN] 生成标准智能记分卡失败: {e_sc}", flush=True)


if __name__ == "__main__":
    main()








