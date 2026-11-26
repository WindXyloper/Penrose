"""
Penrose v1.0 — PMC (Penrose Manifold Chunk, `.pmc` / `.bkp`)
============================================================
基于十条零超参数闭式物理公理的同质微专家黎曼流形图册（Atlas）管理引擎：
- 全宇宙共享唯一的 2048 维母体单位超球面 S^2047；
- 每一个 4KB 微专家 (u_m, v_m) 是钉在 S^2047 上的嘉当活动标架（Cartan Moving Frame）；
- 每一个 `.pmc` 是贴合在母体超球面特定区域上的自适应分辨率局部坐标卡（Local Coordinate Chart）；
- `BridgeAB.pmc` 是连接不同流形板块的测地线联络桥（Transition Bundle），承载霍奇分解的第三极：拓扑调和虫洞场 H^1(M)。
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from bkp_manager import (
    D_BKP,
    PAGE_SIZE,
    BKP_NUM_HEADS,
    BKP_HEAD_DIM,
    BKP_SUBKEY_DIM,
    STANDARD_GRID_SIZES,
    EULER_MASCHERONI,
    GUMBEL_STD_COEFF,
    SIMPLEX_2_GEODESIC_PROJ,
    SIMPLEX_3_BARYCENTER_PROJ,
    JET_NORMAL_ALPHA_STAR,
    JET_TANGENT_NORM_STAR,
    HEAD_INV_GAIN,
    HEAD_UNIFORM_RATIO,
    SUBSPACE_STD_SCALE,
    AMBIENT_NOISE_FLOOR,
    GUMBEL_SINK_DIV_STAR,
    FUNNEL_RESONANCE_APERTURE,
    closed_form_gumbel_threshold,
    round_up_to_standard_grid,
    BKPMountMode as PMCMountMode,
    BKPRoutingConfig as PMCRoutingConfig,
    BKPPackage,
    BKPManager,
)

PMC_MAGIC: bytes = b"PMC2048\x00"


@dataclass
class HelmholtzFieldDiagnostics:
    """
    PMC 流形图册在当前球面状态 q(tau) 处的实时霍奇三分解微分几何诊断量（驱动 Module 5 三态总线分支）：
      V|_M = V_div (负散度数据汇) (+) V_curl (局部无散旋度流) (+) H^1(M) (拓扑调和虫洞/联络桥流)
    - delta_pmc       : (B, T, 2048) 流形图册合成切向推力 V_PMC(q)
    - div_v           : (B, T) 局部散度 Tr(J_local) = sum_m w_m <u_m, v_m>
    - eta_curl        : (B, T) 局部切向旋度能量比 in [0, 1]
    - active_experts  : (B, T, K_active) 当前步唤醒的微专家全局页索引
    - active_weights  : (B, T, K_active) 对应的球面 L2 单位分解权重 w_m(q)
    - max_resonance   : (B, T) 最大微专家共振内积峰值
    - eta_harmonic    : Optional[(B, T)] 霍奇分解第三项 H^1(M)（来自 Bridge.pmc 联络桥与 1-柄虫洞）的能量占比
    """
    delta_pmc: torch.Tensor
    div_v: torch.Tensor
    eta_curl: torch.Tensor
    active_experts: torch.Tensor
    active_weights: torch.Tensor
    max_resonance: torch.Tensor
    eta_harmonic: Optional[torch.Tensor] = None


class PMCPackage(BKPPackage):
    """
    Penrose Manifold Chunk (`.pmc`) 流形局部坐标卡与联络桥实体。
    100% 继承 BKPPackage 的 4KB 同质微专家活动标架 `(u_m, v_m)` 与十条闭式公理算子，
    并扩展静态流线指针表 `stream_next_ptr` 与无 AdamW 罗德里格斯大圆旋转写入。
    """

    def __init__(
        self,
        pkg_id: str,
        grid_k: int = 64,
        mount_mode: PMCMountMode = PMCMountMode.RW_SHARED,
        is_base_pkg: bool = False,
        mmap_file_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            pkg_id=pkg_id,
            grid_k=grid_k,
            mount_mode=mount_mode,
            is_base_pkg=is_base_pkg,
            mmap_file_path=mmap_file_path,
            device=device,
        )
        self.register_buffer(
            "stream_next_ptr",
            torch.full((self.num_experts, 4), -1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self._needs_quadrant_expand: bool = False

    def _rebuild_runtime_buffers_after_topology_change(self) -> None:
        super()._rebuild_runtime_buffers_after_topology_change()
        dev = self.scales.device
        if not hasattr(self, "stream_next_ptr") or self.stream_next_ptr.shape[0] != self.num_experts:
            self.register_buffer(
                "stream_next_ptr",
                torch.full((self.num_experts, 4), -1, dtype=torch.int32, device=dev),
                persistent=False,
            )

    @staticmethod
    def _remap_stream_table(
        old_ptr: torch.Tensor,
        old_to_new_eid: torch.Tensor,
        new_num_experts: int,
        src_eids: Optional[torch.Tensor] = None,
        dst_eids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dev = old_ptr.device
        new_ptr = torch.full((new_num_experts, 4), -1, dtype=torch.int32, device=dev)
        if src_eids is None:
            src_eids = torch.arange(old_ptr.size(0), device=dev, dtype=torch.long)
        if dst_eids is None:
            dst_eids = old_to_new_eid[src_eids]
        valid_rows = (dst_eids >= 0) & (dst_eids < new_num_experts)
        src_v = src_eids[valid_rows]
        dst_v = dst_eids[valid_rows]
        sub_ptr = old_ptr[src_v].long()
        valid_t = (sub_ptr >= 0) & (sub_ptr < old_to_new_eid.numel())
        mapped = torch.full_like(sub_ptr, -1)
        if bool(valid_t.any()):
            mapped[valid_t] = old_to_new_eid[sub_ptr[valid_t]]
        new_ptr[dst_v] = mapped.to(torch.int32)
        return new_ptr

    @torch.no_grad()
    def expand_quadrant_2x(self, perturb_scale: float = 0.05) -> None:
        old_k = self.grid_k
        new_k = old_k * 2
        dev = self.scales.device
        old_ptr = self.stream_next_ptr.clone()
        old_eids = torch.arange(old_k * old_k, device=dev, dtype=torch.long)
        r_old = torch.div(old_eids, old_k, rounding_mode="floor")
        c_old = old_eids % old_k
        old_to_new = r_old * new_k + c_old

        super().expand_quadrant_2x(perturb_scale=perturb_scale)
        self.stream_next_ptr.copy_(
            self._remap_stream_table(old_ptr, old_to_new, new_k * new_k, src_eids=old_eids, dst_eids=old_to_new)
        )
        self._needs_quadrant_expand = False

    @torch.no_grad()
    def permute_rows_cols(self, row_perm: torch.Tensor, col_perm: torch.Tensor) -> None:
        k = self.grid_k
        dev = self.scales.device
        old_ptr = self.stream_next_ptr.clone()
        r_dev = row_perm.to(dev).long()
        c_dev = col_perm.to(dev).long()
        inv_r = torch.empty_like(r_dev)
        inv_c = torch.empty_like(c_dev)
        inv_r[r_dev] = torch.arange(k, device=dev, dtype=torch.long)
        inv_c[c_dev] = torch.arange(k, device=dev, dtype=torch.long)

        old_eids = torch.arange(k * k, device=dev, dtype=torch.long)
        r_old = torch.div(old_eids, k, rounding_mode="floor")
        c_old = old_eids % k
        old_to_new = inv_r[r_old] * k + inv_c[c_old]

        super().permute_rows_cols(row_perm, col_perm)
        self.stream_next_ptr.copy_(
            self._remap_stream_table(old_ptr, old_to_new, k * k, src_eids=old_eids, dst_eids=old_to_new)
        )

    @torch.no_grad()
    def extract_padded_subpackage(self, sub_id: str, row_idx: torch.Tensor, col_idx: torch.Tensor) -> "PMCPackage":
        base_sub = super().extract_padded_subpackage(sub_id, row_idx, col_idx)
        dev = self.u_codebook.device
        pmc_sub = PMCPackage(
            pkg_id=base_sub.pkg_id,
            grid_k=base_sub.grid_k,
            mount_mode=base_sub.mount_mode,
            is_base_pkg=base_sub.is_base_pkg,
            device=dev,
        )
        pmc_sub._copy_manifold_stats_from(base_sub)
        pmc_sub.codebook_pre_centered.copy_(base_sub.codebook_pre_centered)
        pmc_sub._codebook_pre_centered_cached = base_sub._codebook_pre_centered_cached
        pmc_sub.u_codebook.data.copy_(base_sub.u_codebook.data)
        pmc_sub.v_codebook.data.copy_(base_sub.v_codebook.data)
        pmc_sub.expert_pages_int8.copy_(base_sub.expert_pages_int8)
        pmc_sub.scales.copy_(base_sub.scales)
        pmc_sub.activation_counts.copy_(base_sub.activation_counts)
        pmc_sub.purity_r.copy_(base_sub.purity_r)
        pmc_sub._rebuild_runtime_buffers_after_topology_change()

        k_old = self.grid_k
        k_new = pmc_sub.grid_k
        r_dev = row_idx.to(dev).long()
        c_dev = col_idx.to(dev).long()
        inv_r = torch.full((k_old,), -1, dtype=torch.long, device=dev)
        inv_c = torch.full((k_old,), -1, dtype=torch.long, device=dev)
        inv_r[r_dev] = torch.arange(r_dev.numel(), device=dev, dtype=torch.long)
        inv_c[c_dev] = torch.arange(c_dev.numel(), device=dev, dtype=torch.long)

        old_eids = torch.arange(k_old * k_old, device=dev, dtype=torch.long)
        ro = torch.div(old_eids, k_old, rounding_mode="floor")
        co = old_eids % k_old
        rn = inv_r[ro]
        cn = inv_c[co]
        old_to_new = torch.where((rn >= 0) & (cn >= 0), rn * k_new + cn, torch.full_like(old_eids, -1))
        pmc_sub.stream_next_ptr.copy_(
            self._remap_stream_table(self.stream_next_ptr, old_to_new, k_new * k_new)
        )
        return pmc_sub

    @staticmethod
    @torch.no_grad()
    def bkp_merge_direct_sum(pkg_a: BKPPackage, pkg_b: BKPPackage, merged_id: str) -> "PMCPackage":
        base_merged = BKPPackage.bkp_merge_direct_sum(pkg_a, pkg_b, merged_id=merged_id)
        dev = base_merged.u_codebook.device
        pmc_merged = PMCPackage(
            pkg_id=base_merged.pkg_id,
            grid_k=base_merged.grid_k,
            mount_mode=base_merged.mount_mode,
            is_base_pkg=base_merged.is_base_pkg,
            device=dev,
        )
        pmc_merged._copy_manifold_stats_from(base_merged)
        pmc_merged.codebook_pre_centered.copy_(base_merged.codebook_pre_centered)
        pmc_merged._codebook_pre_centered_cached = base_merged._codebook_pre_centered_cached
        pmc_merged.u_codebook.data.copy_(base_merged.u_codebook.data)
        pmc_merged.v_codebook.data.copy_(base_merged.v_codebook.data)
        pmc_merged.expert_pages_int8.copy_(base_merged.expert_pages_int8)
        pmc_merged.scales.copy_(base_merged.scales)
        pmc_merged.activation_counts.copy_(base_merged.activation_counts)
        pmc_merged.purity_r.copy_(base_merged.purity_r)
        pmc_merged._rebuild_runtime_buffers_after_topology_change()

        km = pmc_merged.grid_k
        ka, kb = pkg_a.grid_k, pkg_b.grid_k
        if isinstance(pkg_a, PMCPackage):
            ea = torch.arange(ka * ka, device=dev, dtype=torch.long)
            ra = torch.div(ea, ka, rounding_mode="floor")
            ca = ea % ka
            map_a = ra * km + ca
            ptr_a = PMCPackage._remap_stream_table(pkg_a.stream_next_ptr, map_a, km * km, src_eids=ea, dst_eids=map_a)
            pmc_merged.stream_next_ptr[map_a] = ptr_a[map_a]
        if isinstance(pkg_b, PMCPackage):
            eb = torch.arange(kb * kb, device=dev, dtype=torch.long)
            rb = torch.div(eb, kb, rounding_mode="floor") + ka
            cb = (eb % kb) + ka
            map_b = rb * km + cb
            ptr_b = PMCPackage._remap_stream_table(pkg_b.stream_next_ptr, map_b, km * km, src_eids=eb, dst_eids=map_b)
            pmc_merged.stream_next_ptr[map_b] = ptr_b[map_b]
        return pmc_merged

    @staticmethod
    @torch.no_grad()
    def bkp_merge_triadic_bundle(pkg_a: BKPPackage, pkg_b: BKPPackage, merged_id: str) -> "PMCPackage":
        merged = PMCPackage.bkp_merge_direct_sum(pkg_a, pkg_b, merged_id=merged_id)
        ka, kb = pkg_a.grid_k, pkg_b.grid_k
        k_min = min(ka, kb)
        dev = merged.scales.device
        idx_common = torch.arange(k_min, device=dev)
        eids_a = idx_common * merged.grid_k + idx_common
        eids_b = (ka + idx_common) * merged.grid_k + (ka + idx_common)

        pages_cpu = merged.expert_pages_int8.float()
        scales_cpu = merged.scales.float().cpu()
        eids_a_cpu, eids_b_cpu = eids_a.cpu(), eids_b.cpu()

        ua_cur = (pages_cpu[eids_a_cpu, 0, :] * scales_cpu[eids_a_cpu, 0:1]).to(dev)
        va_cur = (pages_cpu[eids_a_cpu, 1, :] * scales_cpu[eids_a_cpu, 1:2]).to(dev)
        ub_cur = (pages_cpu[eids_b_cpu, 0, :] * scales_cpu[eids_b_cpu, 0:1]).to(dev)
        vb_cur = (pages_cpu[eids_b_cpu, 1, :] * scales_cpu[eids_b_cpu, 1:2]).to(dev)

        v_even = 0.5 * (va_cur + vb_cur)
        v_odd = 0.5 * (va_cur - vb_cur)
        va_triadic = v_even + v_odd
        vb_triadic = v_even - v_odd

        merged.write_experts_to_mmap_pages(eids_a, ua_cur, va_triadic)
        merged.write_experts_to_mmap_pages(eids_b, ub_cur, vb_triadic)
        return merged

    @torch.no_grad()
    def record_streamline_transition(
        self,
        curr_eids: torch.Tensor,
        next_eids: torch.Tensor,
    ) -> None:
        if curr_eids.numel() == 0 or next_eids.numel() == 0:
            return
        c_flat = curr_eids.reshape(-1).to(torch.long)
        n_flat = next_eids.reshape(-1).to(torch.int32)
        valid = (c_flat >= 0) & (c_flat < self.num_experts) & (n_flat >= 0) & (n_flat < self.num_experts)
        if not bool(valid.any()):
            return
        c_valid = c_flat[valid]
        n_valid = n_flat[valid]
        cur_slots = self.stream_next_ptr[c_valid]
        already_in = (cur_slots == n_valid.unsqueeze(-1)).any(dim=-1)
        update_mask = ~already_in
        if bool(update_mask.any()):
            idx_u = c_valid[update_mask]
            val_u = n_valid[update_mask]
            shifted = torch.empty_like(self.stream_next_ptr[idx_u])
            shifted[:, 0] = val_u
            shifted[:, 1:] = self.stream_next_ptr[idx_u, :-1]
            self.stream_next_ptr[idx_u] = shifted

    @torch.no_grad()
    def sever_streamline_transition(
        self,
        curr_eids: torch.Tensor,
        next_eids: torch.Tensor,
    ) -> int:
        """
        对偶流线表拓扑断链算子（Streamline Topological Severing）：
        将 `stream_next_ptr[curr_eids]` 中指向 `next_eids` 的有向因果边重置为 `-1`，
        阻断沿被擦除轨迹的 L2 -> L1 测地流线预取。
        """
        if curr_eids.numel() == 0 or next_eids.numel() == 0:
            return 0
        c_flat = curr_eids.reshape(-1).to(torch.long)
        n_flat = next_eids.reshape(-1).to(torch.int32)
        valid = (c_flat >= 0) & (c_flat < self.num_experts) & (n_flat >= 0) & (n_flat < self.num_experts)
        if not bool(valid.any()):
            return 0
        c_valid = c_flat[valid]
        n_valid = n_flat[valid]
        cur_slots = self.stream_next_ptr[c_valid]
        match_mask = cur_slots == n_valid.unsqueeze(-1)
        severed_cnt = int(match_mask.sum().item())
        if severed_cnt > 0:
            self.stream_next_ptr[c_valid] = torch.where(
                match_mask,
                torch.full_like(cur_slots, -1),
                cur_slots,
            )
        return severed_cnt

    @torch.no_grad()
    def apoptosis_expert_pages(self, eids: torch.Tensor) -> int:
        """
        公理 9 细胞凋亡算子（Cellular Apoptosis）：
        当独占微专家的激活计数回退归零时，原位清零其 4KB Int8 页、缩放因子、激活计数、
        纯度与流线邻接表，释放网格槽位供后续有丝分裂或新概念复用。
        """
        if eids.numel() == 0:
            return 0
        dev_ids = eids.reshape(-1).to(self.scales.device).long()
        cpu_ids = dev_ids.detach().cpu()
        self.expert_pages_int8[cpu_ids] = 0
        if getattr(self, "_expert_pages_dev", None) is not None:
            self._expert_pages_dev[dev_ids.to(self._expert_pages_dev.device)] = 0
        self.scales[dev_ids] = 0.0
        self.activation_counts[dev_ids] = 0
        self.purity_r[dev_ids] = 1.0
        self.stream_next_ptr[dev_ids] = -1
        self._sparse_128_triblock_mode = None
        return int(dev_ids.numel())

    @torch.no_grad()
    def quantize_write_expert_page(
        self,
        eid: int,
        u_vec: torch.Tensor,
        v_vec: torch.Tensor,
        hit_increment: float = 1.0,
        purity_val: float = 1.0,
    ) -> None:
        dev = self.scales.device
        eid_int = int(eid)
        eid_t = torch.tensor([eid_int], dtype=torch.long, device=dev)
        u_unit = F.normalize(u_vec.detach().to(device=dev, dtype=torch.float32).view(1, D_BKP), dim=-1)
        u_f = u_unit * math.sqrt(float(D_BKP))
        v_f = v_vec.detach().to(device=dev, dtype=torch.float32).view(1, D_BKP)
        self.write_experts_to_mmap_pages(eid_t, u_f, v_f)
        self.activation_counts[eid_int] = self.activation_counts[eid_int] + int(max(1.0, round(hit_increment)))
        self.purity_r[eid_int] = float(max(0.0, min(1.0, purity_val)))

        # 同步将第一级笛卡尔路由码本 (u_codebook[r], v_codebook[c]) 对齐到 u_unit 的子空间方向，确保测地圆路由与页内键一致
        r_idx = eid_int // self.grid_k
        c_idx = eid_int % self.grid_k
        u_heads = F.normalize(u_unit.view(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM), dim=-1)
        self.u_codebook.data[:, r_idx, :] = PMCPackage.rodrigues_geodesic_rotation(
            self.u_codebook.data[:, r_idx, :], u_heads[:, 0, :], step_fraction=1.0
        )
        self.v_codebook.data[:, c_idx, :] = PMCPackage.rodrigues_geodesic_rotation(
            self.v_codebook.data[:, c_idx, :], u_heads[:, 1, :], step_fraction=1.0
        )
        self._u_cb_norm_cache = None
        self._v_cb_norm_cache = None

    @torch.no_grad()
    def dequantize_expert_uv(self, eids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if eids.dim() == 0:
            u_out, v_out = self.read_experts_from_mmap_pages(eids.view(1))
            return u_out.squeeze(0), v_out.squeeze(0)
        return self.read_experts_from_mmap_pages(eids)

    @staticmethod
    @torch.no_grad()
    def rodrigues_geodesic_rotation(
        u_curr: torch.Tensor,
        q_target: torch.Tensor,
        step_fraction: float = 1.0,
    ) -> torch.Tensor:
        u_unit = F.normalize(u_curr.float(), dim=-1)
        q_unit = F.normalize(q_target.float(), dim=-1)
        cos_theta = (u_unit * q_unit).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta) * float(step_fraction)
        w_perp = F.normalize(q_unit - cos_theta * u_unit, dim=-1)
        u_rot = torch.cos(theta) * u_unit + torch.sin(theta) * w_perp
        return F.normalize(u_rot, dim=-1)


class PMCManager(BKPManager):
    """
    Penrose Manifold Chunk (`.pmc`) 流形图册（Atlas）与联络桥管理器。
    """

    def __init__(
        self,
        routing_cfg: Optional[PMCRoutingConfig] = None,
        cache_slots: int = 131072,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(config=routing_cfg, cache_slots=cache_slots, device=device)
        self.prefetch_queue_eids: List[Tuple[str, torch.Tensor]] = []
        self.total_kaczmarz_snaps: int = 0
        self.total_mitosis_spawns: int = 0
        self.split_count: int = 0
        self.expand_count: int = 0
        self.merge_count: int = 0
        self.spawn_count: int = 0
        self.closure_count: int = 0
        self.forest_events: List[Dict[str, Any]] = []

    @property
    def mounted_packages(self) -> List[BKPPackage]:
        return list(self.packages.values())

    def get_writable_package(self) -> Optional[PMCPackage]:
        for pid in reversed(self.l1_active_pkg_ids):
            mk = self._mod_key(pid)
            if mk not in self.packages:
                continue
            pkg = self.packages[mk]
            if (
                isinstance(pkg, PMCPackage)
                and (not pkg.is_bridge_pkg)
                and pkg.mount_mode in (PMCMountMode.RW_SHARED, PMCMountMode.COW_PRIVATE)
            ):
                return pkg
        for pkg in reversed(self.mounted_packages):
            if not pkg.is_bridge_pkg and pkg.mount_mode in (PMCMountMode.RW_SHARED, PMCMountMode.COW_PRIVATE):
                if isinstance(pkg, PMCPackage):
                    return pkg
        for pkg in self.mounted_packages:
            if isinstance(pkg, PMCPackage):
                return pkg
        return None

    @torch.no_grad()
    def select_resonant_writable_package(self, psi: torch.Tensor) -> Optional[PMCPackage]:
        """
        母体超球面 S^2047 局部坐标卡归属路由（Atlas Chart Resonance Routing）：
        严格仅在 L1 快表 (self.l1_active_pkg_ids) 的可写局部坐标卡内，复用 BKPManager._standardize_subqueries
        与 _precompute_pkg_subkey_scores 的 O(K_p) 状元分选择最匹配的 `.pmc` 坐标卡写入，绝不全局扫描冷包！
        """
        l1_writable: List[PMCPackage] = []
        for pid in self.l1_active_pkg_ids:
            mk = self._mod_key(pid)
            if mk not in self.packages:
                continue
            pkg = self.packages[mk]
            if (
                isinstance(pkg, PMCPackage)
                and (not pkg.is_bridge_pkg)
                and pkg.mount_mode in (PMCMountMode.RW_SHARED, PMCMountMode.COW_PRIVATE)
            ):
                l1_writable.append(pkg)
        if len(l1_writable) == 0:
            return self.get_writable_package()
        if len(l1_writable) == 1:
            return l1_writable[0]

        psi_probe = psi.detach().float().reshape(-1, D_BKP)[:16]
        q1, q2 = self._standardize_subqueries(psi_probe, update_center=False)
        best_pkg = l1_writable[-1]
        best_score = -1e9
        for chart in l1_writable:
            s1_c, s2_c, _, _ = self._precompute_pkg_subkey_scores(chart, q1, q2)
            score = float((0.5 * (s1_c.amax(dim=-1).mean() + s2_c.amax(dim=-1).mean())).item())
            if score >= best_score:
                best_score = score
                best_pkg = chart
        return best_pkg

    @torch.no_grad()
    def prefetch_along_geodesic_streamline(
        self,
        psi_0: torch.Tensor,
        top_m: int = 8,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Module 1 一阶射流早期预取：仅对 L1 快表 (self.l1_active_pkg_ids) 内的活跃包执行 O(K_p) 子键与 BTB 流线指针预取。
        """
        self.prefetch_queue_eids.clear()
        if len(self.l1_active_pkg_ids) == 0:
            return self.prefetch_queue_eids

        q_flat = psi_0.detach().float().reshape(-1, D_BKP)
        q_probe = q_flat[: min(16, q_flat.size(0))]
        q1, q2 = self._standardize_subqueries(q_probe, update_center=False)

        for pid in self.l1_active_pkg_ids:
            mk = self._mod_key(pid)
            if mk not in self.packages:
                continue
            pkg = self.packages[mk]
            s1_p, s2_p, _, _ = self._precompute_pkg_subkey_scores(pkg, q1, q2)
            top_u = s1_p.amax(dim=1).topk(min(top_m, pkg.grid_k), dim=-1).indices
            top_v = s2_p.amax(dim=1).topk(min(top_m, pkg.grid_k), dim=-1).indices
            eids_curr = (top_u * pkg.grid_k + top_v).reshape(-1).unique()

            if isinstance(pkg, PMCPackage):
                next_ptrs = pkg.stream_next_ptr[eids_curr].reshape(-1)
                valid_next = next_ptrs[next_ptrs >= 0].to(eids_curr.dtype)
                if valid_next.numel() > 0:
                    eids_curr = torch.cat([eids_curr, valid_next]).unique()

            self.prefetch_queue_eids.append((pkg.pkg_id, eids_curr))
        return self.prefetch_queue_eids

    def query_with_helmholtz_diagnostics(
        self,
        q_in: torch.Tensor,
        puncture_dropout_prob: float = 0.0,
        puncture_expert_mask: Optional[torch.Tensor] = None,
    ) -> HelmholtzFieldDiagnostics:
        """
        100% 复用 BKPManager.query_and_inject 的三层测地圆漏斗路由与 L2 能量守恒单位分解：
        - 仅扫描 L1 快表 (l1_active_pkg_ids)，包级冈贝尔极值底线 tau_z^*(N_grid) 秒筛短路；
        - 测地圆半径 [1/sqrt(3)*tau_z^*, 1/sqrt(2)*tau_z^*] 三次厄米特单纯形切子空间投影激活；
        - 零点锚定 Softmax_1 + 严格 L2 能量守恒增益 ortho_gain 插值；
        - 零全局扫描、零重复调页，直接读取 query_and_inject 原生产出的 _last_helmholtz_cache。
        """
        delta_pmc = self.query_and_inject(
            q_in,
            puncture_dropout_prob=puncture_dropout_prob,
            puncture_expert_mask=puncture_expert_mask,
        )

        hc = getattr(self, "_last_helmholtz_cache", None)
        if hc is None:
            B = q_in.shape[0]
            T = q_in.shape[1] if q_in.dim() == 3 else 1
            dev = q_in.device
            zero_scalar = torch.zeros((B, T), device=dev, dtype=torch.float32)
            return HelmholtzFieldDiagnostics(
                delta_pmc=delta_pmc,
                div_v=zero_scalar,
                eta_curl=torch.ones_like(zero_scalar),
                active_experts=torch.zeros((B, T, 1), device=dev, dtype=torch.long),
                active_weights=zero_scalar.unsqueeze(-1),
                max_resonance=zero_scalar,
                eta_harmonic=zero_scalar,
            )
        return HelmholtzFieldDiagnostics(
            delta_pmc=delta_pmc,
            div_v=hc["div_v"],
            eta_curl=hc["eta_curl"],
            active_experts=hc["active_experts"],
            active_weights=hc["active_weights"],
            max_resonance=hc["max_resonance"],
            eta_harmonic=hc["eta_harmonic"],
        )

    @torch.no_grad()
    def geodesic_kaczmarz_snap_update(
        self,
        q_curr: torch.Tensor,
        q_prev: torch.Tensor,
        q_target: torch.Tensor,
        snap_rate: float = 1.0,
        enable_auto_mitosis: bool = True,
    ) -> Dict[str, Any]:
        """
        Penrose v1.0 核心创新：彻底废除 AdamW 的「Kaczmarz 单纯形正交投影 + 罗德里格斯大圆旋转」一步闭式更新器。
        支持母体超球面 S^2047 多坐标卡图册自适应归属路由（Atlas Chart Resonance Routing）。
        """
        q_c = F.normalize(q_curr.detach().float().reshape(-1, D_BKP), dim=-1)
        q_p = F.normalize(q_prev.detach().float().reshape(-1, D_BKP), dim=-1)
        q_t = F.normalize(q_target.detach().float().reshape(-1, D_BKP), dim=-1)
        N = q_c.size(0)

        dq = q_c - (q_c * q_p).sum(dim=-1, keepdim=True) * q_c
        dq_norm = dq.norm(dim=-1, keepdim=True)
        has_tangent = (dq_norm > 1e-5).float()
        dq_unit = F.normalize(dq, dim=-1)
        psi = F.normalize(
            JET_NORMAL_ALPHA_STAR * q_c
            + has_tangent * JET_TANGENT_NORM_STAR * dq_unit
            + (1.0 - has_tangent) * (1.0 - JET_NORMAL_ALPHA_STAR) * q_c,
            dim=-1,
        )

        pkg = self.select_resonant_writable_package(psi)
        if pkg is None:
            return {"updated_pages": 0, "mitosis_pages": 0, "optimizer_memory_bytes": 0}

        v_tan = q_t - (q_t * q_c).sum(dim=-1, keepdim=True) * q_c
        # 公理 3 闭式正交合成：法向向心凝聚分量 (-1/sqrt(3) * q_c) 形成标准负散度势阱 (div V < 0)，
        # 切向推进分量 (sqrt(3) * v_tan) 在 Module 5 切空间正交化滤除法向分量后直接推动状态滑向 q_target！
        v_desired = (1.0 / SIMPLEX_3_BARYCENTER_PROJ) * v_tan - SIMPLEX_3_BARYCENTER_PROJ * q_c

        # 100% 复用 BKPManager._standardize_subqueries 与 normalized_codebooks，确保 Kaczmarz 写入与测地圆读取共享同一子键流形坐标系！
        q1_psi, q2_psi = self._standardize_subqueries(psi, update_center=False)
        u_cb, v_cb = pkg.normalized_codebooks()

        su = torch.einsum("nhd,hkd->nhk", q1_psi.float(), u_cb.float()).mean(dim=1)
        sv = torch.einsum("nhd,hkd->nhk", q2_psi.float(), v_cb.float()).mean(dim=1)
        r_star = su.argmax(dim=-1)
        c_star = sv.argmax(dim=-1)
        eids = r_star * pkg.grid_k + c_star
        dev = psi.device

        # 构造与 query_and_inject 完全对齐的标准化球面查询向量 q_sphere_psi
        q_sphere_psi = F.normalize(torch.stack([q1_psi, q2_psi], dim=2).reshape(N, D_BKP).float(), dim=-1)

        # 1. 向量化记录相邻时间步的测地流线转移边 (零循环)
        if N > 1:
            diff_step = eids[:-1] != eids[1:]
            if bool(diff_step.any()):
                pkg.record_streamline_transition(eids[:-1][diff_step], eids[1:][diff_step])

        # 2. 按唯一命中的微专家页号执行张量级球面重心聚合 (unique_eids, inverse_indices, hit_counts)
        unique_eids, inverse_indices, hit_counts = torch.unique(
            eids, sorted=True, return_inverse=True, return_counts=True
        )
        u_cnt = int(unique_eids.numel())
        if u_cnt == 0:
            return {"updated_pages": 0, "mitosis_pages": 0, "optimizer_memory_bytes": 0}

        psi_sum = torch.zeros((u_cnt, D_BKP), dtype=torch.float32, device=dev)
        v_des_sum = torch.zeros((u_cnt, D_BKP), dtype=torch.float32, device=dev)
        psi_sum.index_add_(0, inverse_indices, q_sphere_psi)
        v_des_sum.index_add_(0, inverse_indices, v_desired)
        inv_cnt = (1.0 / hit_counts.float()).unsqueeze(-1)
        psi_bar = F.normalize(psi_sum * inv_cnt, dim=-1)
        v_des_bar = v_des_sum * inv_cnt

        # 3. 批量读取唯一命中页的存量 (u_old, v_old) 与状态
        u_old, v_old = pkg.dequantize_expert_uv(unique_eids)
        is_empty_u = pkg.scales[unique_eids, 1] <= 1e-6
        cos_v_u = F.cosine_similarity(v_old, v_des_bar, dim=-1)

        tok_to_bar_cos = F.cosine_similarity(v_desired, v_des_bar[inverse_indices], dim=-1)
        intra_fork_mask = (hit_counts[inverse_indices] > 1) & (tok_to_bar_cos < SIMPLEX_2_GEODESIC_PROJ)

        first_occ = torch.zeros(u_cnt, dtype=torch.long, device=dev)
        if bool(intra_fork_mask.any()):
            rev_idx = torch.arange(N - 1, -1, -1, device=dev, dtype=torch.long)
            first_occ.index_copy_(0, inverse_indices[rev_idx], rev_idx)
            has_intra_u = torch.zeros(u_cnt, dtype=torch.bool, device=dev)
            has_intra_u.index_fill_(0, inverse_indices[intra_fork_mask], True)
            psi_bar = torch.where(has_intra_u.unsqueeze(-1), q_sphere_psi[first_occ], psi_bar)
            v_des_bar = torch.where(has_intra_u.unsqueeze(-1), v_desired[first_occ], v_des_bar)
            cos_v_u = F.cosine_similarity(v_old, v_des_bar, dim=-1)
            # 主格点仅保留 first_occ 样本（其余句内分叉样本由 _apply_batched_mitosis_spawns 写入新列并计数）
            hit_counts = torch.where(has_intra_u, torch.ones_like(hit_counts), hit_counts)

        existing_mitosis_u = enable_auto_mitosis & (~is_empty_u) & (cos_v_u < SIMPLEX_2_GEODESIC_PROJ)
        direct_update_u = ~existing_mitosis_u
        return self._apply_batched_kaczmarz_and_mitosis(
            pkg, unique_eids, inverse_indices, hit_counts, q_sphere_psi, v_desired, r_star,
            psi_bar, v_des_bar, u_old, v_old, is_empty_u, cos_v_u,
            direct_update_u, existing_mitosis_u, intra_fork_mask, first_occ,
            snap_rate, enable_auto_mitosis,
        )

    @torch.no_grad()
    def _apply_batched_kaczmarz_and_mitosis(
        self,
        pkg: PMCPackage,
        unique_eids: torch.Tensor,
        inverse_indices: torch.Tensor,
        hit_counts: torch.Tensor,
        psi: torch.Tensor,
        v_desired: torch.Tensor,
        r_star: torch.Tensor,
        psi_bar: torch.Tensor,
        v_des_bar: torch.Tensor,
        u_old: torch.Tensor,
        v_old: torch.Tensor,
        is_empty_u: torch.Tensor,
        cos_v_u: torch.Tensor,
        direct_update_u: torch.Tensor,
        existing_mitosis_u: torch.Tensor,
        intra_fork_mask: torch.Tensor,
        first_occ: torch.Tensor,
        snap_rate: float,
        enable_auto_mitosis: bool,
    ) -> Dict[str, Any]:
        # 4. 对所有非冲突唯一页 (空白页初始化 + 存量非空页 Kaczmarz & Rodrigues 旋转) 一次性张量批处理写入
        if bool(direct_update_u.any()):
            upd_eids = unique_eids[direct_update_u]
            upd_empty = is_empty_u[direct_update_u]
            u_old_d = u_old[direct_update_u]
            v_old_d = v_old[direct_update_u]
            psi_d = psi_bar[direct_update_u]
            v_des_d = v_des_bar[direct_update_u]
            cnt_d = hit_counts[direct_update_u]
            cos_d = cos_v_u[direct_update_u]

            h_old = pkg.activation_counts[upd_eids].float()
            h_m = h_old + cnt_d.float()
            r_m = pkg.purity_r[upd_eids].float()
            js_cap = 1.0 - SUBSPACE_STD_SCALE
            gamma_js = torch.where(
                h_m > 1.0,
                ((h_m * r_m.pow(2) - 1.0) / (h_m - 1.0).clamp_min(1.0)).clamp(0.0, js_cap),
                torch.zeros_like(h_m),
            ).unsqueeze(-1)

            res_v = v_des_d - v_old_d
            v_unit = F.normalize(v_old_d, dim=-1)
            res_parallel = (res_v * v_unit).sum(dim=-1, keepdim=True) * v_unit
            res_perp = res_v - res_parallel
            v_kacz = v_old_d + float(snap_rate) * (res_parallel + (1.0 - gamma_js) * res_perp)

            step_frac = (float(snap_rate) * torch.rsqrt(h_m.clamp_min(1.0))).clamp(max=1.0).unsqueeze(-1)
            u_unit_old = F.normalize(u_old_d, dim=-1)
            q_unit_d = F.normalize(psi_d, dim=-1)
            cos_th = (u_unit_old * q_unit_d).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            theta = torch.acos(cos_th) * step_frac
            w_perp = F.normalize(q_unit_d - cos_th * u_unit_old, dim=-1)
            u_rodrigues = F.normalize(torch.cos(theta) * u_unit_old + torch.sin(theta) * w_perp, dim=-1)

            u_final = torch.where(upd_empty.unsqueeze(-1), q_unit_d, u_rodrigues) * math.sqrt(float(D_BKP))
            v_final = torch.where(upd_empty.unsqueeze(-1), v_des_d, v_kacz)

            pkg.write_experts_to_mmap_pages(upd_eids, u_final, v_final)
            pkg.activation_counts[upd_eids] += cnt_d.to(pkg.activation_counts.dtype)
            # 冯·米塞斯-费舍尔 (vMF) 球面相干纯度精确样本加权递推 (零经验 EMA 魔数)
            vmf_purity = ((h_old * r_m + cnt_d.float() * cos_d.clamp_min(0.0)) / h_m.clamp_min(1.0)).clamp(
                SIMPLEX_3_BARYCENTER_PROJ, 1.0
            )
            new_pur = torch.where(
                upd_empty,
                torch.ones_like(r_m),
                vmf_purity,
            )
            pkg.purity_r[upd_eids] = new_pur

            upd_psi_h = F.normalize(psi_d.view(-1, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM), dim=-1)
            r_upd = torch.div(upd_eids, pkg.grid_k, rounding_mode="floor")
            c_upd = upd_eids % pkg.grid_k
            sf_sub = torch.where(upd_empty, torch.ones_like(h_m), step_frac.squeeze(-1)).view(1, -1, 1)
            u_cur_sub = F.normalize(pkg.u_codebook.data[:, r_upd, :].float(), dim=-1)
            v_cur_sub = F.normalize(pkg.v_codebook.data[:, c_upd, :].float(), dim=-1)
            u_tgt_sub = upd_psi_h[:, :, 0, :].permute(1, 0, 2)
            v_tgt_sub = upd_psi_h[:, :, 1, :].permute(1, 0, 2)
            cos_u = (u_cur_sub * u_tgt_sub).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            cos_v = (v_cur_sub * v_tgt_sub).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            th_u = torch.acos(cos_u) * sf_sub
            th_v = torch.acos(cos_v) * sf_sub
            wu_perp = F.normalize(u_tgt_sub - cos_u * u_cur_sub, dim=-1)
            wv_perp = F.normalize(v_tgt_sub - cos_v * v_cur_sub, dim=-1)
            pkg.u_codebook.data[:, r_upd, :] = F.normalize(torch.cos(th_u) * u_cur_sub + torch.sin(th_u) * wu_perp, dim=-1)
            pkg.v_codebook.data[:, c_upd, :] = F.normalize(torch.cos(th_v) * v_cur_sub + torch.sin(th_v) * wv_perp, dim=-1)
            pkg._u_cb_norm_cache = None
            pkg._v_cb_norm_cache = None
            self.vram_cache.invalidate_or_refresh(pkg, upd_eids)

        return self._apply_batched_mitosis_spawns(
            pkg, unique_eids, psi, v_desired, r_star, psi_bar, v_des_bar,
            direct_update_u, existing_mitosis_u, intra_fork_mask, first_occ, enable_auto_mitosis,
        )

    @torch.no_grad()
    def _apply_batched_mitosis_spawns(
        self,
        pkg: PMCPackage,
        unique_eids: torch.Tensor,
        psi: torch.Tensor,
        v_desired: torch.Tensor,
        r_star: torch.Tensor,
        psi_bar: torch.Tensor,
        v_des_bar: torch.Tensor,
        direct_update_u: torch.Tensor,
        existing_mitosis_u: torch.Tensor,
        intra_fork_mask: torch.Tensor,
        first_occ: torch.Tensor,
        enable_auto_mitosis: bool,
    ) -> Dict[str, Any]:
        dev = psi.device
        mitosis_psi_list: List[torch.Tensor] = []
        mitosis_v_list: List[torch.Tensor] = []
        mitosis_rows_list: List[torch.Tensor] = []
        if bool(existing_mitosis_u.any()):
            mit_eids_u = unique_eids[existing_mitosis_u]
            mitosis_psi_list.append(psi_bar[existing_mitosis_u])
            mitosis_v_list.append(v_des_bar[existing_mitosis_u])
            mitosis_rows_list.append(torch.div(mit_eids_u, pkg.grid_k, rounding_mode="floor"))
        if enable_auto_mitosis and bool(intra_fork_mask.any()):
            non_first_fork = intra_fork_mask.clone()
            non_first_fork[first_occ] = False
            if bool(non_first_fork.any()):
                mitosis_psi_list.append(psi[non_first_fork])
                mitosis_v_list.append(v_desired[non_first_fork])
                mitosis_rows_list.append(r_star[non_first_fork])

        mitosis_count = 0
        if len(mitosis_psi_list) > 0:
            m_psi = torch.cat(mitosis_psi_list, dim=0)
            m_v = torch.cat(mitosis_v_list, dim=0)
            m_rows = torch.cat(mitosis_rows_list, dim=0)
            mitosis_count = int(m_psi.size(0))

            empty_mask_2d = (pkg.scales[:, 1] <= 1e-6).view(pkg.grid_k, pkg.grid_k).clone()
            target_eids = torch.empty(mitosis_count, dtype=torch.long, device=dev)
            target_cols = torch.empty(mitosis_count, dtype=torch.long, device=dev)
            for m_i in range(mitosis_count):
                r_i = m_rows[m_i]
                avail_c = torch.nonzero(empty_mask_2d[r_i], as_tuple=False).squeeze(-1)
                if avail_c.numel() > 0:
                    c_sel = avail_c[0]
                    empty_mask_2d[r_i, c_sel] = False
                    target_cols[m_i] = c_sel
                    target_eids[m_i] = r_i * pkg.grid_k + c_sel
                else:
                    pkg._needs_quadrant_expand = True
                    any_emp = torch.nonzero(empty_mask_2d.view(-1), as_tuple=False).squeeze(-1)
                    t_eid = any_emp[0] if any_emp.numel() > 0 else (r_i * pkg.grid_k)
                    empty_mask_2d.view(-1)[t_eid] = False
                    target_cols[m_i] = t_eid % pkg.grid_k
                    target_eids[m_i] = t_eid

            m_u_scaled = F.normalize(m_psi, dim=-1) * math.sqrt(float(D_BKP))
            pkg.write_experts_to_mmap_pages(target_eids, m_u_scaled, m_v)
            pkg.activation_counts[target_eids] += 1
            pkg.purity_r[target_eids] = 1.0
            m_psi_h = F.normalize(m_psi.view(-1, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM), dim=-1)
            # 有丝分裂复用已命中的行子键 m_rows（保持同行存量微专家的行寻址零漂移），仅将新分配的伴随空列 target_cols 对齐到分支右半子键
            pkg.v_codebook.data[:, target_cols, :] = PMCPackage.rodrigues_geodesic_rotation(
                pkg.v_codebook.data[:, target_cols, :], m_psi_h[:, :, 1, :].permute(1, 0, 2), step_fraction=1.0
            )
            pkg._u_cb_norm_cache = None
            pkg._v_cb_norm_cache = None
            self.vram_cache.invalidate_or_refresh(pkg, target_eids)

        updated_count = int(direct_update_u.sum()) + mitosis_count
        self.total_kaczmarz_snaps += updated_count
        self.total_mitosis_spawns += mitosis_count
        return {
            "updated_pages": int(updated_count),
            "mitosis_pages": int(mitosis_count),
            "total_kaczmarz_snaps": int(self.total_kaczmarz_snaps),
            "total_mitosis_spawns": int(self.total_mitosis_spawns),
            "optimizer_memory_bytes": 0,
        }

    @torch.no_grad()
    def geodesic_kaczmarz_erase_downdate(
        self,
        q_curr: torch.Tensor,
        q_prev: torch.Tensor,
        q_target: torch.Tensor,
        erase_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Penrose v1.0 逆向拖拽擦除算子（Geodesic Kaczmarz Inverse Downdate & Cellular Apoptosis）
        """
        q_c = F.normalize(q_curr.detach().float().reshape(-1, D_BKP), dim=-1)
        q_p = F.normalize(q_prev.detach().float().reshape(-1, D_BKP), dim=-1)
        q_t = F.normalize(q_target.detach().float().reshape(-1, D_BKP), dim=-1)
        N = q_c.size(0)

        dq = q_c - (q_c * q_p).sum(dim=-1, keepdim=True) * q_c
        dq_norm = dq.norm(dim=-1, keepdim=True)
        has_tangent = (dq_norm > 1e-5).float()
        dq_unit = F.normalize(dq, dim=-1)
        psi = F.normalize(
            JET_NORMAL_ALPHA_STAR * q_c
            + has_tangent * JET_TANGENT_NORM_STAR * dq_unit
            + (1.0 - has_tangent) * (1.0 - JET_NORMAL_ALPHA_STAR) * q_c,
            dim=-1,
        )

        pkg = self.select_resonant_writable_package(psi)
        if pkg is None:
            return {
                "apoptosis_pages": 0,
                "peeled_shared_pages": 0,
                "severed_streamlines": 0,
                "optimizer_memory_bytes": 0,
            }

        v_tan = q_t - (q_t * q_c).sum(dim=-1, keepdim=True) * q_c
        v_erase = (1.0 / SIMPLEX_3_BARYCENTER_PROJ) * v_tan - SIMPLEX_3_BARYCENTER_PROJ * q_c

        q1_psi, q2_psi = self._standardize_subqueries(psi, update_center=False)
        u_cb, v_cb = pkg.normalized_codebooks()
        su = torch.einsum("nhd,hkd->nhk", q1_psi.float(), u_cb.float()).mean(dim=1)
        sv = torch.einsum("nhd,hkd->nhk", q2_psi.float(), v_cb.float()).mean(dim=1)
        r_star = su.argmax(dim=-1)
        c_star = sv.argmax(dim=-1)
        eids = r_star * pkg.grid_k + c_star
        return self._apply_batched_erase_downdate(pkg, r_star, eids, q1_psi, q2_psi, v_erase, erase_rate)

    @torch.no_grad()
    def _apply_batched_erase_downdate(
        self,
        pkg: PMCPackage,
        r_star: torch.Tensor,
        eids: torch.Tensor,
        q1_psi: torch.Tensor,
        q2_psi: torch.Tensor,
        v_erase: torch.Tensor,
        erase_rate: float,
    ) -> Dict[str, Any]:
        N = eids.size(0)
        dev = eids.device
        sv_2d = pkg.scales[:, 1].view(pkg.grid_k, pkg.grid_k)
        for r_val in torch.unique(r_star):
            r_int = int(r_val.item())
            act_cols = torch.nonzero(sv_2d[r_int] > 1e-6, as_tuple=False).squeeze(-1)
            if act_cols.numel() == 0:
                eids[r_star == r_val] = -1
                continue
            row_eids = r_int * pkg.grid_k + act_cols
            _, v_row = pkg.dequantize_expert_uv(row_eids)
            tok_mask = r_star == r_val
            v_er_sub = F.normalize(v_erase[tok_mask], dim=-1)
            cos_mat = torch.matmul(v_er_sub, F.normalize(v_row, dim=-1).t())
            best_cos, best_pos = cos_mat.max(dim=-1)
            matched_eids = row_eids[best_pos]
            eids[tok_mask] = torch.where(
                best_cos >= SIMPLEX_3_BARYCENTER_PROJ,
                matched_eids,
                torch.full_like(matched_eids, -1),
            )

        severed_streamlines = 0
        if N > 1:
            diff_step = (eids[:-1] != eids[1:]) & (eids[:-1] >= 0) & (eids[1:] >= 0)
            if bool(diff_step.any()):
                severed_streamlines = pkg.sever_streamline_transition(eids[:-1][diff_step], eids[1:][diff_step])

        valid_tok_mask = eids >= 0
        if not bool(valid_tok_mask.any()):
            return {
                "apoptosis_pages": 0,
                "peeled_shared_pages": 0,
                "severed_streamlines": int(severed_streamlines),
                "optimizer_memory_bytes": 0,
            }

        q_sphere_psi = F.normalize(torch.stack([q1_psi, q2_psi], dim=2).reshape(N, D_BKP).float(), dim=-1)
        unique_eids, inverse_indices, hit_counts = torch.unique(
            eids[valid_tok_mask], sorted=True, return_inverse=True, return_counts=True
        )
        u_cnt = int(unique_eids.numel())
        psi_sum = torch.zeros((u_cnt, D_BKP), dtype=torch.float32, device=dev)
        v_er_sum = torch.zeros((u_cnt, D_BKP), dtype=torch.float32, device=dev)
        psi_sum.index_add_(0, inverse_indices, q_sphere_psi[valid_tok_mask])
        v_er_sum.index_add_(0, inverse_indices, v_erase[valid_tok_mask])
        inv_cnt = (1.0 / hit_counts.float()).unsqueeze(-1)
        psi_bar = F.normalize(psi_sum * inv_cnt, dim=-1)
        v_er_bar = v_er_sum * inv_cnt

        h_old = pkg.activation_counts[unique_eids].float()
        delta_h = hit_counts.float() * float(erase_rate)
        apoptosis_mask = h_old <= delta_h + 1e-5
        shared_mask = ~apoptosis_mask

        apoptosis_cnt = 0
        if bool(apoptosis_mask.any()):
            apop_eids = unique_eids[apoptosis_mask]
            apoptosis_cnt = pkg.apoptosis_expert_pages(apop_eids)
            # 细胞凋亡后重锚定受影响行/列的子键码本至同行/同列剩余存活微专家的页内键重心，消除已擦除分支对存量微专家寻址的残余牵引
            sv_after = pkg.scales[:, 1].view(pkg.grid_k, pkg.grid_k)
            apop_rows = torch.unique(torch.div(apop_eids, pkg.grid_k, rounding_mode="floor"))
            apop_cols = torch.unique(apop_eids % pkg.grid_k)
            for r_v in apop_rows:
                ri = int(r_v.item())
                surv_c = torch.nonzero(sv_after[ri] > 1e-6, as_tuple=False).squeeze(-1)
                if surv_c.numel() > 0:
                    u_surv, _ = pkg.dequantize_expert_uv(ri * pkg.grid_k + surv_c)
                    u_h = F.normalize(u_surv.view(-1, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM).mean(dim=0), dim=-1)
                    pkg.u_codebook.data[:, ri, :] = u_h[:, 0, :]
            for c_v in apop_cols:
                ci = int(c_v.item())
                surv_r = torch.nonzero(sv_after[:, ci] > 1e-6, as_tuple=False).squeeze(-1)
                if surv_r.numel() > 0:
                    u_surv, _ = pkg.dequantize_expert_uv(surv_r * pkg.grid_k + ci)
                    u_h = F.normalize(u_surv.view(-1, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM).mean(dim=0), dim=-1)
                    pkg.v_codebook.data[:, ci, :] = u_h[:, 1, :]
            pkg._u_cb_norm_cache = None
            pkg._v_cb_norm_cache = None
        peeled_cnt = 0
        if bool(shared_mask.any()):
            sh_eids = unique_eids[shared_mask]
            u_old_s, v_old_s = pkg.dequantize_expert_uv(sh_eids)
            psi_s = psi_bar[shared_mask]
            v_er_s = v_er_bar[shared_mask]
            h_old_s = h_old[shared_mask]
            dh_s = delta_h[shared_mask]

            # 正向 Kaczmarz 的精确代数逆解：
            # 1) 若共享微专家上 v_erase 与存量 v_old 高度同向 (cos >= sqrt(2/3)，如共用的语法/格式转移 { -> text)，
            #    正向写入时残差 res_v = v_des - v_old == 0，仅增加了引用计数 H_m；故逆算子仅需回退计数 H_m -= dh_s，严禁削减主干模长！
            # 2) 若共享微专家存在方向折中混合 (1/sqrt(3) <= cos < sqrt(2/3))，则按权重 alpha_peel 剥离偏转分量并保持原模长守恒。
            v_old_norm = v_old_s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            v_old_unit = v_old_s / v_old_norm
            v_er_unit = F.normalize(v_er_s, dim=-1)
            cos_ve = (v_old_unit * v_er_unit).sum(dim=-1, keepdim=True)
            is_divergent_mix = (cos_ve < JET_NORMAL_ALPHA_STAR).float()
            alpha_peel = (dh_s / h_old_s.clamp_min(1.0)).clamp(0.0, 1.0 - 1e-4).unsqueeze(-1) * is_divergent_mix

            v_perp_er = v_er_unit - cos_ve * v_old_unit
            v_peeled_dir = F.normalize(v_old_unit - alpha_peel * v_perp_er, dim=-1)
            v_peeled = v_peeled_dir * v_old_norm

            u_unit_old = F.normalize(u_old_s, dim=-1)
            cos_th = (u_unit_old * psi_s).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            theta_back = -torch.acos(cos_th) * alpha_peel
            w_perp = F.normalize(psi_s - cos_th * u_unit_old, dim=-1)
            u_rev = F.normalize(torch.cos(theta_back) * u_unit_old + torch.sin(theta_back) * w_perp, dim=-1)
            pkg.write_experts_to_mmap_pages(sh_eids, u_rev * math.sqrt(float(D_BKP)), v_peeled)
            pkg.activation_counts[sh_eids] = (h_old_s - dh_s).clamp_min(1.0).round().to(pkg.activation_counts.dtype)
            peeled_cnt = int(sh_eids.numel())

        self.vram_cache.invalidate_or_refresh(pkg, unique_eids)
        return {
            "apoptosis_pages": int(apoptosis_cnt),
            "peeled_shared_pages": int(peeled_cnt),
            "severed_streamlines": int(severed_streamlines),
            "optimizer_memory_bytes": 0,
        }

    @torch.no_grad()
    def spawn_specialist_package_from_queries(
        self,
        new_pkg_id: str,
        grid_k: int = 64,
        mmap_file_path: Optional[str] = None,
        mount_mode: PMCMountMode = PMCMountMode.RW_SHARED,
        init_scale_v: float = 1e-5,
    ) -> PMCPackage:
        """
        自治拓扑算子 6: Auto-Spawn (球面未知流形新包孵化，返回 PMCPackage)
        """
        base_spawned = super().spawn_specialist_package_from_queries(
            new_pkg_id=new_pkg_id,
            grid_k=grid_k,
            mmap_file_path=None,
            mount_mode=mount_mode,
            init_scale_v=init_scale_v,
        )
        dev = base_spawned.u_codebook.device
        pmc_pkg = PMCPackage(
            pkg_id=new_pkg_id,
            grid_k=grid_k,
            mount_mode=mount_mode,
            is_base_pkg=base_spawned.is_base_pkg,
            mmap_file_path=mmap_file_path,
            device=dev,
        )
        pmc_pkg._copy_manifold_stats_from(base_spawned)
        pmc_pkg.codebook_pre_centered.copy_(base_spawned.codebook_pre_centered)
        pmc_pkg._codebook_pre_centered_cached = base_spawned._codebook_pre_centered_cached
        pmc_pkg.u_codebook.data.copy_(base_spawned.u_codebook.data)
        pmc_pkg.v_codebook.data.copy_(base_spawned.v_codebook.data)
        pmc_pkg.expert_pages_int8.copy_(base_spawned.expert_pages_int8)
        pmc_pkg.scales.copy_(base_spawned.scales)
        pmc_pkg.activation_counts.copy_(base_spawned.activation_counts)
        pmc_pkg.purity_r.copy_(base_spawned.purity_r)
        pmc_pkg._rebuild_runtime_buffers_after_topology_change()

        self.unmount_package(new_pkg_id)
        self.mount_package(pmc_pkg, pin_to_l1=True)
        return pmc_pkg

    @torch.no_grad()
    def snapshot_forest_state(self, step: int = 0) -> Dict[str, Any]:
        pkgs_info: List[Dict[str, Any]] = []
        for pid, pkg in self.packages.items():
            woken = int((pkg.activation_counts > 0).sum().item())
            active_sv = int((pkg.scales[:, 1] > 1e-6).sum().item())
            poly_cnt = int(pkg.detect_polysemy_interference_mask().sum().item())
            stream_edges = int((pkg.stream_next_ptr >= 0).sum().item()) if isinstance(pkg, PMCPackage) else 0
            pkgs_info.append({
                "pkg_id": pkg.pkg_id,
                "is_pmc_package": isinstance(pkg, PMCPackage),
                "grid_k": int(pkg.grid_k),
                "num_experts": int(pkg.num_experts),
                "woken_experts": woken,
                "active_scale_v_experts": active_sv,
                "woken_ratio": round(woken / float(max(1, pkg.num_experts)), 4),
                "polysemy_conflict_experts": poly_cnt,
                "streamline_edges": stream_edges,
                "in_l1_tlb": bool(pkg.pkg_id in self.l1_active_pkg_ids),
                "is_bridge_pkg": bool(pkg.is_bridge_pkg),
            })
        return {
            "step": int(step),
            "total_mounted_packages": len(self.packages),
            "l1_active_pkg_ids": list(self.l1_active_pkg_ids),
            "operator_totals": {
                "split_count": int(self.split_count),
                "expand_count": int(self.expand_count),
                "merge_count": int(self.merge_count),
                "spawn_count": int(self.spawn_count),
                "closure_count": int(self.closure_count),
                "mitosis_spawns": int(self.total_mitosis_spawns),
                "kaczmarz_snaps": int(self.total_kaczmarz_snaps),
            },
            "packages": pkgs_info,
        }

    @torch.no_grad()
    def step_autonomous_forest_evolution(
        self,
        step: int,
        q_probe: Optional[torch.Tensor] = None,
        domain_hint: str = "general",
        max_grid_k: int = 256,
        max_total_packages: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Penrose v1.0 内聚自治森林六大演化算子调度引擎（100% 零超参数闭式几何触发判据）：
          1. Auto-Split   : 公理 10 二分图拉普拉斯谱双聚类裂变 (cross_ratio <= 1 - 1/sqrt(2)) + BridgeAB
          2. Auto-Expand  : 公理 9 四象限 (a+b)^2 无损倍增扩容 (woken_ratio >= 1 - exp(-1/sqrt(2)) 或同行满载)
          3. Auto-Merge   : 公理 9 跨包三元正交聚变 (coactivation_ratio >= sqrt(2) - 1)
          4. Auto-Spawn   : 公理 2 冈贝尔极值尾概率未知流形新包孵化 (low_hit_ratio >= 0.5*erfc(tau_z*(64)/sqrt(2)))
          5. Auto-Closure : Insight 5 双正交笛卡尔闭包成核 (R^1024 (+) R^1024 交叉格点结晶)
          6. Auto-Mitosis : 公理 7 & 8 切向复共轭有丝分裂 (H_m*R_m^2 > ln(N) 且 R_m < 1/sqrt(2))
        """
        events: List[Dict[str, Any]] = []
        split_cross_max = 1.0 - SIMPLEX_2_GEODESIC_PROJ
        expand_woken_thresh = 1.0 - math.exp(-SIMPLEX_2_GEODESIC_PROJ)
        merge_coact_min = (1.0 / SIMPLEX_2_GEODESIC_PROJ) - 1.0
        tau_64 = closed_form_gumbel_threshold(64.0)
        spawn_tail_min = 0.5 * math.erfc(tau_64 / math.sqrt(2.0))

        # 1. Auto-Split (谱双聚类裂变)
        for pid in list(self.l1_active_pkg_ids):
            if len(self.packages) >= max_total_packages:
                break
            mk = self._mod_key(pid)
            if mk not in self.packages:
                continue
            pkg = self.packages[mk]
            if pkg.is_bridge_pkg or pkg.grid_k < 128:
                continue
            woken_cnt = int((pkg.activation_counts > 0).sum().item())
            if woken_cnt < max(16, pkg.grid_k // 2):
                continue
            split_res = pkg.bkp_split_spectral(cross_ratio_threshold=split_cross_max, return_bridge=True)
            if split_res is not None:
                pkg_a, pkg_b, pkg_bridge = split_res
                self.split_count += 1
                stem = pid.replace(".pmc", "").replace(".bkp", "")
                pkg_a.pkg_id = f"{stem}_SubA_s{step}.pmc"
                pkg_b.pkg_id = f"{stem}_SubB_s{step}.pmc"
                pkg_a.is_base_pkg = pkg.is_base_pkg
                pkg_b.is_base_pkg = False
                self.unmount_package(pid)
                self.mount_package(pkg_a, pin_to_l1=True)
                self.mount_package(pkg_b, pin_to_l1=True)
                bridge_id = None
                if pkg_bridge is not None:
                    bridge_id = f"{stem}_BridgeAB_s{step}.pmc"
                    pkg_bridge.pkg_id = bridge_id
                    pkg_bridge.is_bridge_pkg = True
                    self.mount_package(pkg_bridge, pin_to_l1=True)
                ev = {
                    "step": int(step),
                    "op": "Auto-Split",
                    "parent_pkg": pid,
                    "child_a": pkg_a.pkg_id,
                    "child_b": pkg_b.pkg_id,
                    "bridge_pkg": bridge_id,
                    "grid_k": f"{pkg.grid_k} -> ({pkg_a.grid_k}, {pkg_b.grid_k})",
                }
                events.append(ev)
                self.forest_events.append(ev)
                break

        # 2. Auto-Expand (四象限倍增扩容)
        if len(events) == 0:
            for pid in list(self.l1_active_pkg_ids):
                mk = self._mod_key(pid)
                if mk not in self.packages:
                    continue
                pkg = self.packages[mk]
                if pkg.is_bridge_pkg or pkg.grid_k >= max_grid_k:
                    continue
                woken_ratio = float((pkg.activation_counts > 0).float().mean().item())
                poly_cnt = int(pkg.detect_polysemy_interference_mask().sum().item())
                needs_exp = getattr(pkg, "_needs_quadrant_expand", False)
                if woken_ratio >= expand_woken_thresh or poly_cnt >= pkg.grid_k or needs_exp:
                    old_k = pkg.grid_k
                    pkg.expand_quadrant_2x()
                    self.vram_cache.evict_package(pkg.pkg_id)
                    self.expand_count += 1
                    ev = {
                        "step": int(step),
                        "op": "Auto-Expand",
                        "pkg_id": pkg.pkg_id,
                        "old_grid_k": int(old_k),
                        "new_grid_k": int(pkg.grid_k),
                        "woken_ratio_before": round(woken_ratio, 4),
                        "polysemy_conflicts": int(poly_cnt),
                    }
                    events.append(ev)
                    self.forest_events.append(ev)
                    break

        return self._step_autonomous_forest_evolution_part2(
            step, events, merge_coact_min, spawn_tail_min, q_probe, domain_hint, max_grid_k, max_total_packages
        )

    @torch.no_grad()
    def _step_autonomous_forest_evolution_part2(
        self,
        step: int,
        events: List[Dict[str, Any]],
        merge_coact_min: float,
        spawn_tail_min: float,
        q_probe: Optional[torch.Tensor],
        domain_hint: str,
        max_grid_k: int,
        max_total_packages: int,
    ) -> List[Dict[str, Any]]:
        # 3. Auto-Merge (跨包三元正交聚变)
        if len(events) == 0 and len(self.pkg_coactivation_counts) > 0:
            best_pair: Optional[Tuple[str, str]] = None
            best_ratio = 0.0
            for (pa, pb), co_cnt in self.pkg_coactivation_counts.items():
                if self._mod_key(pa) not in self.packages or self._mod_key(pb) not in self.packages:
                    continue
                pkg_a = self.packages[self._mod_key(pa)]
                pkg_b = self.packages[self._mod_key(pb)]
                if pkg_a.is_bridge_pkg or pkg_b.is_bridge_pkg or (pkg_a.grid_k + pkg_b.grid_k > max_grid_k):
                    continue
                denom = min(self.pkg_token_hit_counts.get(pa, 0.0), self.pkg_token_hit_counts.get(pb, 0.0))
                if denom < 4.0:
                    continue
                ratio = co_cnt / max(denom, 1.0)
                if ratio >= merge_coact_min and ratio > best_ratio:
                    best_ratio = ratio
                    best_pair = (pa, pb)
            if best_pair is not None:
                pa, pb = best_pair
                pkg_a = self.packages[self._mod_key(pa)]
                pkg_b = self.packages[self._mod_key(pb)]
                merged_id = f"TriadicMerge_{self.merge_count + 1}_s{step}.pmc"
                merged_pkg = PMCPackage.bkp_merge_triadic_bundle(pkg_a, pkg_b, merged_id=merged_id)
                self.unmount_package(pa)
                self.unmount_package(pb)
                self.mount_package(merged_pkg, pin_to_l1=True)
                self.merge_count += 1
                ev = {
                    "step": int(step),
                    "op": "Auto-Merge",
                    "pkg_a": pa,
                    "pkg_b": pb,
                    "merged_pkg": merged_id,
                    "coactivation_ratio": round(best_ratio, 4),
                    "new_grid_k": int(merged_pkg.grid_k),
                }
                events.append(ev)
                self.forest_events.append(ev)

        # 4. Auto-Spawn (未知流形新包孵化)
        if len(events) == 0 and len(self.packages) < max_total_packages:
            total_q = max(1, int(getattr(self, "_total_periodic_queries", 0)))
            low_q = int(getattr(self, "_low_hit_periodic_queries", 0))
            low_ratio = low_q / float(total_q)
            has_pool = self._low_hit_q1_reservoir is not None and self._low_hit_q1_reservoir.size(0) >= 8
            if has_pool and low_ratio >= spawn_tail_min:
                self.spawn_count += 1
                new_id = f"Spec_{domain_hint}_{self.spawn_count}_s{step}.pmc"
                new_pkg = self.spawn_specialist_package_from_queries(new_pkg_id=new_id, grid_k=64)
                ev = {
                    "step": int(step),
                    "op": "Auto-Spawn",
                    "new_pkg": new_pkg.pkg_id,
                    "domain": domain_hint,
                    "low_hit_ratio": round(low_ratio, 4),
                    "grid_k": int(new_pkg.grid_k),
                }
                events.append(ev)
                self.forest_events.append(ev)

        # 5. Auto-Closure (双正交笛卡尔闭包成核) & 6. Auto-Mitosis (存量切向复共轭叶状分裂)
        for pid in list(self.l1_active_pkg_ids):
            mk = self._mod_key(pid)
            if mk not in self.packages:
                continue
            pkg = self.packages[mk]
            if pkg.is_bridge_pkg:
                continue
            c_info = pkg.crystallize_biorthogonal_manifold_closure(max_crystallize_slots=64)
            n_cryst = int(c_info.get("crystallized_count", 0))
            if n_cryst > 0:
                self.closure_count += n_cryst
                ev = {
                    "step": int(step),
                    "op": "Auto-Closure",
                    "pkg_id": pkg.pkg_id,
                    "crystallized_count": n_cryst,
                }
                events.append(ev)
                self.forest_events.append(ev)

            m_info = pkg.nucleate_tangential_conjugate_mitosis(max_mitosis_slots=64)
            n_mit = int(m_info.get("mitosis_count", 0))
            if n_mit > 0:
                self.total_mitosis_spawns += n_mit
                ev = {
                    "step": int(step),
                    "op": "Auto-Mitosis",
                    "pkg_id": pkg.pkg_id,
                    "mitosis_count": n_mit,
                    "mean_conjugate_cos": round(float(m_info.get("mean_conjugate_cos", -1.0)), 4),
                }
                events.append(ev)
                self.forest_events.append(ev)

        if q_probe is not None and len(self.packages) > len(self.l1_active_pkg_ids):
            self.step_l2_page_walker(q_probe.detach())

        self.reset_coactivation_stats()
        return events


