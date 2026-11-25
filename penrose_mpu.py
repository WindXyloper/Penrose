"""
Penrose v1.0 — MPU (Manifold Processing Unit, `penrose_mpu.py`)
===============================================================
严格遵循五段式流水线（Module 1 ~ Module 5）的纯血 2048 维球面流形状态推进器：
- 彻底废除 `nn.Embedding` 词表查表层与 `LM Head` 词表分类层；
- 严格实现「2048 维球面向量进 -> 2048 维球面向量出」的 12 层（8 注意力头）状态推进器：
  * Module 1 (1-Jet Ingress, 0 参数)     : (q(tau), q(tau-1)) -> psi^{(0)}(tau) in S^{2047} + 触发早一轮 PMC 预取
  * Module 2 (Lower 6 Blocks, 8-Head)    : 扫描历史轨迹提取上下文向量割线 P(Gamma) -> psi^{(6)}(tau)
  * Module 3 (PMC Waist Coupler)         : 单位残差门控 psi^{(6)}(tau) * (1 + tanh(W_delta * V_PMC)) -> tilde{psi}^{(6)}(tau)
  * Module 4 (Upper 6 Blocks, 8-Head)    : 轨迹曲率精炼与非阿贝尔李括号耦合 -> 切向推力 Delta_q(tau)
  * Module 5 (Geodesic ResNet Egress)    : 切空间正交化 + 跨核 ResNet 闭环 q(tau+1) = P_{S^{2047}}(q(tau) + Delta_q_perp(tau))
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from bicamera_kernel_draft import (
    BicameraConfig,
    RMSNorm,
    DualSubspaceRoPE,
    HybridDiffSoftmaxAttention,
    OrthogonalMetaMoE,
    BicameraDecoderLayer,
)
from penrose_pmc import (
    D_BKP,
    JET_NORMAL_ALPHA_STAR,
    JET_TANGENT_NORM_STAR,
    PMCManager,
    HelmholtzFieldDiagnostics,
)


@dataclass
class PenroseMPUConfig(BicameraConfig):
    """
    Penrose v1.0 MPU 标准配置：
    - 物理流形总线维度恒锁死为 `d_bkp = 2048`（对应 S^{2047} 单位超球面）；
    - 默认 12 层（前 6 层 Module 2 + 腰部 Module 3 + 后 6 层 Module 4），8 个混合注意力头（4 差分头 + 4 标准头）。
    """
    model_name: str = "Penrose-MPU-v1.0"
    vocab_size: int = 0                # Penrose v1.0 彻底废除静态词表，vocab_size=0
    d_model: int = 2048                # 主干原生维度直接与 2048 维球面状态对齐（或支持轻量正交标架）
    n_layers: int = 12                 # Module 2 (Layers 0..5) + Module 4 (Layers 6..11)
    n_heads: int = 8                   # 8 个混合注意力头 (4 差分 + 4 标准)
    n_kv_heads: int = 4
    head_dim: int = 256                # 8 * 256 = 2048
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    moe_n_routed_experts: int = 8
    moe_top_k: int = 8
    moe_inter_dim: int = 512
    moe_full_active_orthogonal: bool = True
    d_bkp: int = D_BKP                 # 2048
    bkp_bus_layer: int = 6             # 精确位于第 6 层与第 7 层之间的腰部（0-indexed: Layer 6 入口）
    bkp_highway_layers: List[int] = field(default_factory=lambda: [7, 9, 11])
    gradient_checkpointing: bool = False

    @classmethod
    def preset_penrose_full_2048(cls) -> "PenroseMPUConfig":
        """原生 2048 维 12 层 8 头标准 MPU 配置（d_model=2048, 8x256 注意力头）"""
        return cls(
            model_name="Penrose-MPU-v1.0-2048",
            d_model=2048,
            n_layers=12,
            n_heads=8,
            n_kv_heads=4,
            head_dim=256,
            moe_n_routed_experts=8,
            moe_top_k=8,
            moe_inter_dim=512,
            bkp_bus_layer=6,
        )

    @classmethod
    def preset_penrose_agile_12l(cls) -> "PenroseMPUConfig":
        """
        敏捷单卡 A100/4060 12 层 8 头 MPU 配置：
        外部接口严格为 2048 维球面向量进出 (q(tau) in S^{2047})，
        内部 12 层 8 头算子使用正交保距 Stiefel 标架以极致吞吐运行。
        """
        return cls(
            model_name="Penrose-MPU-v1.0-Agile-12L",
            d_model=512,
            n_layers=12,
            n_heads=8,
            n_kv_heads=4,
            head_dim=64,
            moe_n_routed_experts=8,
            moe_top_k=8,
            moe_inter_dim=384,
            bkp_bus_layer=6,
            bkp_highway_layers=[7, 9, 11],
        )


class MPUModule1JetIngress(nn.Module):
    """
    模块 1：入口——1-Jet 相空间合成器（1-Jet Ingress，0 可学习参数）
    - 彻底取代老版 Kernel 第 0 层的 Token Embedding 查表层；
    - 输入当前球面位置向量 q(tau) in S^{2047} 与上一拍位置 q(tau-1) in S^{2047}；
    - 调用公理 5（1-Jet 正交分解）合成为第 0 层 2048 维单位向量 psi^{(0)}(tau)：
          hat{Delta_q}_perp(tau) = Normalize(q(tau) - <q(tau), q(tau-1)> q(tau))
          psi^{(0)}(tau) = sqrt(2/3) * q(tau) + sqrt(1/3) * hat{Delta_q}_perp(tau) in S^{2047}
    - 同时在合成瞬间触发 PMC 异步流线预取！
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q_curr: torch.Tensor,
        q_prev: Optional[torch.Tensor] = None,
        pmc_manager: Optional[PMCManager] = None,
    ) -> torch.Tensor:
        """
        q_curr: (B, T, 2048) 当前时刻球面状态序列
        q_prev: (B, T, 2048) 上一时刻球面状态序列（若为 None，则自动由序列右移一位构造 q(tau-1)）
        返回: psi_0: (B, T, 2048) in S^{2047}
        """
        q_c = F.normalize(q_curr.float(), dim=-1)
        if q_prev is None:
            # 序列并行模式：第 0 拍的 q_prev 等于自身（切向速度为 0），第 tau 拍的 q_prev 为 q(tau-1)
            q_p = torch.cat([q_c[:, :1, :], q_c[:, :-1, :]], dim=1)
        else:
            q_p = F.normalize(q_prev.float(), dim=-1)

        inner = (q_c * q_p).sum(dim=-1, keepdim=True)
        dq_perp = q_c - inner * q_c
        dq_norm = dq_perp.norm(dim=-1, keepdim=True)
        has_motion = (dq_norm > 1e-5).float()
        dq_unit = F.normalize(dq_perp, dim=-1)

        # 公理 5：当存在切向位移时按 sqrt(2/3) q + sqrt(1/3) dq_unit 合成；静止初态退化为纯法向 q_c
        psi_0 = F.normalize(
            has_motion * (JET_NORMAL_ALPHA_STAR * q_c + JET_TANGENT_NORM_STAR * dq_unit)
            + (1.0 - has_motion) * q_c,
            dim=-1,
        ).to(q_curr.dtype)

        # 工程顺带红利（早一轮预取触发点）：在 Module 1 合成 psi^{(0)}(tau) 的瞬间发起 PMC 异步预取
        if pmc_manager is not None:
            pmc_manager.prefetch_along_geodesic_streamline(psi_0)

        return psi_0


class MPUModule3WaistCoupler(nn.Module):
    """
    模块 3：腰部——PMC 流形场耦合门（Waist Manifold Coupler）
    - 将前 6 层（Module 2）输出的上下文割线状态 psi^{(6)}(tau) 投影至 2048 维单位超球面；
    - 唤醒 PMC 四叉树中的局部单纯形微专家（由十条闭式公理驱动），合成流形场切向推力与旋度偏转向量
      Delta_PMC(tau) = V_PMC(psi^{(6)}(tau)) in R^{2048}；
    - 通过腰部投影矩阵 W_delta 与单位残差门控注入主干状态：
          tilde{psi}^{(6)}(tau) = psi^{(6)}(tau) * (1.0 + tanh(W_delta * Delta_PMC(tau)))
    """

    def __init__(self, config: PenroseMPUConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_bkp = config.d_bkp
        self.bus_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.w_delta = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.orthogonal_(self.w_delta.weight, gain=0.25)

        if self.d_model != self.d_bkp:
            self.p_in = nn.Parameter(torch.empty(self.d_model, self.d_bkp))
            nn.init.orthogonal_(self.p_in)
            self.p_out = nn.Parameter(self.p_in.t().contiguous().clone())
        else:
            self.register_parameter("p_in", None)
            self.register_parameter("p_out", None)

    @torch.no_grad()
    def project_stiefel_frames(self) -> None:
        """
        当采用 Agile 降维标架 (d_model < 2048) 时，将 p_in^T 严格重投影回 Stiefel 正交流形 O(2048, d_model)，
        并同步对偶转置矩阵 p_out = p_in^T，消除长周期训练下的正交群漂移；
        在原生 preset_penrose_full_2048 (d_model = 2048) 下 p_in/p_out 为 None，零开销直接返回。
        """
        if self.p_in is None or self.p_out is None:
            return
        w_tall = self.p_in.detach().float().t()  # (2048, d_model)
        q_orth, r_mat = torch.linalg.qr(w_tall, mode="reduced")
        diag_sign = torch.sgn(torch.diagonal(r_mat)).unsqueeze(0)
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        q_orth = (q_orth * diag_sign).to(self.p_in.dtype)
        self.p_in.copy_(q_orth.t().contiguous())
        self.p_out.copy_(q_orth.contiguous())

    def to_manifold_2048(self, h: torch.Tensor, psi_0_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_n = self.bus_norm(h)
        if self.p_in is None:
            q_2048 = F.normalize(h_n.float(), dim=-1)
        else:
            scale = math.sqrt(float(self.d_bkp) / float(self.d_model))
            q_2048 = F.normalize((torch.matmul(h_n, self.p_in) * scale).float(), dim=-1)
        if psi_0_anchor is not None:
            q_2048 = F.normalize(psi_0_anchor.float() + 0.12 * q_2048, dim=-1)
        return q_2048.to(h.dtype)

    def to_model_space(self, delta_2048: torch.Tensor) -> torch.Tensor:
        if self.p_out is None:
            return delta_2048
        return torch.matmul(delta_2048, self.p_out)

    def forward(
        self,
        psi_6: torch.Tensor,
        psi_0_anchor: Optional[torch.Tensor] = None,
        pmc_manager: Optional[PMCManager] = None,
        pmc_dropout: bool = False,
        puncture_dropout_prob: float = 0.0,
        puncture_expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, HelmholtzFieldDiagnostics]:
        """
        返回:
        - psi_6_coupled : (B, T, d_model) 经单位残差门控耦合后的腰部状态 tilde{psi}^{(6)}(tau)
        - z_highway     : (B, T, d_model) 供 Module 4 (后 6 层) 双线性元算子读取的流形载荷向量
        - diag          : HelmholtzFieldDiagnostics（含 div_v, eta_curl, delta_pmc_2048 等）
        """
        B, T, _ = psi_6.shape
        q_waist_2048 = self.to_manifold_2048(psi_6, psi_0_anchor=psi_0_anchor)

        if pmc_dropout or pmc_manager is None:
            zero_d = torch.zeros_like(psi_6)
            zero_2048 = torch.zeros((B, T, self.d_bkp), device=psi_6.device, dtype=psi_6.dtype)
            zero_s = torch.zeros((B, T), device=psi_6.device, dtype=torch.float32)
            empty_i = torch.zeros((B, T, 1), device=psi_6.device, dtype=torch.long)
            diag = HelmholtzFieldDiagnostics(
                delta_pmc=zero_2048,
                div_v=zero_s,
                eta_curl=torch.ones_like(zero_s),
                active_experts=empty_i,
                active_weights=zero_s.unsqueeze(-1),
                max_resonance=zero_s,
            )
            return psi_6, zero_d, diag

        diag = pmc_manager.query_with_helmholtz_diagnostics(
            q_waist_2048,
            puncture_dropout_prob=puncture_dropout_prob,
            puncture_expert_mask=puncture_expert_mask,
        )
        delta_model = self.to_model_space(diag.delta_pmc.to(psi_6.dtype))
        z_rms = delta_model * torch.rsqrt(delta_model.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(psi_6.dtype)
        # 单位残差门控（Unit Residual Gating，保证 Delta_PMC=0 时恒等直通）
        psi_6_coupled = psi_6 * (1.0 + torch.tanh(self.w_delta(z_rms)))
        return psi_6_coupled, delta_model, diag


class MPUModule5GeodesicEgress(nn.Module):
    """
    模块 5：出口——跨核球面 ResNet 闭环器（Geodesic Residual Egress，取代原 LM Head）
    - 彻底砍掉老版最后一层庞大的 LM Head (Linear 2048 -> Vocab_Size) 和 Softmax；
    - 输入模块 1 的入口球面位置 q(tau) in S^{2047} 与模块 4（第 12 层）精炼出的推力 Delta_q(tau) in R^{2048}：
          1. 切空间正交化: Delta_q_perp(tau) = Delta_q(tau) - <Delta_q(tau), q(tau)> q(tau)
          2. 跨核 ResNet 球面闭环输出: q(tau+1) = P_{S^{2047}}(q(tau) + Delta_q_perp(tau)) in S^{2047}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q_curr: torch.Tensor,
        delta_q_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_curr     : (B, T, 2048) 入口球面单位向量 q(tau) in S^{2047}
        delta_q_raw: (B, T, 2048) 第 12 层输出的切向位移推力向量 Delta_q(tau)
        返回:
        - q_next      : (B, T, 2048) 下一拍球面状态向量 q(tau+1) in S^{2047}
        - delta_q_perp: (B, T, 2048) 正交切空间净推力向量 Delta_q_perp(tau)
        """
        q_c = F.normalize(q_curr.float(), dim=-1)
        dq = delta_q_raw.float()
        # 1. 切空间正交化：消除虚耗在法向拉伸上的分量
        radial_proj = (dq * q_c).sum(dim=-1, keepdim=True) * q_c
        dq_perp = dq - radial_proj
        # 2. 跨核球面 ResNet 残差闭环
        q_next = F.normalize(q_c + dq_perp, dim=-1).to(q_curr.dtype)
        return q_next, dq_perp.to(q_curr.dtype)


@dataclass
class MPUStepOutput:
    """
    MPU 单次（或全序列并行）推进输出包：
    - q_next        : (B, T, 2048) Module 5 输出的下一拍单位球面向量 q(tau+1) in S^{2047}
    - psi_0         : (B, T, 2048) Module 1 合成的 1-Jet 相空间入射向量 psi^{(0)}(tau)
    - delta_q_perp  : (B, T, 2048) Module 4 & 5 精炼出的正交切空间推力 Delta_q_perp(tau)
    - helmholtz_diag: HelmholtzFieldDiagnostics（含局部散度 div_v、旋度比 eta_curl，直接驱动三态总线分支）
    - loss          : Optional[torch.Tensor] 当处于分支 C（并行训练态）时计算的球面弦长误差
    """
    q_next: torch.Tensor
    psi_0: torch.Tensor
    delta_q_perp: torch.Tensor
    helmholtz_diag: HelmholtzFieldDiagnostics
    loss: Optional[torch.Tensor] = None


class PenroseMPU(nn.Module):
    """
    Penrose v1.0 核心流形处理器（Manifold Processing Unit）：
    严格五段式流水线：
      (q(tau), q(tau-1))
        --[Module 1: 1-Jet Ingress (0 params)]--> psi^{(0)}(tau)
        --[Module 2: Lower 6 Blocks (8-Head)]---> psi^{(6)}(tau)
        --[Module 3: PMC Waist Coupler]---------> tilde{psi}^{(6)}(tau)
        --[Module 4: Upper 6 Blocks (8-Head)]---> Delta_q(tau)
        --[Module 5: Geodesic ResNet Egress]----> q(tau+1) in S^{2047}
    """

    def __init__(self, config: Optional[PenroseMPUConfig] = None) -> None:
        super().__init__()
        self.config = config or PenroseMPUConfig.preset_penrose_full_2048()
        assert self.config.n_layers == 12, "PenroseMPU 标准五段式流水线固定为 12 层 (下 6 层 Module 2 + 上 6 层 Module 4)"
        assert self.config.bkp_bus_layer == 6, "Module 3 腰部耦合门严格位于第 6 层与第 7 层之间 (bkp_bus_layer=6)"

        self.module1_ingress = MPUModule1JetIngress()

        if self.config.d_model != D_BKP:
            self.stiefel_entry = nn.Parameter(torch.empty(D_BKP, self.config.d_model))
            nn.init.orthogonal_(self.stiefel_entry)
            self.stiefel_exit = nn.Parameter(self.stiefel_entry.t().contiguous().clone())
        else:
            self.register_parameter("stiefel_entry", None)
            self.register_parameter("stiefel_exit", None)

        self.rope = DualSubspaceRoPE(
            sub_dim=self.config.head_dim // 2,
            max_seq_len=self.config.max_seq_len,
            theta=self.config.rope_theta,
        )

        self.module2_lower_stack = nn.ModuleList([
            BicameraDecoderLayer(self.config, idx) for idx in range(0, 6)
        ])

        self.module3_waist_coupler = MPUModule3WaistCoupler(self.config)

        self.module4_upper_stack = nn.ModuleList([
            BicameraDecoderLayer(self.config, idx) for idx in range(6, 12)
        ])
        self.final_norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.step_gain = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        self.module5_egress = MPUModule5GeodesicEgress()

    @torch.no_grad()
    def project_stiefel_frames(self) -> None:
        """
        在 Agile 配置 (d_model < 2048) 下，将 stiefel_entry/exit 与腰部 p_in/out 严格重投影回 Stiefel 正交流形 O(2048, d_model)；
        在原生 preset_penrose_full_2048 (d_model = 2048) 下四块矩阵全为 None，无任何升降维损耗。
        """
        if self.stiefel_entry is not None and self.stiefel_exit is not None:
            w_tall = self.stiefel_entry.detach().float()  # (2048, d_model)
            q_orth, r_mat = torch.linalg.qr(w_tall, mode="reduced")
            diag_sign = torch.sgn(torch.diagonal(r_mat)).unsqueeze(0)
            diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
            q_orth = (q_orth * diag_sign).to(self.stiefel_entry.dtype)
            self.stiefel_entry.copy_(q_orth.contiguous())
            self.stiefel_exit.copy_(q_orth.t().contiguous())
        self.module3_waist_coupler.project_stiefel_frames()

    def _enter_backbone(self, psi_0_2048: torch.Tensor) -> torch.Tensor:
        if self.stiefel_entry is None:
            return psi_0_2048
        scale = math.sqrt(float(self.config.d_model) / float(D_BKP))
        return torch.matmul(psi_0_2048, self.stiefel_entry) * scale

    def _exit_backbone(self, h_model: torch.Tensor) -> torch.Tensor:
        if self.stiefel_exit is None:
            return h_model
        scale = math.sqrt(float(D_BKP) / float(self.config.d_model))
        return torch.matmul(h_model, self.stiefel_exit) * scale

    @staticmethod
    def compute_geodesic_chord_loss(
        q_pred: torch.Tensor,
        q_target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        分支 C（并行训练态 / 零解码器直算 Loss）：
        直接在 Module 5 出口计算预测波包 q(tau+1) 与目标波包 q*(tau+1) 的球面弦长/余弦距离：
            L_Penrose(tau) = 1 - <q(tau+1), q*(tau+1)> = 0.5 * ||q(tau+1) - q*(tau+1)||_2^2
        完全无需任何 LM Head 大矩阵乘法或 Softmax！
        """
        qp = F.normalize(q_pred.float(), dim=-1)
        qt = F.normalize(q_target.float(), dim=-1)
        cos_sim = (qp * qt).sum(dim=-1)
        step_loss = 1.0 - cos_sim
        if loss_mask is not None:
            m = loss_mask.float().view_as(step_loss)
            return (step_loss * m).sum() / (m.sum() + 1e-6)
        return step_loss.mean()

    def forward(
        self,
        q_curr: torch.Tensor,
        q_prev: Optional[torch.Tensor] = None,
        pmc_manager: Optional[PMCManager] = None,
        q_target: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        pmc_dropout: bool = False,
        puncture_dropout_prob: float = 0.0,
        puncture_expert_mask: Optional[torch.Tensor] = None,
    ) -> MPUStepOutput:
        """
        MPU 五段式前向流水线（同时支持全序列 [B, T, 2048] 并行训练与单拍 [B, 1, 2048] 状态机流转）。
        """
        # 1. Module 1: 1-Jet 相空间合成 + 早一轮异步 PMC 预取触发
        psi_0 = self.module1_ingress(q_curr, q_prev=q_prev, pmc_manager=pmc_manager)

        # 2. Module 2: 前 6 层上下文割线积分栈 (Lower 6 Blocks)
        h = self._enter_backbone(psi_0)
        for layer in self.module2_lower_stack:
            h, _, _ = layer(
                h,
                rope=self.rope,
                bkp_bus=None,
                bkp_manager=None,
                bkp_dropout=True,
                delta_h_bkp_highway=None,
            )

        # 3. Module 3: 腰部 PMC 流形场耦合门 (Waist Manifold Coupler)
        h_coupled, z_highway, diag = self.module3_waist_coupler(
            h,
            psi_0_anchor=psi_0,
            pmc_manager=pmc_manager,
            pmc_dropout=pmc_dropout,
            puncture_dropout_prob=puncture_dropout_prob,
            puncture_expert_mask=puncture_expert_mask,
        )

        # 4. Module 4: 后 6 层轨迹曲率精炼栈与风格塑形器 (Upper 6 Blocks)
        # 零 .item() 同步阻塞：纯张量掩码判断 z_highway 是否为非零流形脉冲
        has_highway = z_highway.abs().amax(dim=-1, keepdim=True) > 1e-7
        z_highway = torch.where(has_highway, z_highway, h_coupled.detach() * 0.1)
        h_upper = h_coupled
        for layer in self.module4_upper_stack:
            h_upper, _, _ = layer(
                h_upper,
                rope=self.rope,
                bkp_bus=None,
                bkp_manager=None,
                bkp_dropout=False,
                delta_h_bkp_highway=z_highway,
            )
        h_final = self.final_norm(h_upper)
        delta_q_backbone = self._exit_backbone(h_final) * torch.clamp(self.step_gain, 0.05, 2.0)
        delta_q_total = delta_q_backbone + diag.delta_pmc.to(delta_q_backbone.dtype)

        # 5. Module 5: 跨核球面 ResNet 闭环器 (Geodesic Residual Egress)
        q_next, delta_q_perp = self.module5_egress(q_curr, delta_q_total)

        loss = None
        if q_target is not None:
            loss = self.compute_geodesic_chord_loss(q_next, q_target, loss_mask=loss_mask)

        return MPUStepOutput(
            q_next=q_next,
            psi_0=psi_0,
            delta_q_perp=delta_q_perp,
            helmholtz_diag=diag,
            loss=loss,
        )


