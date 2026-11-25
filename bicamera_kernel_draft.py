import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ==============================================================================
# 1. 模型全局配置契约 (3B / 7B / 14B 标准规格预设)
# ==============================================================================
@dataclass
class BicameraConfig:
    model_name: str = "Bicamera-Kernel-3B"
    vocab_size: int = 102400
    d_model: int = 2048
    n_layers: int = 24
    
    # 注意力配置 (50% Differential Heads + 50% Standard Softmax Heads, GQA 4:1)
    n_heads: int = 16          # 必须为偶数: 前 n_heads//2 为差分头，后 n_heads//2 为标准头
    n_kv_heads: int = 4        # 必须为偶数: 前 n_kv_heads//2 供差分头，后 n_kv_heads//2 供标准头
    head_dim: int = 128        # 差分头内部拆为 2 x 64 正负流
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    
    # 每层自涌现正交 Meta-MoE 配置 (纯正交路由元算子池，无常驻共享专家，受 .bkp 乐谱指令路由调制)
    moe_n_routed_experts: int = 16
    moe_top_k: int = 4
    moe_n_shared_experts: int = 0
    moe_inter_dim: int = 1024
    moe_bias_update_rate: float = 1e-3  # 免辅助损失动态负载均衡步长
    moe_full_active_orthogonal: bool = True  # v35: 100% 全激活正交基元算子模式 (零同步稠密融合 + 三重正交约束)
    
    # 腰部 .bkp 总线配置
    d_bkp: int = 2048                                      # 全系物理锁死 2048 维 (4KB/微专家)
    bkp_bus_layer: int = 7                                 # 0-indexed: 第 8 层 (腰部总线层)
    # 历史遗留字段: 自 v21 起后段全部层 (l >= bkp_bus_layer) 均挂载知识原料端口,
    # 该列表仅用于旧断点/旧配置的向后兼容读取, 不再决定哪些层挂载 Highway 门控。
    bkp_highway_layers: List[int] = field(default_factory=lambda: [11, 15, 19, 23])
    bkp_num_heads: int = 8                                 # .bkp 多头乘积键检索头数
    bkp_top_k_max: int = 512                               # 动态配额上限 [0, 512]
    gradient_checkpointing: bool = False                   # 跨层激活重计算开关 (3B/7B/14B 默认开启)
    
    rms_norm_eps: float = 1e-6

    def knowledge_operator_layers(self) -> List[int]:
        """
        返回挂载 .bkp 知识原料端口的全部层索引 (腰部总线层及其后所有层)。

        自 v21 起，后段每一层 (l >= bkp_bus_layer) 的 OrthogonalMetaMoE 均被改造为纯双线性元算子:
            MetaMoE_l(h, z_l) = O_l(h) · z_l,   O_l(h) 由本层上下文 h 生成, z_l 为 .bkp 知识向量
        当 z_l = 0 (拔除 .bkp) 时该层算子严格恒等于 0，后段不再保留任何可绕开 .bkp
        私藏静态百科知识的普通 FFN 后门层。
        """
        return [i for i in range(self.n_layers) if i >= self.bkp_bus_layer]


    @classmethod
    def preset_3b(cls) -> "BicameraConfig":
        return cls(
            model_name="Bicamera-Kernel-3B",
            d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
            moe_n_routed_experts=16, moe_top_k=4, moe_inter_dim=1024,
            bkp_bus_layer=7, bkp_highway_layers=[11, 15, 19, 23],
            gradient_checkpointing=True,
        )

    @classmethod
    def preset_7b(cls) -> "BicameraConfig":
        return cls(
            model_name="Bicamera-Kernel-7B",
            d_model=3072, n_layers=28, n_heads=24, n_kv_heads=6,
            moe_n_routed_experts=24, moe_top_k=4, moe_inter_dim=1024,
            bkp_bus_layer=8, bkp_highway_layers=[13, 18, 23, 27],
            gradient_checkpointing=True,
        )

    @classmethod
    def preset_14b(cls) -> "BicameraConfig":
        return cls(
            model_name="Bicamera-Kernel-14B",
            d_model=4096, n_layers=36, n_heads=32, n_kv_heads=8,
            moe_n_routed_experts=32, moe_top_k=4, moe_inter_dim=896,
            bkp_bus_layer=11, bkp_highway_layers=[17, 23, 29, 35],
            gradient_checkpointing=True,
        )
        
    @classmethod
    def preset_micro_4060(cls) -> "BicameraConfig":
        """单卡 8GB 显存 (RTX 4060) / 16GB 昇腾 310B4 主机训练预设 (~125M 参数, 死锁 d_bkp=2048)"""
        return cls(
            model_name="Bicamera-Kernel-Micro-125M",
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            head_dim=64,
            moe_n_routed_experts=8,
            moe_top_k=2,
            moe_inter_dim=512,
            bkp_bus_layer=3,                 # 第 4 层 (前 1/3 处)
            bkp_highway_layers=[5, 7, 9, 11], # 后段每 2 层一次高速路直注
            bkp_top_k_max=16,
            gradient_checkpointing=True,
        )

    @classmethod
    def preset_micro_125m(cls, vocab_size: int = 16384) -> "BicameraConfig":
        cfg = cls.preset_micro_4060()
        cfg.vocab_size = vocab_size
        return cfg

    @classmethod
    def preset_nano_edge(cls) -> "BicameraConfig":
        """快速闭环验证 / 边缘轻量训练预设 (~35M 参数, 死锁 d_bkp=2048)"""
        return cls(
            model_name="Bicamera-Kernel-Nano-35M",
            d_model=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            head_dim=64,
            moe_n_routed_experts=4,
            moe_top_k=2,
            moe_inter_dim=384,
            bkp_bus_layer=2,                 # 第 3 层 (前 1/3 处)
            bkp_highway_layers=[3, 5, 7],    # 后段高速路直注
            bkp_top_k_max=32,
        )


# ==============================================================================
# 2. 基础组件: FP32-Safe RMSNorm, 对称子通道 RoPE & SwiGLU 专家
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 强制在 FP32 下计算方差，防止 BF16/FP16 深层激活平方溢出
        x_fp32 = x.float()
        norm_x = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x.type_as(x) * self.weight


class DualSubspaceRoPE(nn.Module):
    """
    专为差分注意力定制的对称旋转位置编码:
    以 sub_dim = head_dim // 2 (64维) 为基本旋转周期，同时施加于 128 维的左右两个 64 维半区。
    这保证了 Differential Attention 拆分为 (2, 64) 正负双流时，两个子流拥有 100% 相同的频率谱。
    """
    def __init__(self, sub_dim: int = 64, max_seq_len: int = 8192, theta: float = 500000.0):
        super().__init__()
        self.sub_dim = sub_dim
        self.theta = float(theta)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, sub_dim, 2).float() / sub_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, sub_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _ensure_seq_len(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self.cos_cached.size(0) and self.cos_cached.device == device:
            return
        new_max = max(seq_len, self.cos_cached.size(0))
        if seq_len > self.cos_cached.size(0):
            new_max = max(seq_len, int(self.cos_cached.size(0) * 2))
        inv_f = self.inv_freq.to(device=device, dtype=torch.float32)
        t = torch.arange(new_max, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_f)
        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()

    def _apply_rotary_64(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (..., sub_dim=64)
        half = self.sub_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, head_dim=128) -> 拆为 (B, T, H, 2, 64) 对称旋转后还原
        B, T, H, D = x.shape
        self._ensure_seq_len(T, x.device)
        cos = self.cos_cached[:T].view(1, T, 1, 1, -1).type_as(x)
        sin = self.sin_cached[:T].view(1, T, 1, 1, -1).type_as(x)
        x_sub = x.view(B, T, H, 2, self.sub_dim)
        return self._apply_rotary_64(x_sub, cos, sin).view(B, T, H, D)


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, inter_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, inter_dim, bias=False)
        self.up_proj = nn.Linear(d_model, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ==============================================================================
# 3. 混合注意力层: 50% Differential Attention + 50% Standard Softmax Attention
# ==============================================================================
class HybridDiffSoftmaxAttention(nn.Module):
    """
    同层并列双物理注意力:
    - 前半部分头 (n_heads // 2): Differential Attention (正反双流相减，滤除先验共模噪声)
    - 后半部分头 (n_heads // 2): Standard Softmax Attention (宽域语义平滑聚合)
    - 支持 Phase 2 -> Phase 1 (<think>) 的时序单向梯度截断 (Temporal Stop-Gradient)
    """
    def __init__(self, config: BicameraConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.has_bkp_control = layer_idx >= config.bkp_bus_layer

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        if self.has_bkp_control:
            # Kernel 即工具范式 (.bkp-Programmed Attention Tool):
            # 在腰部总线及之后的所有层中，Attention 不再是脱离 .bkp 自行运转的静态层，
            # 而是由 .bkp 语境指令调制的上下文抓取工具:
            # - bkp_q_steer : 指导当前语境下 Attention 探针应向哪个方向对焦扫描前后文
            # - bkp_v_filter: 纯无偏置双线性决定从命中上下文中提取什么语义特征 (零 .bkp 指令时严格 0 输出)
            self.bkp_q_steer = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
            self.bkp_v_filter = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 差分注意力专属参数
        n_diff_heads = self.n_heads // 2
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.randn(n_diff_heads, self.head_dim // 2) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_diff_heads, self.head_dim // 2) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_diff_heads, self.head_dim // 2) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_diff_heads, self.head_dim // 2) * 0.1)
        self.diff_group_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _sdpa_pair(
        self,
        q_diff: torch.Tensor,
        k_diff: torch.Tensor,
        v_diff: torch.Tensor,
        q_soft: torch.Tensor,
        k_soft: torch.Tensor,
        v_soft: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将 32 维差分正/负子流预乘 2^(1/4) 并零填充至 64 维，使 d_q = d_k = d_v = 64，
        # 并将 out_pos, out_neg, out_soft 三路合并为单次连续张量的硬件 FlashAttention-2 调用
        half_h = self.n_heads // 2
        scale_comp = math.sqrt(math.sqrt(2.0))  # 补偿 32 -> 64 维 SDPA 默认缩放因子 1/sqrt(64) 与 1/sqrt(32) 的比值
        q_pos = F.pad(q_diff[:, :, :, 0, :] * scale_comp, (0, self.head_dim // 2))
        k_pos = F.pad(k_diff[:, :, :, 0, :] * scale_comp, (0, self.head_dim // 2))
        q_neg = F.pad(q_diff[:, :, :, 1, :] * scale_comp, (0, self.head_dim // 2))
        k_neg = F.pad(k_diff[:, :, :, 1, :] * scale_comp, (0, self.head_dim // 2))

        q_fused = torch.cat([q_pos, q_neg, q_soft], dim=1).contiguous()
        k_fused = torch.cat([k_pos, k_neg, k_soft], dim=1).contiguous()
        v_fused = torch.cat([v_diff, v_diff, v_soft], dim=1).contiguous()

        if attn_mask is not None:
            out_fused = F.scaled_dot_product_attention(q_fused, k_fused, v_fused, attn_mask=attn_mask, is_causal=False)
        else:
            out_fused = F.scaled_dot_product_attention(q_fused, k_fused, v_fused, is_causal=True)
        out_pos, out_neg, out_soft = torch.split(out_fused, [half_h, half_h, half_h], dim=1)

        lam_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).view(1, half_h, 1, 1))
        lam_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).view(1, half_h, 1, 1))
        lam = (lam_1 - lam_2 + self.lambda_init).type_as(out_pos)

        out_diff = self.diff_group_norm(out_pos - lam * out_neg) * (1.0 - self.lambda_init)
        return out_diff, out_soft

    def forward(
        self,
        x: torch.Tensor,
        rope: DualSubspaceRoPE,
        phase2_mask: Optional[torch.Tensor] = None,
        z_bkp: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        phase2_mask: (B, T) bool 张量，True 代表 Phase 2 (Output 渲染阶段)，False 代表 Prompt 与 Phase 1 (<think>)。
                     若提供，则 Phase 2 Token 在读取 Phase 1 KV Cache 时自动截断反向梯度 (Temporal Stop-Gradient)。
        z_bkp      : (B, T, D) 来自 .bkp 的语境操纵指令向量；在后段层 (has_bkp_control=True) 中，
                     指导 Attention 工具如何对焦前后文 (q_steer) 与提取什么上下文特征 (v_filter)。
        """
        B, T, _ = x.shape
        if self.has_bkp_control:
            if z_bkp is None:
                zero_out = torch.zeros_like(x)
                zero_anchor = torch.zeros((B, T, self.head_dim), device=x.device, dtype=x.dtype)
                return zero_out, zero_anchor
            z_rms = z_bkp * torch.rsqrt(z_bkp.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(z_bkp.dtype)
            q_raw = self.q_proj(x) * (1.0 + torch.tanh(self.bkp_q_steer(z_rms)))
            v_raw = self.v_proj(x) * self.bkp_v_filter(z_rms)
        else:
            q_raw = self.q_proj(x)
            v_raw = self.v_proj(x)

        q = rope(self.q_norm(q_raw.view(B, T, self.n_heads, self.head_dim)))
        k = rope(self.k_norm(self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)))
        v = v_raw.view(B, T, self.n_kv_heads, self.head_dim)

        k_exp = k.repeat_interleave(self.n_rep, dim=2)
        v_exp = v.repeat_interleave(self.n_rep, dim=2)

        half_h = self.n_heads // 2
        q_diff = q[:, :, :half_h, :].transpose(1, 2).view(B, half_h, T, 2, self.head_dim // 2)
        k_diff = k_exp[:, :, :half_h, :].transpose(1, 2).view(B, half_h, T, 2, self.head_dim // 2)
        v_diff = v_exp[:, :, :half_h, :].transpose(1, 2)

        q_soft = q[:, :, half_h:, :].transpose(1, 2)
        k_soft = k_exp[:, :, half_h:, :].transpose(1, 2)
        v_soft = v_exp[:, :, half_h:, :].transpose(1, 2)

        if phase2_mask is None or not self.training or not getattr(self, "enable_phase2_kv_detach", False):
            # 全图直通因果注意力 (Full-Graph O(1) Credit Assignment):
            # 默认关闭旧版 Phase 2 对 Phase 1 (<think>) KV 的 Stop-Gradient 双重 SDPA 截断，
            # 让最终结论 (Answer) 的误差梯度通过 50% 狄拉克差分注意力头以 O(1) 路径直达 <think> 关键推理步，
            # 彻底消除训练态向串行链式 RNN 的退化，并将 Attention 前向/反向吞吐提升近 1 倍。
            out_diff, out_soft = self._sdpa_pair(q_diff, k_diff, v_diff, q_soft, k_soft, v_soft, attn_mask=attn_mask)
        else:
            # 仅当显式开启 enable_phase2_kv_detach 时执行时序分相梯度阻断
            out_diff_p1, out_soft_p1 = self._sdpa_pair(
                q_diff, k_diff, v_diff, q_soft, k_soft, v_soft, attn_mask=attn_mask
            )
            
            p1_kv_mask_5d = (~phase2_mask).view(B, 1, T, 1, 1)
            p1_kv_mask_4d = (~phase2_mask).view(B, 1, T, 1)
            k_diff_sg = torch.where(p1_kv_mask_5d, k_diff.detach(), k_diff)
            v_diff_sg = torch.where(p1_kv_mask_4d, v_diff.detach(), v_diff)
            k_soft_sg = torch.where(p1_kv_mask_4d, k_soft.detach(), k_soft)
            v_soft_sg = torch.where(p1_kv_mask_4d, v_soft.detach(), v_soft)

            out_diff_p2, out_soft_p2 = self._sdpa_pair(
                q_diff, k_diff_sg, v_diff_sg, q_soft, k_soft_sg, v_soft_sg, attn_mask=attn_mask
            )
            p2_q_mask = phase2_mask.view(B, 1, T, 1)
            out_diff = torch.where(p2_q_mask, out_diff_p2, out_diff_p1)
            out_soft = torch.where(p2_q_mask, out_soft_p2, out_soft_p1)

        out = torch.cat([out_diff, out_soft], dim=1).transpose(1, 2).contiguous().view(B, T, -1)
        
        # 仅提取后半部分标准 KV 头 (n_kv_heads // 2:) 的均值供顶层 Pointer Copy Head 寻址，避免混入差分正负子流
        half_kv = self.n_kv_heads // 2
        k_copy_anchor = k[:, :, half_kv:, :].mean(dim=2)  # (B, T, head_dim)
        return self.o_proj(out), k_copy_anchor


# ==============================================================================
# 4. 自涌现正交元算子池: Orthogonal Meta-MoE (带免辅助损失偏置负载均衡)
# ==============================================================================
class OrthogonalMetaMoE(nn.Module):
    def __init__(self, config: BicameraConfig):
        super().__init__()
        self.n_routed = config.moe_n_routed_experts
        self.top_k = config.moe_top_k
        self.bias_update_rate = config.moe_bias_update_rate
        self.full_active_orthogonal = bool(getattr(config, "moe_full_active_orthogonal", True))
        self.router = nn.Linear(config.d_model, self.n_routed, bias=False)
        nn.init.orthogonal_(self.router.weight)

        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.register_buffer("last_expert_counts", torch.zeros(self.n_routed))

        # 遵循白皮书 2.1.1 & 2.1.3 节: 移除常驻 shared_expert，100% 由受正交约束的路由元算子组成。
        # v21 起每个专家被显式拆分为"逻辑控制门 (gate_proj, 只读上下文)"与"特征值载荷 (up_proj, 只读 .bkp 知识原料)"，
        # 构成双线性算子 O_l(h) · z_l，零知识原料时输出严格为 0，从数学结构上杜绝后段 MoE 私自背诵静态百科。
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.moe_inter_dim) for _ in range(self.n_routed)
        ])

    def _dense_orthogonal_basis_experts(
        self,
        x_ctx_flat: torch.Tensor,
        x_val_flat: torch.Tensor,
        mod_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        v35 100% 全激活正交基双线性元算子 (Full-Active Orthogonal Basis Operators):
        - 彻底废除 Top-K 硬截断、bincount/argsort 分桶与 counts_int.max().item() CPU-GPU 同步阻塞；
        - 全部 E 个正交基元算子在单次融合 2D GEMM 中 100% 并发激活，由零均值球面正交调制系数 mod_weights 加权合成；
        - 针对 14B 大模型 (E*I >= 16384): 自动按 Token 分块执行融合 GEMM，将单层瞬态激活显存硬封顶在 <350MB；
        - 严格保持 up_proj 无偏置双线性结构: 当 x_val_flat ≡ 0 (拔除 .bkp) 时 up_act ≡ 0，输出严格恒等于 0。
        """
        N = x_ctx_flat.size(0)
        E = self.n_routed
        I = self.routed_experts[0].gate_proj.weight.size(0)
        EI = E * I

        w_gate = torch.cat([e.gate_proj.weight for e in self.routed_experts], dim=0)  # (E * I, D)
        w_up = torch.cat([e.up_proj.weight for e in self.routed_experts], dim=0)      # (E * I, D)
        w_down = torch.cat([e.down_proj.weight for e in self.routed_experts], dim=1)  # (D, E * I)
        mod_w = mod_weights.unsqueeze(-1).to(x_ctx_flat.dtype)                        # (N, E, 1)

        # 14B 显存水位保护: 控制单次瞬态 (chunk_N, E * I) 激活张量不超过 3.3e7 元素 (~64MB BF16)
        max_hidden_elems = 33554432
        if N * EI <= max_hidden_elems:
            gate_act = F.linear(x_ctx_flat, w_gate)                                   # (N, E * I)
            up_act = F.linear(x_val_flat, w_up)                                       # (N, E * I)
            hidden = (F.silu(gate_act) * up_act).view(N, E, I) * mod_w
            return F.linear(hidden.view(N, EI), w_down)                               # (N, D)

        chunk_n = max(256, (max_hidden_elems // max(1, EI) // 256) * 256)
        out_chunks = []
        for s_i in range(0, N, chunk_n):
            e_i = min(N, s_i + chunk_n)
            c_len = e_i - s_i
            g_c = F.linear(x_ctx_flat[s_i:e_i], w_gate)
            u_c = F.linear(x_val_flat[s_i:e_i], w_up)
            h_c = (F.silu(g_c) * u_c).view(c_len, E, I) * mod_w[s_i:e_i]
            out_chunks.append(F.linear(h_c.view(c_len, EI), w_down))
        return torch.cat(out_chunks, dim=0)

    def _bmm_experts(
        self,
        x_ctx_flat: torch.Tensor,
        x_val_flat: torch.Tensor,
        flat_weights: torch.Tensor,
        sort_order: torch.Tensor,
        sorted_experts: torch.Tensor,
        sorted_token_idx: torch.Tensor,
        intra_idx: torch.Tensor,
        max_c: int,
    ) -> torch.Tensor:
        """
        双输入双线性元算子融合实现:
        - 逻辑控制门 gate_proj 只吃本层上下文 x_ctx -> 决定"调用哪一个认知原语、开多大开度"
        - 特征值通路 up_proj   只吃知识原料 x_val -> 决定"用什么内容去填充该算子"
        - hidden = SiLU(W_gate · x_ctx) ⊙ (W_up · x_val)
        因 up_proj 为无偏置线性映射，当 x_val ≡ 0 时 hidden 严格为 0、算子输出严格为 0，
        即"零原料零输出"公理: 后段算子被数学结构剥夺了独立背诵静态知识的能力。
        """
        D = x_ctx_flat.size(-1)
        sorted_ctx = x_ctx_flat[sorted_token_idx]
        sorted_val = x_val_flat[sorted_token_idx]
        sorted_w = flat_weights[sort_order]

        padded_ctx = x_ctx_flat.new_zeros((self.n_routed, max_c, D))
        padded_ctx[sorted_experts, intra_idx] = sorted_ctx
        padded_val = x_val_flat.new_zeros((self.n_routed, max_c, D))
        padded_val[sorted_experts, intra_idx] = sorted_val

        # 组装 Gate / Up / Down 权重
        w_gate = torch.stack([e.gate_proj.weight for e in self.routed_experts], dim=0)  # (E, I, D)
        w_up = torch.stack([e.up_proj.weight for e in self.routed_experts], dim=0)      # (E, I, D)
        w_down = torch.stack([e.down_proj.weight for e in self.routed_experts], dim=0)  # (E, D, I)

        # 两次满载 Tensor Core 的批量矩阵乘 (总 FLOP 与原融合 (E,D,2I) 单次 BMM 完全一致)
        gate_act = torch.bmm(padded_ctx, w_gate.transpose(1, 2))                        # (E, max_c, I)
        up_act = torch.bmm(padded_val, w_up.transpose(1, 2))                            # (E, max_c, I)
        hidden = F.silu(gate_act) * up_act                                              # (E, max_c, I)
        expert_out = torch.bmm(hidden, w_down.transpose(1, 2))                          # (E, max_c, D)

        seg_out = (expert_out[sorted_experts, intra_idx] * sorted_w).to(x_ctx_flat.dtype)
        routed_out = torch.zeros_like(x_ctx_flat)
        routed_out.index_add_(0, sorted_token_idx, seg_out)
        return routed_out

    def forward(self, x: torch.Tensor, x_payload: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x        : (B, T, D) 本层上下文表征 (RMSNorm 之后的语境，恒为单位 RMS 尺度)
        x_payload: (B, T, D) 可选外部知识原料 z_l (来自 .bkp 的检索结果)
                   - 非 None: 进入纯双线性元算子模式 O_l(x) · x_payload
                   - None   : 退化为标准单输入 Meta-MoE (仅允许用于腰部之前的前段编码层)
        """
        B, T, D = x.shape
        x_ctx_flat = x.view(-1, D)
        x_val_flat = x_payload.view(-1, D).to(x_ctx_flat.dtype) if x_payload is not None else x_ctx_flat

        # 路由专家调制 (带知识原料时让路由器同时感知语境方向与知识方向)
        router_in = x_ctx_flat if x_payload is None else 0.5 * (x_ctx_flat + x_val_flat)
        if self.full_active_orthogonal or self.top_k >= self.n_routed:
            raw_logits = self.router(router_in).float() + self.expert_bias.float()
            centered = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
            inv_e = 1.0 / float(self.n_routed)
            inv_sqrt_e = 1.0 / math.sqrt(float(self.n_routed))
            # 零均值球面正交调制: 基准均分项 1/E + 正负对称正交分量 (1/sqrt(E))*tanh(centered)
            mod_weights = (inv_e + inv_sqrt_e * torch.tanh(centered)).to(x_ctx_flat.dtype)
            if self.training:
                # 统计各正交基算子的批内正向激励份额供负载均衡微调
                self.last_expert_counts.copy_(torch.sigmoid(centered.detach()).sum(dim=0))

            need_grad = torch.is_grad_enabled() and (
                x_ctx_flat.requires_grad
                or x_val_flat.requires_grad
                or mod_weights.requires_grad
                or self.routed_experts[0].gate_proj.weight.requires_grad
                or self.routed_experts[0].up_proj.weight.requires_grad
            )
            if need_grad:
                routed_out = checkpoint(
                    self._dense_orthogonal_basis_experts,
                    x_ctx_flat,
                    x_val_flat,
                    mod_weights,
                    use_reentrant=False,
                )
            else:
                routed_out = self._dense_orthogonal_basis_experts(x_ctx_flat, x_val_flat, mod_weights)
            return routed_out.view(B, T, D)

        scores = torch.sigmoid(self.router(router_in))
        _, topk_idx = torch.topk(scores + self.expert_bias, k=self.top_k, dim=-1)
        topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        flat_experts = topk_idx.view(-1)
        flat_weights = topk_weights.view(-1, 1).type_as(x_ctx_flat)

        # 单次 bincount + argsort 将派发给同一专家的 Token 聚拢，并打包为 (E, max_c, D) 执行 2 次融合 BMM，
        # 将原本 8x3=24 次串行小矩阵乘合并为 2 次满载 Tensor Core 的批量矩阵乘
        counts_int = torch.bincount(flat_experts, minlength=self.n_routed)
        if self.training:
            self.last_expert_counts.copy_(counts_int.detach().to(torch.float32))

        M = flat_experts.numel()
        if M == 0:
            return torch.zeros_like(x)

        sort_order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[sort_order]
        sorted_token_idx = sort_order // self.top_k

        # 计算每个派发项在所属专家桶内的局部偏移 intra_idx: [0, c_e)
        cum_counts = torch.cumsum(counts_int, dim=0)
        expert_starts = cum_counts - counts_int
        intra_idx = torch.arange(M, device=x_ctx_flat.device, dtype=sorted_experts.dtype) - expert_starts[sorted_experts]

        # 桶容量向上对齐至 128 的整数倍: 消除 CUDACachingAllocator 动态尺寸碎片池并对齐 Tensor Core Tile
        raw_max_c = int(counts_int.max().item())
        max_c = ((raw_max_c + 127) // 128) * 128 if raw_max_c > 0 else 0
        if max_c == 0:
            return torch.zeros_like(x)

        # 前向即时释放 3D Zero-Padded 瞬态大张量 (padded_ctx / padded_val / hidden / expert_out)，避免跨层叠加留存
        # 注意: Stage 2/3 冻结主干时 self.training=False 但 x_val 携带 .bkp 梯度 (need_grad=True)，因此仅以 need_grad 判定
        need_grad = torch.is_grad_enabled() and (
            x_ctx_flat.requires_grad
            or x_val_flat.requires_grad
            or flat_weights.requires_grad
            or self.routed_experts[0].gate_proj.weight.requires_grad
            or self.routed_experts[0].up_proj.weight.requires_grad
        )
        if need_grad:
            routed_out = checkpoint(
                self._bmm_experts,
                x_ctx_flat,
                x_val_flat,
                flat_weights,
                sort_order,
                sorted_experts,
                sorted_token_idx,
                intra_idx,
                max_c,
                use_reentrant=False,
            )
        else:
            routed_out = self._bmm_experts(
                x_ctx_flat, x_val_flat, flat_weights, sort_order, sorted_experts, sorted_token_idx, intra_idx, max_c
            )

        return routed_out.view(B, T, D)

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """每个训练 Step 结束后调用: 免辅助损失动态更新路由偏置 (DeepSeek-V3 策略)"""
        avg_count = self.last_expert_counts.mean()
        delta = torch.sign(avg_count - self.last_expert_counts) * self.bias_update_rate
        self.expert_bias.add_(delta)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        v35 三重正交基元算子约束 (Triple Stiefel Subspace Orthogonality, 全向量化单次归约):
        1. 输出写入子空间正交 (down_proj 列空间二阶能量包络去均值正交)；
        2. 知识读取子空间正交 (up_proj 输入通道二阶能量包络去均值正交，确保各算子读取 .bkp 的不同特征切面)；
        3. 路由触发探针正交 (router.weight 行向量正交，确保 E 个元算子沿两两垂直的语境/知识方向独立调制)。
        """
        w_down_all = torch.stack([e.down_proj.weight for e in self.routed_experts], dim=0).float()  # (E, D, I)
        w_up_all = torch.stack([e.up_proj.weight for e in self.routed_experts], dim=0).float()      # (E, I, D)

        e_down = w_down_all.pow(2).mean(dim=-1)                                                     # (E, D)
        down_mat = F.normalize(e_down - e_down.mean(dim=-1, keepdim=True), dim=-1)                  # (E, D)

        e_up = w_up_all.pow(2).mean(dim=1)                                                          # (E, D)
        up_mat = F.normalize(e_up - e_up.mean(dim=-1, keepdim=True), dim=-1)                        # (E, D)

        eye = torch.eye(self.n_routed, device=self.router.weight.device, dtype=torch.float32)
        gram_down = torch.matmul(down_mat, down_mat.t())
        gram_up = torch.matmul(up_mat, up_mat.t())

        r_norm = F.normalize(self.router.weight.float(), dim=-1)
        gram_router = torch.matmul(r_norm, r_norm.t())

        return (
            torch.norm(gram_down - eye, p="fro") ** 2
            + 0.5 * torch.norm(gram_up - eye, p="fro") ** 2
            + 0.5 * torch.norm(gram_router - eye, p="fro") ** 2
        )


# ==============================================================================
# 5. 腰部 .bkp 挂载总线 (含半正交保距投影 P_out = P_in^T)
# ==============================================================================
class BKPWaistBus(nn.Module):
    def __init__(self, config: BicameraConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_bkp = config.d_bkp
        self.bus_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # 腰部总线查询直接由前段双物理因果自注意力 (50% 差分头 + 50% 软注意力头) 汇聚的全域上下文 h_t 驱动:
        # 默认 bkp_window_k=1 (彻底废除冗余的固定局部滑动平均窗口，消除引用窗口外远距前文时的近视拖尾干扰)
        self.window_k = int(getattr(config, "bkp_window_k", 1))
        self.window_beta = float(getattr(config, "bkp_window_beta", 0.0))
        self.window_decay = float(getattr(config, "bkp_window_decay", 0.75))

        if self.window_k > 1:
            raw_hist = [self.window_decay ** (i - 1) for i in range(1, self.window_k)]
            hist_sum = sum(raw_hist)
            self.window_weights = (1.0 - self.window_beta,) + tuple(
                (w / hist_sum) * self.window_beta for w in raw_hist
            )
        else:
            self.window_weights = (1.0,)

        if self.d_model != self.d_bkp:
            self.p_in = nn.Parameter(torch.empty(self.d_model, self.d_bkp))
            nn.init.orthogonal_(self.p_in)
            # 输出投影矩阵 p_out (d_bkp, d_model) 与输入查询端口 p_in 解耦，初始严格对齐 p_in^T；
            # 当 Stage 2/3 冻结前段与 p_in 以锁死跨包通用查询坐标系时，p_out 仍可协同旋转以无损读取 2048 维微专家新特征
            self.p_out = nn.Parameter(self.p_in.t().contiguous().clone())
        else:
            self.register_parameter("p_in", None)
            self.register_parameter("p_out", None)

        # 跨核流形平移-仿射折叠偏置 (Zero-Overhead Affine Folding Buffer):
        # 默认全 0 (persistent=False，保持与已有检查点 state_dict 100% 兼容)；
        # 当调用 calibrate_and_fold_from_reference() 挂载异源内核的 .bkp 森林时，
        # 线性变换矩阵 W_BA 直接左乘折叠进 p_in / p_out，平移偏置写入 q_affine_bias，推理期 0 额外矩阵乘开销。
        self.register_buffer("q_affine_bias", torch.zeros(self.d_bkp), persistent=False)
        self.has_affine_calibration: bool = False
        self.calibrated_from_run: Optional[str] = None
        self.calibration_metrics: dict = {}

    def reset_affine_calibration(self) -> None:
        self.q_affine_bias.zero_()
        self.has_affine_calibration = False
        self.calibrated_from_run = None
        self.calibration_metrics = {}

    def _apply_causal_sliding_window(self, h: torch.Tensor) -> torch.Tensor:
        if self.window_k <= 1 or h.dim() != 3 or h.size(1) <= 1:
            return self.bus_norm(h)
        eps = self.bus_norm.eps
        h_fp32 = h.float()
        # 逐 Token 单位 RMS 预归一，防止窗口内单一高模长异常词元主导方向
        h_unit = h_fp32 * torch.rsqrt(h_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        # 严格因果左侧零填充，聚合最新 Token t 及前 k-1 个历史 Token
        h_pad = F.pad(h_unit, (0, 0, self.window_k - 1, 0))
        seq_len = h_unit.size(1)
        h_mix = h_unit * self.window_weights[0]
        for lag in range(1, self.window_k):
            start = self.window_k - 1 - lag
            h_mix = h_mix + h_pad[:, start : start + seq_len, :] * self.window_weights[lag]
        return self.bus_norm(h_mix.to(h.dtype))

    def project_to_bkp_space(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self._apply_causal_sliding_window(h)
        if self.p_in is None:
            q_out = h_norm
        else:
            scale = math.sqrt(self.d_bkp / self.d_model) if self.d_bkp > self.d_model else 1.0
            q_out = torch.matmul(h_norm, self.p_in) * scale
        if self.has_affine_calibration:
            q_out = q_out + self.q_affine_bias.to(q_out.dtype)
        return q_out

    def project_to_model_space(self, delta_bkp: torch.Tensor) -> torch.Tensor:
        if self.p_out is not None:
            return torch.matmul(delta_bkp, self.p_out)
        return delta_bkp if self.p_in is None else torch.matmul(delta_bkp, self.p_in.t())

    def compute_isometric_loss(self) -> torch.Tensor:
        """
        正交保距正则项:
        - 对处于可导状态的 p_in 与 p_out 分别施加半正交约束，保证查询下探与特征回写均为保距映射
        """
        if self.p_in is None:
            return torch.tensor(0.0, device=self.bus_norm.weight.device)
        loss = torch.tensor(0.0, device=self.bus_norm.weight.device)
        if self.p_in.requires_grad:
            p_in_fp32 = self.p_in.float()
            gram_in = (
                torch.matmul(p_in_fp32.t(), p_in_fp32)
                if self.d_model >= self.d_bkp
                else torch.matmul(p_in_fp32, p_in_fp32.t())
            )
            eye_in = torch.eye(gram_in.size(0), device=gram_in.device, dtype=torch.float32)
            loss = loss + torch.norm(gram_in - eye_in, p="fro") ** 2
        if self.p_out is not None and self.p_out.requires_grad:
            p_out_fp32 = self.p_out.float()
            gram_out = (
                torch.matmul(p_out_fp32, p_out_fp32.t())
                if self.d_model >= self.d_bkp
                else torch.matmul(p_out_fp32.t(), p_out_fp32)
            )
            eye_out = torch.eye(gram_out.size(0), device=gram_out.device, dtype=torch.float32)
            loss = loss + torch.norm(gram_out - eye_out, p="fro") ** 2
        return loss

    def forward(
        self,
        h_in: torch.Tensor,
        bkp_manager=None,
        bkp_dropout: bool = False,
    ) -> torch.Tensor:
        if bkp_dropout or bkp_manager is None:
            return torch.zeros_like(h_in)
        q_bkp = self.project_to_bkp_space(h_in)
        delta_bkp_2048 = bkp_manager.query_and_inject(q_bkp)
        return self.project_to_model_space(delta_bkp_2048)


# ==============================================================================
# 6. Transformer 标准层 & 主干 BicameraKernel (含因果 Pointer Copy Head)
# ==============================================================================
class BicameraDecoderLayer(nn.Module):
    def __init__(self, config: BicameraConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_checkpoint = bool(getattr(config, "gradient_checkpointing", False))
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = HybridDiffSoftmaxAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = OrthogonalMetaMoE(config)

        self.has_knowledge_port = layer_idx >= config.bkp_bus_layer
        self.is_bus_entry_layer = layer_idx == config.bkp_bus_layer
        self.has_highway_tap = layer_idx > config.bkp_bus_layer
        if self.is_bus_entry_layer:
            self.bkp_context_carrier = nn.Linear(config.d_model, config.d_model, bias=False)
        if self.has_highway_tap:
            self.bkp_highway_gate = nn.Parameter(torch.zeros(config.d_model))

    def _attn_subblock(
        self,
        h: torch.Tensor,
        rope: DualSubspaceRoPE,
        phase2_mask: Optional[torch.Tensor],
        z_bkp: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn(self.attn_norm(h), rope=rope, phase2_mask=phase2_mask, z_bkp=z_bkp, attn_mask=attn_mask)

    def _moe_subblock(self, h: torch.Tensor, z_payload: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.has_knowledge_port:
            return self.moe(self.ffn_norm(h), z_payload)
        return self.moe(self.ffn_norm(h))

    def forward(
        self,
        h: torch.Tensor,
        rope: DualSubspaceRoPE,
        bkp_bus: Optional[BKPWaistBus] = None,
        bkp_manager=None,
        bkp_dropout: bool = False,
        delta_h_bkp_highway: Optional[torch.Tensor] = None,
        phase2_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        
        # 1. 腰部总线层: 发起 .bkp 联想检索，并用 .bkp 语境指令对跨腰部语境状态流执行以单位增益 1.0 为中心的残差乘法调制
        if bkp_bus is not None:
            delta_h_bkp_highway = bkp_bus(
                h, bkp_manager=bkp_manager, bkp_dropout=bkp_dropout
            )
            if delta_h_bkp_highway is not None:
                z_bus_rms = delta_h_bkp_highway * torch.rsqrt(
                    delta_h_bkp_highway.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
                ).to(delta_h_bkp_highway.dtype)
                h = h * (1.0 + torch.tanh(self.bkp_context_carrier(z_bus_rms)))

        if self.has_knowledge_port:
            if delta_h_bkp_highway is None:
                zero_h = torch.zeros_like(h)
                zero_anchor = torch.zeros(
                    (h.size(0), h.size(1), self.attn.head_dim), device=h.device, dtype=h.dtype
                )
                return zero_h, None, zero_anchor
            if self.has_highway_tap:
                z_l = (0.5 + 1.5 * torch.sigmoid(self.bkp_highway_gate)) * delta_h_bkp_highway
            else:
                z_l = delta_h_bkp_highway
        else:
            z_l = None

        # 2. 本层混合双物理注意力工具 (后段层由 z_l 指导对焦与上下文提取)
        attn_need_grad = torch.is_grad_enabled() and (h.requires_grad or self.attn.q_proj.weight.requires_grad)
        ckpt_active = self.use_checkpoint and attn_need_grad
        if ckpt_active:
            attn_out, k_copy_anchor = checkpoint(
                self._attn_subblock, h, rope, phase2_mask, z_l, attn_mask, use_reentrant=False
            )
        else:
            attn_out, k_copy_anchor = self._attn_subblock(h, rope, phase2_mask, z_l, attn_mask)
        h = h + attn_out

        if self.has_knowledge_port:
            # 3. 后段纯双线性元算子与语境向量滑动: h_next = h + z_l + O_l(RMSNorm(h)) · z_l
            moe_out = self.moe(self.ffn_norm(h), z_l)
            h = h + z_l + moe_out
            return h, delta_h_bkp_highway, k_copy_anchor

        # 4. 腰部之前的前段编码层 (提问器): 提取输入语境特征以构造 .bkp 查询
        h = h + self.moe(self.ffn_norm(h))
        return h, delta_h_bkp_highway, k_copy_anchor


class BicameraKernel(nn.Module):
    def __init__(self, config: BicameraConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = DualSubspaceRoPE(
            sub_dim=config.head_dim // 2, max_seq_len=config.max_seq_len, theta=config.rope_theta
        )
        self.bkp_bus = BKPWaistBus(config)

        self.layers = nn.ModuleList([
            BicameraDecoderLayer(config, idx) for idx in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # 纯净词表输出头 (精确符号寻址与语境融合完全由各层内部的 50% 狄拉克差分注意力头 + 50% 贝叶斯软注意力头原生承担)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        bkp_manager=None,
        bkp_dropout: bool = False,
        phase2_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取 final_norm 后的隐状态 h: (B, T, D) 与占位门控 zero_gate: (B, T, 1)，不实体化全尺寸 (B, T, V) Logits"""
        B, T = input_ids.shape
        prefix_frozen = not self.tok_embeddings.weight.requires_grad
        with torch.set_grad_enabled(not prefix_frozen and torch.is_grad_enabled()):
            h = self.tok_embeddings(input_ids)
        delta_h_bkp_highway = None
        last_k_copy_anchor = None

        for idx, layer in enumerate(self.layers):
            bus_module = self.bkp_bus if idx == self.config.bkp_bus_layer else None
            # 若前段层 (idx < bkp_bus_layer) 完全冻结，跳过 Autograd 图构建与激活留存以加速 Stage 3/4
            layer_frozen = (idx < self.config.bkp_bus_layer) and prefix_frozen
            with torch.set_grad_enabled(not layer_frozen and torch.is_grad_enabled()):
                h, delta_h_bkp_highway, last_k_copy_anchor = layer(
                    h,
                    rope=self.rope,
                    bkp_bus=bus_module,
                    bkp_manager=bkp_manager,
                    bkp_dropout=bkp_dropout,
                    delta_h_bkp_highway=delta_h_bkp_highway,
                    phase2_mask=phase2_mask,
                    attn_mask=None,
                )

        h = self.final_norm(h)
        zero_gate = torch.zeros((B, T, 1), device=h.device, dtype=h.dtype)
        return h, zero_gate

    def forward(
        self,
        input_ids: torch.Tensor,
        bkp_manager=None,
        bkp_dropout: bool = False,
        phase2_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, zero_gate = self.forward_features(
            input_ids,
            bkp_manager=bkp_manager,
            bkp_dropout=bkp_dropout,
            phase2_mask=phase2_mask,
        )
        if target_ids is not None:
            nll_loss = self.compute_chunked_lm_head_nll(h, target_ids, loss_mask=loss_mask)
            return nll_loss, zero_gate, zero_gate
        vocab_logits = self.lm_head(h)
        return vocab_logits, zero_gate, zero_gate

    @torch.no_grad()
    def extract_waist_normalized_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        仅运行前段编码层 layers[0 .. bkp_bus_layer - 1] 并经过 bkp_bus.bus_norm，
        提取腰部总线入口处的归一化隐状态 H_norm: (B, T, d_model)。
        """
        h = self.tok_embeddings(input_ids)
        for idx in range(self.config.bkp_bus_layer):
            h, _, _ = self.layers[idx](
                h,
                rope=self.rope,
                bkp_bus=None,
                bkp_manager=None,
                bkp_dropout=True,
                delta_h_bkp_highway=None,
                phase2_mask=None,
                attn_mask=None,
            )
        return self.bkp_bus._apply_causal_sliding_window(h)

    @torch.no_grad()
    def calibrate_and_fold_waist_bus(
        self,
        reference_kernel: "BicameraKernel",
        anchor_batches: List[torch.Tensor],
        ridge_lambda: float = 0.05,
        affine_gamma: float = 1.0,
        fold_p_out: bool = False,
        ref_name: str = "reference_kernel",
    ) -> dict:
        """
        基于锚点句的跨核流形平移-仿射混合校准与 BKPWaistBus 原地权重折叠 (Zero-Overhead Affine Folding):
        1. 在少量锚点序列 anchor_batches 上分别提取当前核 (B) 与参考核 (A) 的腰部归一化隐状态 H_B, H_A in R^(M x d_model)；
        2. 在满秩 d_model 空间闭式求解中心平移 (mu_B, mu_A)、Procrustes 正交旋转 R_BA 与 Tikhonov 仿射变换 W_BA_ridge；
        3. 按混合系数 affine_gamma in [0, 1] 合成变换矩阵:
              W_BA = (1 - affine_gamma) * R_BA + affine_gamma * W_BA_ridge
              c_BA = mu_A - mu_B @ W_BA
        4. 原地左乘折叠进当前核的 bkp_bus.p_in 与 q_affine_bias:
              p_in_folded  <- W_BA @ reference_kernel.bkp_bus.p_in
              q_affine_bias <- scale * (c_BA @ reference_kernel.bkp_bus.p_in)
           若 fold_p_out=True，同时将逆映射 W_AB 右乘折叠进 p_out:
              p_out_folded <- reference_kernel.bkp_bus.p_out @ W_AB
        5. 折叠完成后，当前核无需任何外置适配器或运行时额外矩阵乘，即可零开销直接挂载参考核的 .bkp 知识森林。
        """
        device = next(self.parameters()).device
        hb_list, ha_list = [], []
        for batch_ids in anchor_batches:
            inp = batch_ids.to(device)
            hb = self.extract_waist_normalized_states(inp).float().reshape(-1, self.config.d_model)
            ha = reference_kernel.extract_waist_normalized_states(inp).float().reshape(-1, self.config.d_model)
            hb_list.append(hb)
            ha_list.append(ha)

        HB = torch.cat(hb_list, dim=0)  # (M, d_model)
        HA = torch.cat(ha_list, dim=0)  # (M, d_model)
        M, D = HB.shape

        mu_B = HB.mean(dim=0, keepdim=True)  # (1, D)
        mu_A = HA.mean(dim=0, keepdim=True)  # (1, D)
        HB_c = HB - mu_B
        HA_c = HA - mu_A

        # 1) Procrustes 正交基解 R_BA = U @ V^T
        cov_BA = (HB_c.t() @ HA_c) / float(max(1, M))
        U, _, Vh = torch.linalg.svd(cov_BA, full_matrices=False)
        R_BA = U @ Vh
        R_AB = R_BA.t()

        # 2) 正交锚定 Tikhonov 仿射解 (Shrinkage toward Orthogonal Procrustes R_BA, 防止尾部小奇异值向 0 坍缩):
        #    min_W  (1/M)||HB_c @ W - HA_c||_F^2 + lambda * ||W - R_BA||_F^2
        #    => W_BA_ridge = (HB_c^T HB_c / M + lambda * I)^(-1) (HB_c^T HA_c / M + lambda * R_BA)
        eye_d = torch.eye(D, device=device, dtype=torch.float32)
        lam = float(ridge_lambda)
        gram_B = (HB_c.t() @ HB_c) / float(max(1, M)) + lam * eye_d
        W_BA_ridge = torch.linalg.solve(gram_B, cov_BA + lam * R_BA)

        gram_A = (HA_c.t() @ HA_c) / float(max(1, M)) + lam * eye_d
        W_AB_ridge = torch.linalg.solve(gram_A, cov_BA.t() + lam * R_AB)

        # 3) 正交-仿射混合变换矩阵 W_BA 与平移偏置 c_BA
        gamma = float(max(0.0, min(1.0, affine_gamma)))
        W_BA = (1.0 - gamma) * R_BA + gamma * W_BA_ridge
        W_AB = (1.0 - gamma) * R_AB + gamma * W_AB_ridge
        c_BA = (mu_A - mu_B @ W_BA).squeeze(0)  # (D,)

        # 计算流形对齐拟合优度 R^2 与余弦相似度
        var_A = HA_c.pow(2).sum().clamp_min(1e-8)
        raw_cos = F.cosine_similarity(HB, HA, dim=-1).mean().item()
        HA_orth = HB_c @ R_BA + mu_A
        r2_orth = float((1.0 - (HA - HA_orth).pow(2).sum() / var_A).item())
        HA_hat = HB @ W_BA + c_BA.unsqueeze(0)
        r2_folded = float((1.0 - (HA - HA_hat).pow(2).sum() / var_A).item())
        cal_cos = F.cosine_similarity(HA_hat, HA, dim=-1).mean().item()

        # 4) 原地矩阵折叠进 self.bkp_bus (兼容 d_model == d_bkp == 2048 时 p_in / p_out 初始为 None 的等维总线)
        param_dtype = self.bkp_bus.bus_norm.weight.dtype
        ref_p_in = (
            reference_kernel.bkp_bus.p_in.detach().float()
            if reference_kernel.bkp_bus.p_in is not None
            else torch.eye(self.config.d_bkp, device=device, dtype=torch.float32)
        )
        ref_p_out = (
            reference_kernel.bkp_bus.p_out.detach().float()
            if reference_kernel.bkp_bus.p_out is not None
            else torch.eye(self.config.d_bkp, device=device, dtype=torch.float32)
        )
        scale = math.sqrt(self.config.d_bkp / self.config.d_model) if self.config.d_bkp > self.config.d_model else 1.0

        p_in_folded = W_BA @ ref_p_in  # (d_model, d_bkp)
        b_in_folded = (c_BA @ ref_p_in) * scale  # (d_bkp,)
        if reference_kernel.bkp_bus.has_affine_calibration:
            b_in_folded = b_in_folded + reference_kernel.bkp_bus.q_affine_bias.detach().float()

        if self.bkp_bus.p_in is None:
            self.bkp_bus.p_in = nn.Parameter(p_in_folded.to(dtype=param_dtype), requires_grad=False)
        else:
            self.bkp_bus.p_in.data.copy_(p_in_folded.to(self.bkp_bus.p_in.dtype))
        self.bkp_bus.q_affine_bias.copy_(b_in_folded.to(self.bkp_bus.q_affine_bias.dtype))
        self.bkp_bus.has_affine_calibration = True
        self.bkp_bus.calibrated_from_run = ref_name

        if fold_p_out:
            p_out_folded = ref_p_out @ W_AB  # (d_bkp, d_model)
            if self.bkp_bus.p_out is None:
                self.bkp_bus.p_out = nn.Parameter(p_out_folded.to(dtype=param_dtype), requires_grad=False)
            else:
                self.bkp_bus.p_out.data.copy_(p_out_folded.to(self.bkp_bus.p_out.dtype))

        metrics = {
            "reference_kernel": ref_name,
            "anchor_tokens": int(M),
            "affine_gamma": round(gamma, 3),
            "ridge_lambda": float(ridge_lambda),
            "fold_p_out": bool(fold_p_out),
            "raw_cosine_sim": round(raw_cos, 4),
            "calibrated_cosine_sim": round(cal_cos, 4),
            "r2_orthogonal_pct": round(r2_orth * 100.0, 2),
            "r2_folded_affine_pct": round(r2_folded * 100.0, 2),
            "nonlinear_residual_pct": round((1.0 - r2_folded) * 100.0, 2),
        }
        self.bkp_bus.calibration_metrics = metrics
        return metrics

    @staticmethod
    def _chunk_lm_head_ce(
        h_chunk: torch.Tensor,
        lm_weight: torch.Tensor,
        target_chunk: torch.Tensor,
        mask_chunk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits_c = F.linear(h_chunk, lm_weight)
        dev_type = logits_c.device.type if logits_c.device.type in ("cuda", "npu", "cpu") else "cuda"
        with torch.autocast(device_type=dev_type, enabled=False):
            v_in = logits_c.float() if logits_c.dtype == torch.float16 else logits_c
            if mask_chunk is None:
                return F.cross_entropy(v_in, target_chunk, reduction="sum")
            nll_c = F.cross_entropy(v_in, target_chunk, reduction="none").float()
            return (nll_c * mask_chunk).sum()

    def compute_chunked_lm_head_nll(
        self,
        h: torch.Tensor,
        target_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        chunk_tokens: int = 2048,
    ) -> torch.Tensor:
        """
        流式分块 LM-Head + Cross-Entropy:
        按 chunk_tokens (默认 2048) 切分 (B*T, D) 隐状态，前向不实体化 (B, T, V) 全量 Logits，
        反向逐块即时重算 Logits 与 Softmax 梯度，在数学 100% 恒等的前提下消除数 GB 末端显存峰值。
        """
        h_flat = h.reshape(-1, h.size(-1))
        y_flat = target_ids.reshape(-1)
        m_flat = loss_mask.reshape(-1).float() if loss_mask is not None else None
        total_tokens = h_flat.size(0)
        lm_w = self.lm_head.weight
        use_ckpt = torch.is_grad_enabled() and (h_flat.requires_grad or lm_w.requires_grad)

        loss_sum = torch.zeros((), dtype=torch.float32, device=h.device)
        for s in range(0, total_tokens, chunk_tokens):
            e = min(total_tokens, s + chunk_tokens)
            h_c = h_flat[s:e]
            y_c = y_flat[s:e]
            m_c = m_flat[s:e] if m_flat is not None else None
            if use_ckpt:
                c_sum = checkpoint(self._chunk_lm_head_ce, h_c, lm_w, y_c, m_c, use_reentrant=False)
            else:
                c_sum = self._chunk_lm_head_ce(h_c, lm_w, y_c, m_c)
            loss_sum = loss_sum + c_sum.float()

        denom = (m_flat.sum() + 1e-6) if m_flat is not None else float(max(1, total_tokens))
        return loss_sum / denom

    def compute_composite_nll_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        vocab_logits: torch.Tensor,
        copy_attn_logits: Optional[torch.Tensor] = None,
        copy_gate: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        纯净词表负对数似然损失 (Standard Cross-Entropy NLL Loss):
        移除末端冗余的 Pointer Copy 旁路，使精确符号寻址与长尾知识检索 100% 由各层内部的
        双物理注意力头 (50% Differential + 50% Softmax) 与腰部 .bkp 知识包承担。
        """
        if vocab_logits.dim() == 0:
            return vocab_logits
        dev_type = vocab_logits.device.type if vocab_logits.device.type in ("cuda", "npu", "cpu") else "cuda"
        with torch.autocast(device_type=dev_type, enabled=False):
            v_flat = vocab_logits.reshape(-1, vocab_logits.size(-1))
            y_flat = target_ids.reshape(-1)
            m_flat = loss_mask.reshape(-1).float() if loss_mask is not None else None
            total_tokens = v_flat.size(0)
            chunk_tokens = 2048
            loss_sum = torch.zeros((), dtype=torch.float32, device=vocab_logits.device)
            for s in range(0, total_tokens, chunk_tokens):
                e = min(total_tokens, s + chunk_tokens)
                v_c = v_flat[s:e].float() if vocab_logits.dtype == torch.float16 else v_flat[s:e]
                y_c = y_flat[s:e]
                if m_flat is None:
                    loss_sum = loss_sum + F.cross_entropy(v_c, y_c, reduction="sum").float()
                else:
                    nll_c = F.cross_entropy(v_c, y_c, reduction="none").float()
                    loss_sum = loss_sum + (nll_c * m_flat[s:e]).sum()
            denom = (m_flat.sum() + 1e-6) if m_flat is not None else float(max(1, total_tokens))
            return loss_sum / denom