import os
import math
import mmap
import struct
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, overload
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 全局物理常数与 VFS 挂载权限契约
# ==============================================================================
D_BKP: int = 2048                  # 腰部总线物理锁死维度
PAGE_SIZE: int = 4096              # 单个微专家物理体积: 2048B (u_key) + 2048B (v_val) = 4096B (1 OS Page)
BKP_NUM_HEADS: int = 8             # 多头 PEER 检索头数
BKP_HEAD_DIM: int = 256            # 单头维度 (2048 // 8 = 256)
BKP_SUBKEY_DIM: int = 128          # 笛卡尔半键维度 (256 // 2 = 128)
BKP_MAGIC: bytes = b"BKP2048\x00"  # .bkp 二进制文件魔数 (8 Bytes)
STANDARD_GRID_SIZES: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)

# 零超参数闭式物理常数 (Zero-Hyperparameter Closed-Form Physical Constants):
# 全部由欧拉常数 gamma_EM、球面三角剖分最小单形 (d_s = 3) 与 2-测地线中点几何唯一导出，零人工经验魔数
EULER_MASCHERONI: float = 0.5772156649015329          # 欧拉-马歇罗尼常数 gamma_EM
GUMBEL_STD_COEFF: float = math.pi / math.sqrt(6.0)    # 冈贝尔极值分布标准差系数 pi / sqrt(6) ≈ 1.28255
SIMPLEX_2_GEODESIC_PROJ: float = 1.0 / math.sqrt(2.0) # 2-测地线 (边) 中点投影衰减比 1/sqrt(2) ≈ 0.70711
SIMPLEX_3_BARYCENTER_PROJ: float = 1.0 / math.sqrt(3.0) # 3-单纯形 (面) 重心投影衰减比 1/sqrt(3) ≈ 0.57735
JET_NORMAL_ALPHA_STAR: float = math.sqrt(2.0 / 3.0)   # 1-Jet 单纯形内无符号翻转法向能量比 sqrt(2/3) ≈ 0.81650
JET_TANGENT_NORM_STAR: float = math.sqrt(1.0 / 3.0)   # 1-Jet 单纯形内无符号翻转切向能量比 sqrt(1/3) ≈ 0.57735


def closed_form_gumbel_threshold(n_grid: float) -> float:
    """
    零超参数冈贝尔极值上确界闭式解 (Gumbel Extreme-Value Supremum):
    给定正交子键网格规模 N_grid，独立标准高斯投影最大值的极值分布参数为:
        b_N = sqrt(2 * ln(N_grid) - ln(4 * pi * ln(N_grid)))
        mu_Gumbel(N_grid) = b_N + gamma_EM / b_N
        sigma_Gumbel(N_grid) = (pi / sqrt(6)) / b_N
    返回极值噪声上确界 tau_z^*(N_grid) = mu_Gumbel(N_grid) + sigma_Gumbel(N_grid)。
    """
    n_eff = max(4.0, float(n_grid))
    ln_n = math.log(n_eff)
    b_n = math.sqrt(max(1.0, 2.0 * ln_n - math.log(4.0 * math.pi * ln_n)))
    return b_n + (EULER_MASCHERONI + GUMBEL_STD_COEFF) / b_n


# 由七大第一性原理与物理维度 (D_BKP=2048, BKP_NUM_HEADS=8, BKP_HEAD_DIM=256) 唯一导出的派生闭式常数
HEAD_INV_GAIN: float = 1.0 / math.sqrt(float(BKP_NUM_HEADS))           # 8 头正交等距子空间幅值缩放比 1/sqrt(8) ≈ 0.35355
HEAD_UNIFORM_RATIO: float = 1.0 / float(BKP_NUM_HEADS)                 # 单头均匀子空间能量权重 1/8 = 0.125
SUBSPACE_STD_SCALE: float = 1.0 / math.sqrt(float(BKP_HEAD_DIM))       # 单头球面各向同性标准差 1/sqrt(256) = 0.0625
AMBIENT_NOISE_FLOOR: float = 1.0 / math.sqrt(float(D_BKP))             # 2048 维母体超球面随机正交内积标准差 1/sqrt(2048) ≈ 0.02210
GUMBEL_SINK_DIV_STAR: float = (                                        # 8 头检索下负散度势阱显著超越背景噪声的冈贝尔闭式门限 ≈ -0.0475
    -closed_form_gumbel_threshold(float(BKP_NUM_HEADS)) / math.sqrt(float(D_BKP))
)
FUNNEL_RESONANCE_APERTURE: float = 0.5 * (1.0 + SIMPLEX_2_GEODESIC_PROJ)  # 单字重叠 1/sqrt(2) 与完整词重合 1.0 的几何中点孔径 ≈ 0.85355



class BKPMountMode(Enum):
    """对应 Penrose Harness VFS 三级目录的 OS 页表级读写保护权限"""
    RO_SYS = "ro_sys"          # /bkp/sys/ (Base.bkp 等): PROT_READ 全局硬件只读
    RO_LIB = "ro_lib"          # /bkp/lib/ (共享领域包推理态): PROT_READ 多上下文零拷贝共享
    RW_SHARED = "rw_shared"    # /bkp/workspace/ 主栈帧或拖拽训练态: PROT_READ|PROT_WRITE, MAP_SHARED (PTE Dirty 延迟落盘)
    COW_PRIVATE = "cow_priv"   # /bkp/workspace/.stack_tmp/ (key_lemma 子栈帧): MAP_PRIVATE 写时复制，支持零 I/O 回滚


@dataclass
class BKPRoutingConfig:
    num_heads: int = BKP_NUM_HEADS
    sub_top_k: int = 16                # 每个检索头在行/列子键中初筛的候选数 k_sub (构成 k_sub x k_sub 候选子格)
    head_top_k: int = 4                # 单头过线微专家席位上限
    global_top_k_max: int = 512        # 单 Token 全局动态激活微专家上限 Omega_active in [0, 512]
    alpha_0: float = 1.0               # 冈贝尔极值基准孔径缩放因子 (alpha_0=1.0 严格对应零超参数闭式解 tau_z^*(N_grid))
    geodesic_closure_soft_gamma: float = 0.0  # 未知语境自适应测地圆软化闭包系数 (0.0=关闭, 默认由 1/sqrt(3) 闭式驱动)
    geodesic_closure_damping: float = 0.45    # 软化闭包插值权重的平滑阻尼系数
    vram_cache_slots: int = 131072     # GPU 显存只读 LRU 缓存槽位数 (131072 * 4KB = 512 MB)
    l1_tlb_max_packages: int = 32      # GPU L1 路由快表常驻包数量上限


def round_up_to_standard_grid(k: int) -> int:
    """将任意非对称子块边长向上补齐 (Round Up) 至最近的 2^m 标准 .bkp 协议档位"""
    for std_k in STANDARD_GRID_SIZES:
        if k <= std_k:
            return std_k
    return STANDARD_GRID_SIZES[-1]


# ==============================================================================
# 2. 单个 .bkp 知识包实体 (4KB OS 页对齐存储、笛卡尔码本与拓扑扩缩容/分裂/合并)
# ==============================================================================
class BKPPackage(nn.Module):
    """
    单个 .bkp 模块化知识包:
    - 显存/内存常驻元数据:
        * u_codebook, v_codebook: (8, K_p, 128) 笛卡尔行列子键页表 (特征维强制 L2 单位范数)
        * scales: (M_p, 2) FP16 解量化缩放因子 (s_m^u, s_m^v)
        * purity_r: (M_p,) 球面矢端单义相干纯度 R_{a,b} in [0, 1]
    - 4KB 物理页对齐微专家池:
        * expert_pages_int8: (M_p, 2, 2048) torch.int8 张量 (支持直接绑定 OS mmap 虚拟内存页)
    """
    def __init__(
        self,
        pkg_id: str,
        grid_k: int = 64,
        mount_mode: BKPMountMode = BKPMountMode.RW_SHARED,
        is_base_pkg: bool = False,
        mmap_file_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        assert grid_k in STANDARD_GRID_SIZES, f"grid_k={grid_k} 必须属于标准 2^m 档位 {STANDARD_GRID_SIZES}"
        self.pkg_id = pkg_id
        self.grid_k = grid_k
        self.num_experts = grid_k * grid_k
        self.mount_mode = mount_mode
        self.is_base_pkg = is_base_pkg
        self.mmap_file_path = mmap_file_path
        self._mmap_obj: Optional[mmap.mmap] = None
        self._file_obj = None

        # 1. 笛卡尔行列子键码本: U_p, V_p in R^(8 x K_p x 128)
        trainable_keys = mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE)
        u_init = F.normalize(torch.randn(BKP_NUM_HEADS, grid_k, BKP_SUBKEY_DIM, device=device), dim=-1)
        v_init = F.normalize(torch.randn(BKP_NUM_HEADS, grid_k, BKP_SUBKEY_DIM, device=device), dim=-1)
        self.u_codebook = nn.Parameter(u_init, requires_grad=trainable_keys)
        self.v_codebook = nn.Parameter(v_init, requires_grad=trainable_keys)

        # 2. 常驻元数据: 解量化缩放因子 (s_u, s_v)、累计激活频次矩阵 C_{a,b}、单义纯度 R_{a,b} 与质量指数 Q_p
        #    初始 v_val 严格置 0，保证新包挂载或扩容瞬间前向输出残差为 0.0000
        init_scales = torch.zeros(self.num_experts, 2, dtype=torch.float16, device=device)
        # rand_u in [-64, 64] 均匀分布方差为 1/3，scale=2/64 使 u_dequant 初始 RMS 约为 1.15，确保点积处于有效激活区
        init_scales[:, 0] = 2.0 / 64.0
        self.register_buffer("scales", init_scales)
        self.register_buffer("activation_counts", torch.zeros(self.num_experts, dtype=torch.int64, device=device))
        self.register_buffer("purity_r", torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer("quality_q", torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer("ema_route_clarity", torch.tensor(1.0, dtype=torch.float32, device=device))

        # 2b. 查询静态表征锥全局背景中心 (Query Static Cone EMA Center) —— 严格因果的跨 Token 去共模基线
        #     前向路由永远只读取"历史上已固化"的 EMA 中心 q_ema_mean，绝不触碰当前 Batch/当前句子的 dim=0 均值，
        #     因此 B=8 x T=1024 训练与 B=1 单步自回归推理逐 Token 数值严格恒等 (零未来数据泄露、零长度跳变)。
        self.register_buffer(
            "q_ema_mean", torch.zeros(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM, dtype=torch.float32, device=device)
        )
        self.register_buffer("q_ema_steps", torch.zeros((), dtype=torch.float32, device=device))

        # 2c. 二级球面去均值背景中心 (Post-Normalization Spherical De-Biasing Center) —— 封杀"反去均值偏度作弊"
        #     一级共模中心 q_ema_mean 扣在逐 Token 归一化之前；但紧随其后的 /std2 是逐 Token 的非线性缩放，
        #     对偏度点云 (多数 Token 紧贴锥心、少数离群) 归一化后，球面方向均值会重新长回非零。
        #     该残余均值同样以 EMA 方式固化并持久化，在最后一次线性内积前做纯线性平移扣除。
        self.register_buffer(
            "q_post_ema_mean", torch.zeros(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM, dtype=torch.float32, device=device)
        )
        self.register_buffer("q_post_ema_steps", torch.zeros((), dtype=torch.float32, device=device))

        # 2d. 流形球面化同胚变换 (Manifold-to-Sphere Homeomorphism T <-> T^-1):
        #     将各向异性的语料流形通过保体积对称 ZCA 度规变换 W = (Sigma + eps*I)^(-1/2) 映射为各向同性标准球面 S^{127}，
        #     在标准球面上执行严格正交均匀子键路由与微专家计算，最后经逆变换 W^{-1} = (Sigma + eps*I)^{+1/2}
        #     无损映射回原流形拓扑，自动实现"流形密集处微专家密集、稀疏处微专家稀疏"的保度规剖分。
        eye_sub = torch.eye(BKP_SUBKEY_DIM, dtype=torch.float32, device=device).view(
            1, 1, BKP_SUBKEY_DIM, BKP_SUBKEY_DIM
        )
        self.register_buffer(
            "q_ema_cov", eye_sub.expand(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM, BKP_SUBKEY_DIM).contiguous().clone()
        )
        self.register_buffer(
            "q_whiten_mat", eye_sub.expand(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM, BKP_SUBKEY_DIM).contiguous().clone()
        )
        self.register_buffer(
            "q_dewhiten_mat", eye_sub.expand(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM, BKP_SUBKEY_DIM).contiguous().clone()
        )
        self.register_buffer("q_cov_steps", torch.zeros((), dtype=torch.float32, device=device))

        # 2.5 子键码本预中心化标记 (1 表示 u_codebook/v_codebook 已经过球面零均值中心化并固化，无需重复减均值以防扩容漂移)
        self.register_buffer("codebook_pre_centered", torch.zeros((), dtype=torch.int32, device=device))
        self._codebook_pre_centered_cached: int = 0
        # 2.6 跨簇桥接包标记 (1 表示该包为保存 Split 非对角 2ab 跨簇突触的 Bridge_AB.bkp，环境空间定标时不重复计入子键)
        self.register_buffer("is_bridge_flag", torch.zeros((), dtype=torch.int32, device=device))
        self._is_bridge_cached: bool = False
        self._sparse_128_triblock_mode: Optional[bool] = None
        self._quality_q_cached: float = 1.0
        self._cov_step_counter: int = 0

        # 3. 4KB 页对齐微专家池 ([M_p, 2, 2048], int8, 存于 System RAM / OS mmap)
        self.expert_pages_int8: torch.Tensor = self._init_or_mmap_expert_pages(mmap_file_path, self.num_experts, mount_mode)
        if mmap_file_path is None:
            # 内存模式下初始化随机正交读取键 u_key，并将输出值 v_val 严格置 0
            rand_u = torch.randint(-64, 64, (self.num_experts, D_BKP), dtype=torch.int8)
            self.expert_pages_int8[:, 0, :].copy_(rand_u)
            self.expert_pages_int8[:, 1, :].zero_()

        # 4. 白皮书外置存储铁律与 14B 显存红线控制:
        #    - 全量稀疏动量池 (_sparse_m1_cpu, _sparse_m2_cpu) 与主权重池 (_master_uv_fp32) 统一采用紧凑 bfloat16，
        #      单颗 128x128 知识包主权重仅占 128MB (较 FP32 减半)，保障未来 14B 主干 + 多包森林在单卡 A100 上充裕运行；
        #    - 严格区分训练态与只读推理态:
        #      * 可写训练态 (RW_SHARED / COW_PRIVATE, K_p <= 256): 允许将活跃可写包驻留 GPU 以最大化 A100 拖拽吞吐；
        #      * 只读推理态 (RO_SYS / RO_LIB): 严格遵循白皮书三级页表契约，4KB 微专家页 100% 驻留磁盘/主机内存 OS mmap，
        #        GPU 显存仅驻留 ~256KB 的 L1 路由子键页表 (u_codebook, v_codebook)，前向命中后通过 PCIe 按需拉取进入 VRAMReadOnlyExpertCache！
        is_rw = mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE)
        n_slot = self.num_experts if is_rw else 0
        self._gpu_resident_pages: bool = bool(is_rw and device.type == "cuda" and self.grid_k <= 256)
        self._gpu_resident_train: bool = bool(is_rw and device.type == "cuda" and self.grid_k <= 256)
        buf_dev = device if self._gpu_resident_train else torch.device("cpu")
        grad_dtype = torch.float32

        self._sparse_steps_cpu: torch.Tensor = torch.zeros(n_slot, dtype=torch.int32, device=buf_dev)
        self._sparse_m1_cpu: torch.Tensor = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
        self._sparse_m2_cpu: torch.Tensor = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
        self._grad_acc_cpu: torch.Tensor = torch.zeros(0, dtype=grad_dtype, device=buf_dev)
        self._grad_hit_counts: torch.Tensor = torch.zeros(0, dtype=torch.float32, device=buf_dev)
        self._grad_dirty_mask: torch.Tensor = torch.zeros(n_slot, dtype=torch.bool, device=device)
        self._mmap_dirty_mask: torch.Tensor = torch.zeros(n_slot, dtype=torch.bool, device=buf_dev)
        self._expert_pages_dev: Optional[torch.Tensor] = (
            self.expert_pages_int8.to(device=device, copy=True) if self._gpu_resident_pages else None
        )
        self._last_active_eids: Optional[torch.Tensor] = None
        self._frozen_expert_mask_cpu: Optional[torch.Tensor] = None

        # 5. 训练态未量化主权重池 (Master Weights, 统一使用 bfloat16 节省 50% 主权重显存):
        master_dtype = torch.bfloat16
        self._master_uv_fp32: Optional[torch.Tensor] = (
            torch.zeros((n_slot, 2, D_BKP), dtype=master_dtype, device=buf_dev) if is_rw else None
        )
        if self._master_uv_fp32 is not None:
            self._init_master_from_pages()

    @torch.no_grad()
    def _init_master_from_pages(self) -> None:
        """从当前 INT8 物理页与解量化缩放因子反量化重建未量化 FP32 主权重 (原位转换，零额外大张量临时内存)"""
        if self._master_uv_fp32 is None or self._master_uv_fp32.numel() == 0:
            return
        dev = self._master_uv_fp32.device
        src_pages = self._expert_pages_dev if self._expert_pages_dev is not None else self.expert_pages_int8
        if src_pages.numel() != self.num_experts * 2 * D_BKP:
            return
        self._master_uv_fp32.copy_(src_pages)
        scale = self.scales.to(device=dev, dtype=self._master_uv_fp32.dtype).unsqueeze(-1)
        self._master_uv_fp32.mul_(scale)
        u = self._master_uv_fp32[:, 0, :]
        inv_rms = torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        u.mul_(inv_rms)

    def _init_or_mmap_expert_pages(
        self, file_path: Optional[str], num_experts: int, mode: BKPMountMode
    ) -> torch.Tensor:
        """
        将磁盘上的 .bkp 文件按 4KB 操作系统标准页映射至进程虚拟地址空间。
        修改该张量时，CPU 硬件 MMU 自动置位对应虚拟页 PTE 低 12 位中的 Dirty 位，无需软件维护脏页表。
        """
        total_bytes = num_experts * PAGE_SIZE
        if file_path is None:
            return torch.empty((num_experts, 2, D_BKP), dtype=torch.int8, device="cpu")

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.truncate(total_bytes)
                # 初始化 u_key 随机，v_val 全 0
                init_buf = torch.zeros((num_experts, 2, D_BKP), dtype=torch.int8)
                init_buf[:, 0, :] = torch.randint(-64, 64, (num_experts, D_BKP), dtype=torch.int8)
                f.write(init_buf.numpy().tobytes())

        if mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
            f_obj = open(file_path, "rb")
            mm = mmap.mmap(f_obj.fileno(), total_bytes, access=mmap.ACCESS_READ)
        elif mode == BKPMountMode.COW_PRIVATE:
            f_obj = open(file_path, "r+b")
            mm = mmap.mmap(f_obj.fileno(), total_bytes, access=mmap.ACCESS_COPY)
        else:
            f_obj = open(file_path, "r+b")
            mm = mmap.mmap(f_obj.fileno(), total_bytes, access=mmap.ACCESS_WRITE)

        self._file_obj = f_obj
        self._mmap_obj = mm
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            flat_tensor = torch.frombuffer(mm, dtype=torch.int8, count=total_bytes)
        return flat_tensor.view(num_experts, 2, D_BKP)
    
    @torch.no_grad()
    def sync_to_mmap(self) -> None:
        """将 GPU 页缓存驻留期间更新的脏微专家页批量刷回 CPU OS mmap 虚拟内存页 (避免每步触发随机 4KB 磁盘 I/O)"""
        if (
            self._expert_pages_dev is not None
            and self._mmap_dirty_mask.numel() > 0
            and self.expert_pages_int8.numel() == self.num_experts * 2 * D_BKP
            and self.mount_mode not in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB)
        ):
            if bool(self._mmap_dirty_mask.any().item()):
                dirty_cpu = torch.where(self._mmap_dirty_mask)[0].cpu()
                dirty_dev = dirty_cpu.to(self._expert_pages_dev.device)
                self.expert_pages_int8[dirty_cpu] = self._expert_pages_dev[dirty_dev].cpu()
                self._mmap_dirty_mask.zero_()
                if self._mmap_obj is not None and self.mount_mode == BKPMountMode.RW_SHARED:
                    self._mmap_obj.flush()

    def save_metadata(self, meta_path: str) -> None:
        torch.save(self.state_dict(), meta_path)

    def save_package(self, bkp_path: Optional[str] = None) -> str:
        """
        统一封装的知识包落盘接口 (对上层完全隐藏内部 4KB 页文件与页表元数据的物理存储细节):
        调用方只需传入一个 `.bkp` 路径 (或省略以使用当前绑定路径)，内部自动同步 4KB OS mmap 物理页与伴随页表状态。
        若知识包经历过 Auto-Expand 象限倍增扩容导致磁盘文件字节数与当前 num_experts * 4096 不一致，
        自动关闭旧尺寸 mmap、完整覆写新尺寸 4KB 物理页并重建 mmap 映射，彻底杜绝扩容截断隐患。
        """
        target_bkp = str(bkp_path or self.mmap_file_path)
        if not target_bkp:
            raise ValueError(f"知识包 {self.pkg_id} 未指定落盘路径")
        self.sync_to_mmap()
        expected_bytes = int(self.num_experts * PAGE_SIZE)
        path_differs = self._mmap_file_path_differs(target_bkp)
        disk_size_mismatch = (
            (not os.path.isfile(target_bkp))
            or (os.path.getsize(target_bkp) != expected_bytes)
            or (self._mmap_obj is None)
            or (len(self._mmap_obj) != expected_bytes)
        )
        if path_differs or disk_size_mismatch:
            src_pages = (
                self._expert_pages_dev.detach().cpu()
                if self._expert_pages_dev is not None
                else self.expert_pages_int8.detach().cpu()
            ).clone()
            if not path_differs and self._mmap_obj is not None:
                if self._mmap_obj is not None:
                    self._mmap_obj.close()
                    self._mmap_obj = None
                if self._file_obj is not None:
                    self._file_obj.close()
                    self._file_obj = None
            with open(target_bkp, "wb") as f:
                f.write(src_pages.contiguous().numpy().tobytes())
            if not path_differs and self.mmap_file_path:
                self.expert_pages_int8 = self._init_or_mmap_expert_pages(
                    self.mmap_file_path, self.num_experts, self.mount_mode
                )
        meta_path = target_bkp + ".meta.pt"
        torch.save(super().state_dict(), meta_path)
        return target_bkp

    def _mmap_file_path_differs(self, target_bkp: str) -> bool:
        if not self.mmap_file_path:
            return True
        try:
            return os.path.abspath(self.mmap_file_path) != os.path.abspath(target_bkp)
        except Exception:
            return True

    @classmethod
    def load_package(
        cls,
        bkp_path: str,
        mount_mode: BKPMountMode = BKPMountMode.RO_SYS,
        is_base_pkg: bool = False,
        device: torch.device = torch.device("cpu"),
        pkg_id: Optional[str] = None,
    ) -> "BKPPackage":
        """
        统一封装的知识包加载接口 (对外部完全隐藏 `.meta.pt` 等底层元数据实现细节):
        调用方只需指定 `.bkp` 知识包路径与挂载模式 (`RO_SYS` / `RW_SHARED`)，自动完成网格推断、4KB mmap 映射与流形状态恢复。
        """
        bkp_p = str(bkp_path)
        meta_p = bkp_p + ".meta.pt"
        if not os.path.isfile(bkp_p):
            raise FileNotFoundError(f"未找到 .bkp 知识包文件: {bkp_p}")
        file_bytes = os.path.getsize(bkp_p)
        num_experts = max(1, file_bytes // PAGE_SIZE)
        grid_k = int(round(math.sqrt(num_experts)))
        resolved_id = pkg_id if pkg_id is not None else os.path.basename(bkp_p)
        pkg = cls(
            pkg_id=resolved_id,
            grid_k=grid_k,
            mount_mode=mount_mode,
            is_base_pkg=is_base_pkg,
            mmap_file_path=bkp_p,
            device=device,
        )
        if os.path.isfile(meta_p):
            st = torch.load(meta_p, map_location=device, weights_only=True)
            pkg.load_state_dict(st, strict=False)
        return pkg

    @torch.no_grad()
    def set_mount_mode(
        self,
        new_mode: BKPMountMode,
        trainable_codebook: Optional[bool] = None,
        lock_manifold_metric: bool = True,
    ) -> None:
        """
        动态切换知识包挂载状态 (如从 RO_SYS 只读态切换为 RW_SHARED 增量微调态，或微调完毕切回 RO_SYS):
        - 若切入 RW_SHARED: 自动将只读 ACCESS_READ mmap 重新升级为可写 ACCESS_WRITE mmap，并重建 bfloat16 主权重池；
        - 若切回 RO_SYS: 自动将脏页刷回 OS mmap，重新降级为只读 ACCESS_READ mmap 并释放 GPU 训练缓冲。
        """
        old_mode = self.mount_mode
        if new_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
            self.sync_to_mmap()
            if self.mmap_file_path and old_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE):
                self.close_mmap()
                self.expert_pages_int8 = self._init_or_mmap_expert_pages(
                    self.mmap_file_path, self.num_experts, new_mode
                )
            self.mount_mode = new_mode
            self.u_codebook.requires_grad_(False)
            self.v_codebook.requires_grad_(False)
            self._master_uv_fp32 = None
            self._expert_pages_dev = None
            self._gpu_resident_pages = False
            self._gpu_resident_train = False
            return

        # 切入可写模式 (RW_SHARED / COW_PRIVATE): 若原为只读 ACCESS_READ mmap，必须重映射为可写 mmap
        if self.mmap_file_path and old_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
            self.close_mmap()
            self.expert_pages_int8 = self._init_or_mmap_expert_pages(
                self.mmap_file_path, self.num_experts, new_mode
            )
        self.mount_mode = new_mode
        dev = self.scales.device
        allow_cb = bool(trainable_codebook) if trainable_codebook is not None else (not lock_manifold_metric)
        self.u_codebook.requires_grad_(allow_cb)
        self.v_codebook.requires_grad_(allow_cb)
        self._gpu_resident_pages = bool(dev.type == "cuda" and self.grid_k <= 256)
        self._gpu_resident_train = bool(dev.type == "cuda" and self.grid_k <= 256)
        buf_dev = dev if self._gpu_resident_train else torch.device("cpu")
        n_slot = self.num_experts
        if self._expert_pages_dev is None and self._gpu_resident_pages:
            self._expert_pages_dev = self.expert_pages_int8.to(device=dev, copy=True)
        if self._master_uv_fp32 is None or self._master_uv_fp32.numel() != n_slot * 2 * D_BKP:
            self._sparse_steps_cpu = torch.zeros(n_slot, dtype=torch.int32, device=buf_dev)
            self._sparse_m1_cpu = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
            self._sparse_m2_cpu = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
            self._grad_dirty_mask = torch.zeros(n_slot, dtype=torch.bool, device=dev)
            self._mmap_dirty_mask = torch.zeros(n_slot, dtype=torch.bool, device=buf_dev)
            self._master_uv_fp32 = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
            self._init_master_from_pages()

    def state_dict(self, *args, **kwargs):
        self.sync_to_mmap()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
        """载入元数据后必须立即重建未量化主权重与主机侧标量缓存: 否则 scales 会用 __init__ 的默认初值反量化出错误的主权重真值"""
        result = super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
        self._codebook_pre_centered_cached = int(self.codebook_pre_centered.item())
        self._is_bridge_cached = bool(int(self.is_bridge_flag.item()) != 0)
        self._quality_q_cached = float(self.quality_q.clamp(min=1e-6, max=1.0).item())
        self._cov_step_counter = int(self.q_cov_steps.item())
        self._sparse_128_triblock_mode = None
        if self.mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE):
            self._init_master_from_pages()
        else:
            self._master_uv_fp32 = None
            self._expert_pages_dev = None
            self._gpu_resident_pages = False
        return result

    def close_mmap(self) -> None:
        """释放 OS mmap 与文件句柄 (Windows NTFS 重挂载或删除文件前必须调用)"""
        self.sync_to_mmap()
        self._expert_pages_dev = None
        self._master_uv_fp32 = None
        self.expert_pages_int8 = torch.empty(0, dtype=torch.int8)
        if self._mmap_obj is not None:
            self._mmap_obj.close()
            self._mmap_obj = None
        if self._file_obj is not None:
            self._file_obj.close()
            self._file_obj = None

    def normalized_codebooks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回特征维严格保持单位 L2 范数的行列子键码本 (8, K_p, 128)。
        - 对于常驻 Base.bkp (且未预中心化固化时): 沿簇维度 (dim=1) 执行零均值中心化以均匀覆盖全局语义球面；
        - 若已执行四象限扩容 (codebook_pre_centered == 1) 或为专有领域扩展包: 直接按特征维单位化，严防新增插值子键均值拖拽存量子键方向。
        """
        if self.is_base_pkg and self._codebook_pre_centered_cached == 0:
            u_c = self.u_codebook - self.u_codebook.mean(dim=1, keepdim=True)
            v_c = self.v_codebook - self.v_codebook.mean(dim=1, keepdim=True)
            return F.normalize(u_c, dim=-1), F.normalize(v_c, dim=-1)
        return F.normalize(self.u_codebook, dim=-1), F.normalize(self.v_codebook, dim=-1)

    def query_cone_center(self) -> torch.Tensor:
        """
        返回已固化的查询静态表征锥背景中心 (8, 2, 128)。
        该统计量随知识包元数据一起持久化，前向路由严格只读，任何单步推理都不会改动它。
        """
        return self.q_ema_mean

    @torch.no_grad()
    def update_query_cone_center(self, batch_mean: torch.Tensor, momentum: float = SUBSPACE_STD_SCALE) -> None:
        """
        以指数移动平均 (EMA / BatchNorm running stats 思想) 吸收当前训练批次的逐 Token 标准化查询均值:
        mu_ema <- (1 - beta) * mu_ema + beta * mean_{t}(q_hat_t), beta = max(momentum, 1/(k+1))
        - 前 1/momentum 步退化为精确算术平均 (Bias Correction)，快速固化全语料静态表征锥方向；
        - 之后转为几何遗忘 EMA，持续跟踪表征锥的缓慢漂移；
        - 关键性质: 更新只发生在前向计算完成之后，前向永远使用更新前的历史统计量 → 严格因果，且全程零 .item() 同步。
        """
        target = batch_mean.detach().float().reshape(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM).to(self.q_ema_mean.device)
        beta = torch.clamp(1.0 / (self.q_ema_steps + 1.0), min=float(momentum))
        self.q_ema_mean.mul_(1.0 - beta).add_(target * beta)
        # 仅保留极端数值安全上界 (0.99)，不截断高共模头的真实语料静态中心模长
        ema_rms = torch.sqrt(self.q_ema_mean.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        self.q_ema_mean.mul_((0.99 / ema_rms.clamp(min=0.99)))
        self.q_ema_steps.add_(1.0)

    @torch.no_grad()
    def _recompute_spherical_charts(self, shrinkage: Optional[float] = None) -> None:
        """
        由固化的子空间协方差度规 q_ema_cov 闭式求解保体积流形球面化正逆同胚算子:
        - 白化正则项 rho^*: 采用随机矩阵 Marchenko-Pastur / Ledoit-Wolf 闭式最优收缩率
          rho^* = min(1.0, d_sub / N_eff)，其中 N_eff 为历史有效观测样本规模，无需人为指定经验常数；
        - 正向球面化 W_s = (Sigma_s)^(-1/2): 压缩高方差大头主轴、拉伸放大密集簇内部切空间，将流形展开至标准球面；
        - 逆向还原 W_s^{-1} = (Sigma_s)^(+1/2): 将标准球面上的微专家响应精确映射回原流形拓扑，且 W_s @ W_s^{-1} == I。
        """
        d = BKP_SUBKEY_DIM
        if shrinkage is None:
            # 每步包含至少 d 个独立 Token 切片，有效累积样本量 N_eff = d * (1 + q_cov_steps)
            steps_eff = max(1.0, float(self.q_cov_steps.item()))
            n_eff = float(d) * (1.0 + steps_eff * 0.5)
            rho = min(1.0, float(d) / n_eff)
        else:
            rho = float(shrinkage)
        cov_flat = self.q_ema_cov.float().reshape(-1, d, d)
        cov_sym = 0.5 * (cov_flat + cov_flat.transpose(-1, -2))
        eye = torch.eye(d, dtype=torch.float32, device=cov_sym.device).unsqueeze(0)
        cov_reg = (1.0 - rho) * cov_sym + rho * eye
        eigvals, eigvecs = torch.linalg.eigh(cov_reg)
        eig_clamped = eigvals.clamp(min=0.20, max=16.0)
        # 几何均值归一化使 det(W_s) = det(W_s^{-1}) = 1.0 (严格保体积幺模度规变换)
        geom_mean = torch.exp(torch.log(eig_clamped).mean(dim=-1, keepdim=True)).clamp(min=1e-6)
        eig_unit = (eig_clamped / geom_mean).clamp(min=0.16, max=16.0)
        inv_sqrt = eig_unit.rsqrt()
        fwd_sqrt = eig_unit.sqrt()
        w_mat = torch.bmm(eigvecs * inv_sqrt.unsqueeze(-2), eigvecs.transpose(-1, -2))
        dew_mat = torch.bmm(eigvecs * fwd_sqrt.unsqueeze(-2), eigvecs.transpose(-1, -2))
        self.q_whiten_mat.copy_(w_mat.view(BKP_NUM_HEADS, 2, d, d))
        self.q_dewhiten_mat.copy_(dew_mat.view(BKP_NUM_HEADS, 2, d, d))

    @torch.no_grad()
    def update_query_manifold_metric(self, batch_cov: torch.Tensor, momentum: float = SUBSPACE_STD_SCALE) -> None:
        """以 EMA 吸收当前批次去均值查询的二阶流形协方差度规，并刷新球面化正逆变换矩阵 (零 .item() 同步)"""
        d = BKP_SUBKEY_DIM
        target = batch_cov.detach().float().reshape(BKP_NUM_HEADS, 2, d, d).to(self.q_ema_cov.device)
        tr = target.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
        target = target / tr
        beta = torch.clamp(1.0 / (self.q_cov_steps + 1.0), min=float(momentum))
        self.q_ema_cov.mul_(1.0 - beta).add_(target * beta)
        self.q_cov_steps.add_(1.0)
        self._cov_step_counter += 1
        if self._cov_step_counter <= 12 or (self._cov_step_counter % 4 == 0):
            self._recompute_spherical_charts()

    def query_post_cone_center(self) -> torch.Tensor:
        """
        返回逐 Token 归一化之后残留的球面方向均值 (8, 2, 128) —— 二级去偏基线。
        与一级 q_ema_mean 一样，前向路由严格只读历史固化值，故逐 Token 因果独立性 100% 保持。
        """
        return self.q_post_ema_mean

    @torch.no_grad()
    def update_query_post_cone_center(self, batch_mean: torch.Tensor, momentum: float = SUBSPACE_STD_SCALE) -> None:
        """
        以 EMA 吸收当前训练批次的"归一化后球面方向均值" E_t[q_final_t] (零 .item() 同步):
        - 该均值是 /std2 逐 Token 非线性除法的产物 (减完一级中心后 E_t[diff]=0，但 E_t[diff/|diff|] ≠ 0)；
        - 若不在最后一次线性内积前扣除它，正交子键中唯一对齐该偏度轴的那一个会永久白嫖
          +1.64 sigma 的静态起跑底分，凭正交排斥势 (MaxCos -> 0.058) 形成他人无法竞争的护城河；
        - 同理前 1/momentum 步退化为精确算术平均 (Bias Correction)，之后转为几何遗忘 EMA。
        """
        target = batch_mean.detach().float().reshape(BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM).to(self.q_post_ema_mean.device)
        beta = torch.clamp(1.0 / (self.q_post_ema_steps + 1.0), min=float(momentum))
        self.q_post_ema_mean.mul_(1.0 - beta).add_(target * beta)
        # 由 Jensen 不等式，逐坐标 RMS 归一化向量的方向均值满足 ||m||_rms <= 1.0，此处仅作数值安全上界
        post_rms = torch.sqrt(self.q_post_ema_mean.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        self.q_post_ema_mean.mul_((1.0 / post_rms.clamp(min=1.0)))
        self.q_post_ema_steps.add_(1.0)

    def compute_codebook_dispersion_loss(self) -> torch.Tensor:
        """
        微观切空间球面库仑排斥与最坏近邻硬屏障 (Spherical Coulomb Repulsion + Symmetry-Breaking MaxCos Barrier):
        为什么普通余弦平方惩罚 (cos^2) 挡不住子键完全重合 (MaxCos -> 0.999)?
        - 因为球面归一化 F.normalize(u) 的雅可比矩阵使 d(cos theta)/du 的模长正比于 sin(theta) -> 0！
          当两个子键被强梯度拉近到 theta -> 0 (cos -> 1) 时，普通多项式损失的切向排斥梯度反而衰减为 0 (球面奇点死锁)！
        解决方案:
        1. 对称破缺保护: 若检测到同头内存在余弦 > 0.95 的重合克隆子键 (j < i)，无分支原位施加正交微扰打破 theta=0 奇点；
        2. 库仑对数屏障势 -log(1 - excess): 其导数 1/(1 - cos) 与雅可比因子 sin(theta) 相消后产生 ~ 2/theta 的库仑静电排斥力，
           距离越近斥力越强，从根源上杜绝子键抱团克隆！
        """
        if self.training and self.u_codebook.requires_grad:
            with torch.no_grad():
                for cb in (self.u_codebook, self.v_codebook):
                    cb_t = F.normalize(cb.data.float() - cb.data.float().mean(dim=1, keepdim=True), dim=-1)
                    # 无分支原位微扰: 消除 clone_mask.any().item() CPU-GPU 同步屏障
                    g_triu = torch.triu(torch.bmm(cb_t, cb_t.transpose(1, 2)), diagonal=1)
                    max_prev, _ = g_triu.max(dim=1)
                    clone_mask = (max_prev > 0.95).unsqueeze(-1)
                    noise = torch.randn_like(cb.data)
                    dot = (noise * cb.data).sum(dim=-1, keepdim=True)
                    ortho_dir = F.normalize(noise - dot * cb.data, dim=-1)
                    perturbed = F.normalize(cb.data + 0.05 * ortho_dir, dim=-1)
                    cb.data.copy_(torch.where(clone_mask, perturbed, cb.data))

        u_tangent, v_tangent = self.normalized_codebooks()
        u_tangent = u_tangent.float()
        v_tangent = v_tangent.float()
        k_p = u_tangent.size(1)
        eye = torch.eye(k_p, device=u_tangent.device, dtype=torch.float32).unsqueeze(0)

        gram_u = torch.bmm(u_tangent, u_tangent.transpose(1, 2))
        gram_v = torch.bmm(v_tangent, v_tangent.transpose(1, 2))
        gram_u_zero_diag = gram_u * (1.0 - eye)
        gram_v_zero_diag = gram_v * (1.0 - eye)
        gram_u_off = gram_u - eye * 2.0
        gram_v_off = gram_v - eye * 2.0

        # 1. 全局紧框架正交体积项 (Tight-Frame Orthogonality)
        loss_mean = (gram_u_zero_diag.pow(2).sum(dim=-1).mean() + gram_v_zero_diag.pow(2).sum(dim=-1).mean()) * 0.15

        # 2. 逐子键最近邻库仑对数排斥势
        max_nn_u = gram_u_off.amax(dim=-1)  # (8, K_p)
        max_nn_v = gram_v_off.amax(dim=-1)  # (8, K_p)
        exc_u = F.relu(max_nn_u + 0.02).clamp(max=0.95)
        exc_v = F.relu(max_nn_v + 0.02).clamp(max=0.95)
        loss_coulomb = (-torch.log(1.0 - exc_u) - torch.log(1.0 - exc_v)).mean() * 1.2

        # 3. 全局最坏近邻 MaxCos 连续对数势
        peak_u = F.relu(max_nn_u.max() - 0.05).clamp(max=0.95)
        peak_v = F.relu(max_nn_v.max() - 0.05).clamp(max=0.95)
        loss_peak = (-torch.log(1.0 - peak_u) - torch.log(1.0 - peak_v)) * 1.5

        return loss_mean + loss_coulomb + loss_peak

    _SR_POOL_ROWS: int = 8192
    _sr_noise_pool_cpu: Optional[torch.Tensor] = None
    _sr_noise_pool_dev: Optional[torch.Tensor] = None
    _sr_phase: int = 0

    @classmethod
    def quantize_per_expert_int8(cls, vec_fp: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单个或一批微专家向量 (..., 2048) 执行对称 INT8 量化:
        支持 GPU/CPU 双端预生成环形伪随机相位池实现 O(1) 零开销无偏随机舍入 (Stochastic Rounding)。
        """
        raw_max = vec_fp.abs().amax(dim=-1, keepdim=True)
        max_abs = raw_max.clamp(min=1e-6)
        step_size = (max_abs / 127.0).to(torch.float16).float().clamp(min=1e-7)
        scaled = vec_fp / step_size
        if stochastic:
            if scaled.dim() == 2 and scaled.size(1) == D_BKP:
                n_rows = scaled.size(0)
                req_rows = max(cls._SR_POOL_ROWS, n_rows) * 2
                if scaled.is_cpu:
                    if cls._sr_noise_pool_cpu is None or cls._sr_noise_pool_cpu.size(0) < req_rows:
                        cls._sr_noise_pool_cpu = torch.empty((req_rows, D_BKP), dtype=torch.float32, device="cpu").uniform_(-0.5, 0.5)
                    pool = cls._sr_noise_pool_cpu
                else:
                    if cls._sr_noise_pool_dev is None or cls._sr_noise_pool_dev.device != scaled.device or cls._sr_noise_pool_dev.size(0) < req_rows:
                        cls._sr_noise_pool_dev = torch.empty((req_rows, D_BKP), dtype=torch.float32, device=scaled.device).uniform_(-0.5, 0.5)
                    pool = cls._sr_noise_pool_dev
                max_start = pool.size(0) - n_rows
                cls._sr_phase = (cls._sr_phase + 131) % max(1, max_start)
                noise = pool[cls._sr_phase : cls._sr_phase + n_rows]
                quant = torch.round(scaled + noise)
            else:
                noise = torch.empty_like(scaled).uniform_(-0.5, 0.5)
                quant = torch.round(scaled + noise)
        else:
            quant = torch.round(scaled)
        vec_int8 = torch.where(raw_max > 0.0, quant.clamp(-127, 127), torch.zeros_like(quant)).to(torch.int8)
        out_scale = torch.where(raw_max > 0.0, step_size, torch.zeros_like(step_size)).squeeze(-1).to(torch.float16)
        return vec_int8, out_scale

    @torch.no_grad()
    def write_experts_to_mmap_pages(
        self,
        expert_ids: torch.Tensor,
        u_key_fp: torch.Tensor,
        v_val_fp: torch.Tensor,
    ) -> None:
        """
        写入微专家向量:
        1. 若存在未量化 BF16 主权重池 (可写训练态)，先将连续浮点真值原位无损写入主权重池；
        2. 再由主权重 (或直接由输入) 执行 INT8 量化，刷新 4KB 物理页与解量化缩放因子。
           存在主权重时使用确定性最近邻舍入 (Round-to-Nearest, 最小均方误差)，
           因为亚量化步长的微小更新已由主权重无损累积，无需再用随机舍入制造额外方差；
           不存在主权重 (纯外科手术 / 推理侧不适用可写态) 时退回无偏随机舍入。
        CPU 执行物理页写操作时，硬件 MMU 自动将对应 4KB 页的 PTE Dirty 位翻转为 1。
        """
        if self.mount_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
            raise PermissionError(f"知识包 {self.pkg_id} 处于硬件只读保护模式 ({self.mount_mode.value})，拒绝写入！")

        u_fp = u_key_fp.detach().float()
        u_fp = u_fp * torch.rsqrt(u_fp.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        v_fp = v_val_fp.detach().float()
        dev_ids = expert_ids.detach().to(self.scales.device).long()

        has_master = self._master_uv_fp32 is not None and self._master_uv_fp32.numel() > 0
        if has_master:
            m_dev = self._master_uv_fp32.device
            m_ids = dev_ids if dev_ids.device == m_dev else dev_ids.to(device=m_dev)
            uv_stacked = torch.stack([u_fp, v_fp], dim=1).to(device=m_dev, dtype=self._master_uv_fp32.dtype)
            self._master_uv_fp32.index_copy_(0, m_ids, uv_stacked)

        # 将 (u, v) 合并为 (2*N, 2048) 一次性完成 GPU/CPU 向量化 INT8 量化
        n_exp = u_fp.size(0)
        uv_fp = torch.cat([u_fp, v_fp], dim=0)
        uv_q, s_uv = self.quantize_per_expert_int8(uv_fp, stochastic=not has_master)
        pages_q = torch.stack([uv_q[:n_exp], uv_q[n_exp:]], dim=1)

        if self._expert_pages_dev is not None:
            self._expert_pages_dev[dev_ids] = pages_q.to(self._expert_pages_dev.device)
            if self._mmap_dirty_mask.numel() > 0:
                self._mmap_dirty_mask[dev_ids.to(self._mmap_dirty_mask.device)] = True
            else:
                self.expert_pages_int8.index_copy_(0, dev_ids.cpu(), pages_q.cpu())
        else:
            cpu_ids = expert_ids.detach().cpu().long()
            self.expert_pages_int8.index_copy_(0, cpu_ids, pages_q.cpu())

        self.scales[dev_ids] = torch.stack([s_uv[:n_exp], s_uv[n_exp:]], dim=1).to(self.scales.device)

    @torch.no_grad()
    def read_experts_from_mmap_pages(self, expert_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 4KB 物理页（优先 GPU 镜像缓冲，否则从 CPU mmap 缓冲）读取并解量化微专家 (u_key_fp, v_val_fp)。
        """
        dev = self.scales.device
        dev_ids = expert_ids.detach().to(dev).long()
        sc = self.scales[dev_ids].float()  # [N, 2]
        if self._expert_pages_dev is not None:
            pages = self._expert_pages_dev[dev_ids.to(self._expert_pages_dev.device)].to(dev).float()
        else:
            pages = self.expert_pages_int8[dev_ids.cpu()].to(dev).float()
        u_fp = pages[:, 0, :] * sc[:, 0:1]
        v_fp = pages[:, 1, :] * sc[:, 1:2]
        return u_fp, v_fp

    @torch.no_grad()
    def update_quality_index(self) -> float:
        """
        更新晶格质量指数 Q_p = (C_route * C_purity * C_ortho)^(1/3) in [0, 1] (白皮书 2.2.4 节):
        - C_route : 晶格激活分布香农有效均衡度与单 Token 路由清晰度的几何合成
        - C_purity: 活跃微专家球面余弦相干纯度均值
        - C_ortho : 切空间子键码本最坏最近邻正交裕度 (1 - max_cos) 与微专家谱有效秩率的合成
        """
        active_mask = self.activation_counts > 0
        m_act = int(active_mask.sum().item())
        if m_act == 0:
            self.quality_q.fill_(1.0)
            self._quality_q_cached = 1.0
            return 1.0

        # 1. C_route: 晶格激活分布香农均衡覆盖率 C_balance (消除全 Batch 拍平大锅饭熵将均衡激活误判为 0.02 的问题)
        counts_act = self.activation_counts[active_mask].float()
        p_cnt = counts_act / (counts_act.sum() + 1e-12)
        ent_cnt = -(p_cnt * torch.log(p_cnt.clamp(min=1e-12))).sum().item()
        c_balance = float(min(1.0, max(0.01, ent_cnt / math.log(max(2, self.num_experts)))))
        c_route = c_balance

        # 2. C_purity: 活跃微专家的球面矢端相干纯度均值
        c_purity = float(self.purity_r[active_mask].mean().clamp(0.01, 1.0).item())

        # 3. C_ortho: 切空间子键码本最坏最近邻正交裕度 (1 - max_{i!=j} cos) 结合微专家谱有效秩
        u_tan = F.normalize(self.u_codebook.float() - self.u_codebook.float().mean(dim=1, keepdim=True), dim=-1)
        v_tan = F.normalize(self.v_codebook.float() - self.v_codebook.float().mean(dim=1, keepdim=True), dim=-1)
        k_p = u_tan.size(1)
        eye = torch.eye(k_p, device=u_tan.device, dtype=torch.float32).unsqueeze(0)
        gu_max = (torch.bmm(u_tan, u_tan.transpose(1, 2)) - eye * 2.0).max().item()
        gv_max = (torch.bmm(v_tan, v_tan.transpose(1, 2)) - eye * 2.0).max().item()
        c_codebook = float(min(1.0, max(0.01, 1.0 - max(gu_max, gv_max))))
        c_ortho = c_codebook

        q_val = (c_route * c_purity * c_ortho) ** (1.0 / 3.0)
        self.quality_q.fill_(q_val)
        self._quality_q_cached = float(min(1.0, max(1e-6, q_val)))
        return q_val

    @torch.no_grad()
    def detect_polysemy_interference_mask(self) -> torch.Tensor:
        """
        零超参数多义干涉闭式判据 (Rayleigh Uniformity & Orthogonal Pair Superposition Criterion):
            H_m * R_m^2 > ln(N_grid)  AND  R_m < 1 / sqrt(2)
        - H_m * R_m^2 > ln(N_grid): 瑞利球面随机游走检验，证明该微专家具有超越网格随机噪声的统计显著激活积累；
        - R_m < 1 / sqrt(2): 两个正交概念等权叠加时的合成球面模长精确为 ||0.5*g_1 + 0.5*g_2|| = 1/sqrt(2) ≈ 0.7071，
          低于此界说明该微专家承载了至少两个正交概念的多义干涉叠加态。
        """
        h_m = self.activation_counts.float()
        r_m = self.purity_r.float()
        ln_grid = math.log(max(4.0, float(self.grid_k)))
        return (h_m * r_m.pow(2) > ln_grid) & (r_m < SIMPLEX_2_GEODESIC_PROJ)

    # --------------------------------------------------------------------------
    # 拓扑生命周期管理: 四象限扩容 (a+b)^2、最小二乘缩容、谱双聚类分裂与主对角线合并
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _sync_device_pages_to_cpu_buffer(self) -> None:
        """拓扑变换前将 GPU 常驻脏页同步回 CPU expert_pages_int8 (严禁向 RO_SYS 的 ACCESS_READ 只读 mmap 内存页写入，防止 Linux SIGSEGV)"""
        if self.mount_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
            return
        if self._expert_pages_dev is None or self.expert_pages_int8.numel() != self._expert_pages_dev.numel():
            return
        if self._mmap_dirty_mask.numel() == 0 or not bool(self._mmap_dirty_mask.any().item()):
            return
        dirty_cpu = torch.where(self._mmap_dirty_mask)[0].cpu()
        dirty_dev = dirty_cpu.to(self._expert_pages_dev.device)
        self.expert_pages_int8[dirty_cpu] = self._expert_pages_dev[dirty_dev].cpu()
        self._mmap_dirty_mask.zero_()

    @torch.no_grad()
    def evict_gpu_cache(self) -> None:
        """显式卸载本包的 GPU 物理页镜像，将显存归还给系统 (CPU expert_pages_int8 完好保留)"""
        self._sync_device_pages_to_cpu_buffer()
        self._expert_pages_dev = None
        self._gpu_resident_pages = False

    @torch.no_grad()
    def _rebuild_runtime_buffers_after_topology_change(self) -> None:
        """拓扑尺寸或页内容改变后，重建 GPU 轻量页缓存与 CPU 稀疏优化器状态 (严守白皮书外置存储契约)"""
        dev = self.u_codebook.device
        is_rw = self.mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE)
        n_slot = self.num_experts if is_rw else 0
        self._gpu_resident_pages = bool(is_rw and dev.type == "cuda" and self.grid_k <= 256)
        self._gpu_resident_train = bool(is_rw and dev.type == "cuda" and self.grid_k <= 256)
        buf_dev = dev if self._gpu_resident_train else torch.device("cpu")
        grad_dtype = torch.float32

        # 先释放旧的大张量引用，避免新旧缓冲同时驻留
        self._expert_pages_dev = None
        self._master_uv_fp32 = None

        if self._sparse_steps_cpu.numel() != n_slot or self._sparse_steps_cpu.device != buf_dev:
            self._sparse_steps_cpu = torch.zeros(0, dtype=torch.int32)
            self._sparse_m1_cpu = torch.zeros(0, dtype=torch.bfloat16)
            self._sparse_m2_cpu = torch.zeros(0, dtype=torch.bfloat16)
            self._grad_acc_cpu = torch.zeros(0, dtype=torch.float32)
            self._sparse_steps_cpu = torch.zeros(n_slot, dtype=torch.int32, device=buf_dev)
            self._sparse_m1_cpu = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
            self._sparse_m2_cpu = torch.zeros((n_slot, 2, D_BKP), dtype=torch.bfloat16, device=buf_dev)
            self._grad_hit_counts = torch.zeros(0, dtype=torch.float32, device=buf_dev)
            self._grad_dirty_mask = torch.zeros(n_slot, dtype=torch.bool, device=dev)
            self._mmap_dirty_mask = torch.zeros(n_slot, dtype=torch.bool, device=buf_dev)

        self._expert_pages_dev = (
            self.expert_pages_int8.to(device=dev, copy=True) if self._gpu_resident_pages else None
        )
        master_dtype = torch.bfloat16
        self._master_uv_fp32 = (
            torch.zeros((n_slot, 2, D_BKP), dtype=master_dtype, device=buf_dev) if is_rw else None
        )
        self._sparse_128_triblock_mode = None
        if self._master_uv_fp32 is not None:
            self._init_master_from_pages()

    @torch.no_grad()
    def _copy_manifold_stats_from(self, src: "BKPPackage") -> None:
        """继承源包的查询锥中心与球面白化度规，保证分裂/合并后的子包共享同一寻址流形坐标系"""
        self.q_ema_mean.copy_(src.q_ema_mean.to(self.q_ema_mean.device))
        self.q_ema_steps.copy_(src.q_ema_steps.to(self.q_ema_steps.device))
        self.q_post_ema_mean.copy_(src.q_post_ema_mean.to(self.q_post_ema_mean.device))
        self.q_post_ema_steps.copy_(src.q_post_ema_steps.to(self.q_post_ema_steps.device))
        self.q_ema_cov.copy_(src.q_ema_cov.to(self.q_ema_cov.device))
        self.q_whiten_mat.copy_(src.q_whiten_mat.to(self.q_whiten_mat.device))
        self.q_dewhiten_mat.copy_(src.q_dewhiten_mat.to(self.q_dewhiten_mat.device))
        self.q_cov_steps.copy_(src.q_cov_steps.to(self.q_cov_steps.device))
        self.quality_q.copy_(src.quality_q.to(self.quality_q.device))
        self._quality_q_cached = getattr(src, "_quality_q_cached", 1.0)
        self._cov_step_counter = getattr(src, "_cov_step_counter", 0)

    @property
    def is_bridge_pkg(self) -> bool:
        return self._is_bridge_cached

    @is_bridge_pkg.setter
    def is_bridge_pkg(self, val: bool) -> None:
        flag_b = bool(val)
        self._is_bridge_cached = flag_b
        self.is_bridge_flag.fill_(1 if flag_b else 0)

    @torch.no_grad()
    def enable_empty_pages_for_training(
        self,
        init_scale_v: Optional[float] = None,
        freeze_existing_pages: bool = False,
        sparse_cross_bridge_only: bool = False,
    ) -> int:
        """
        将 scale_v=0 的空白微专家页激活为可训练态 (v_val 仍为 0，前向不变，但可接收反向梯度)。
        若 freeze_existing_pages=True，则锚定冻结原本已非空的成熟微专家页，仅允许新解锁的空白象限页吸收残差梯度！
        若 sparse_cross_bridge_only=True 且 grid_k >= 128，则在 2ab 非对角交叉象限中仅解锁 (k/4 x k/4) 跨簇桥接核心区，
        确保后续 Auto-Split 裂变时 100% 跨簇微专家可零丢弃无损装入单颗 K/2 x K/2 的 Bridge_AB.bkp 桥接包！
        """
        empty_mask = self.scales[:, 1] == 0
        if freeze_existing_pages:
            self._frozen_expert_mask_cpu = (~empty_mask).detach()
        else:
            self._frozen_expert_mask_cpu = None
        if self.is_bridge_pkg:
            k = self.grid_k
            half = k // 2
            elig_2d = torch.ones((k, k), dtype=torch.bool, device=empty_mask.device)
            elig_2d[:half, :half] = False
            elig_2d[half:, half:] = False
            empty_mask = empty_mask & elig_2d.view(-1)
        elif sparse_cross_bridge_only and self.grid_k >= 128:
            k = self.grid_k
            half = k // 2
            quarter = k // 4
            elig_2d = torch.zeros((k, k), dtype=torch.bool, device=empty_mask.device)
            elig_2d[:half, :half] = True
            elig_2d[half:, half:] = True
            elig_2d[:quarter, half : half + quarter] = True
            elig_2d[half : half + quarter, :quarter] = True
            empty_mask = empty_mask & elig_2d.view(-1)
        n_empty = int(empty_mask.sum().item())
        if n_empty > 0:
            s_val = float(init_scale_v) if init_scale_v is not None else (1.0 / math.sqrt(D_BKP))
            self.scales[empty_mask, 1] = s_val
            if self._master_uv_fp32 is not None:
                self._init_master_from_pages()
        self._sparse_128_triblock_mode = None
        return n_empty

    @torch.no_grad()
    def seal_unwoken_empty_pages(self, min_activations: int = 1) -> int:
        """训练结束后将仍未被唤醒、激活不足 min_activations 次或值向量仍为全零的微专家页重新封印 (scales[:, 1] = 0)，防止半生空页在推理时稀释信号"""
        self._sync_device_pages_to_cpu_buffer()
        # 先释放训练期大张量缓冲，并采用纯布尔 any(dim=-1) 避免 int8.sum() 提升为 int64 触发 CPU Cgroup OOM
        self._master_uv_fp32 = None
        self._sparse_m1_cpu = torch.zeros(0, dtype=torch.bfloat16)
        self._sparse_m2_cpu = torch.zeros(0, dtype=torch.bfloat16)
        self._sparse_steps_cpu = torch.zeros(0, dtype=torch.int32)
        zero_v_mask = (~(self.expert_pages_int8[:, 1, :] != 0).any(dim=-1)).to(self.scales.device)
        unwoken_mask = (self.activation_counts < max(1, min_activations)) | zero_v_mask
        n_sealed = int(unwoken_mask.sum().item())
        if n_sealed > 0:
            self.scales[unwoken_mask, 1] = 0.0
        self._frozen_expert_mask_cpu = None
        return n_sealed

    @torch.no_grad()
    def crystallize_biorthogonal_manifold_closure(
        self,
        min_mature_scale_v: float = 5e-5,
        closure_scale_ratio: float = 0.85,
        max_crystallize_slots: int = 1024,
        cluster_a_mask: Optional[torch.Tensor] = None,
        cluster_b_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        正交笛卡尔流形闭包成核算子 (Biorthogonal Cartesian Manifold Closure Nucleation):
        1. 寻址正交性: 交叉格点 (i*, j*) 位于正交码本网格 U_{i*} x V_{j*} 上，与同行/同列父专家天然正交；
        2. 读取键正交直和组装: 左半键 u^{(1)} 继承同行成熟主轴，右半键 u^{(2)} 继承同列成熟主轴 (R^1024 (+) R^1024)；
        3. 载荷重心闭包合成: 输出值向量 v_{i*,j*} 由同行与同列成熟微专家的值向量执行球面重心合成与正交去冗余。
        """
        self._sync_device_pages_to_cpu_buffer()
        k = self.grid_k
        dev = self.scales.device
        sv_2d = self.scales[:, 1].float().view(k, k)
        mature_2d = sv_2d > float(min_mature_scale_v)

        if cluster_a_mask is not None and cluster_b_mask is not None:
            ma_2d = (cluster_a_mask.to(dev).bool().view(k, k)) & mature_2d
            mb_2d = (cluster_b_mask.to(dev).bool().view(k, k)) & mature_2d
            ra = ma_2d.any(dim=1)
            ca = ma_2d.any(dim=0)
            rb = mb_2d.any(dim=1)
            cb = mb_2d.any(dim=0)
            cross_ab = (ra.unsqueeze(1) & cb.unsqueeze(0)) | (rb.unsqueeze(1) & ca.unsqueeze(0))
            closure_cand_2d = cross_ab & (~mature_2d)
        else:
            row_has_mature = mature_2d.any(dim=1)
            col_has_mature = mature_2d.any(dim=0)
            closure_cand_2d = (row_has_mature.unsqueeze(1) & col_has_mature.unsqueeze(0)) & (~mature_2d)
        cand_rc = torch.nonzero(closure_cand_2d, as_tuple=False)
        if cand_rc.numel() == 0:
            return {
                "crystallized_count": 0,
                "mature_before": int(mature_2d.sum().item()),
                "mature_after": int(mature_2d.sum().item()),
                "mean_operator_offdiag_cos": 0.0,
            }

        pages_cpu = self.expert_pages_int8.float()
        scales_cpu = self.scales.float().cpu()
        u_all = (pages_cpu[:, 0, :] * scales_cpu[:, 0:1]).view(k, k, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM)
        v_all = (pages_cpu[:, 1, :] * scales_cpu[:, 1:2]).view(k, k, D_BKP)
        sv_cpu = sv_2d.cpu()
        mature_cpu = mature_2d.cpu()

        w_mat = (sv_cpu * mature_cpu.float()).unsqueeze(-1)
        row_w_sum = w_mat.sum(dim=1, keepdim=True).clamp(min=1e-12)
        col_w_sum = w_mat.sum(dim=0, keepdim=True).clamp(min=1e-12)

        w_u = (sv_cpu * mature_cpu.float()).view(k, k, 1, 1)
        row_u1 = (u_all[:, :, :, 0, :] * w_u).sum(dim=1) / row_w_sum.view(k, 1, 1)
        col_u2 = (u_all[:, :, :, 1, :] * w_u).sum(dim=0) / col_w_sum.view(1, k, 1, 1).squeeze(0)

        row_v_bar = (v_all * w_mat).sum(dim=1) / row_w_sum.squeeze(1)
        col_v_bar = (v_all * w_mat).sum(dim=0) / col_w_sum.squeeze(0)

        r_idx_all = cand_rc[:, 0].cpu()
        c_idx_all = cand_rc[:, 1].cpu()
        row_peak_sv = (sv_cpu * mature_cpu.float()).max(dim=1).values
        col_peak_sv = (sv_cpu * mature_cpu.float()).max(dim=0).values
        cand_priority = torch.sqrt((row_peak_sv[r_idx_all] * col_peak_sv[c_idx_all]).clamp(min=1e-12))
        stream_ptr = getattr(self, "stream_next_ptr", None)
        if stream_ptr is not None:
            sp_cpu = stream_ptr.detach().cpu()
            src_e, slot_j = torch.nonzero(sp_cpu >= 0, as_tuple=True)
            if src_e.numel() > 0:
                dst_e = sp_cpu[src_e, slot_j].long()
                s_mask = torch.zeros((k, k), dtype=torch.float32)
                s_mask[src_e // k, dst_e % k] = 100.0
                cand_priority = cand_priority * (1.0 + s_mask[r_idx_all, c_idx_all])
        if cand_rc.shape[0] > max_crystallize_slots:
            _, top_ord = torch.topk(cand_priority, k=int(max_crystallize_slots), largest=True)
            r_sel = r_idx_all[top_ord]
            c_sel = c_idx_all[top_ord]
        else:
            r_sel = r_idx_all
            c_sel = c_idx_all

        n_cryst = int(r_sel.numel())
        flat_eids_cpu = r_sel * k + c_sel

        # 组装正交直和读取键 u_new: 左半继承行主轴 + 列特异正交切向分量，右半继承列主轴 + 行特异正交切向分量
        # 既保证左/右半空间正交直和 (R^1024 (+) R^1024)，又保证同行不同列格点之间满秩线性独立
        u_cb_cpu = self.u_codebook.detach().float().cpu()  # (8, k, 128)
        v_cb_cpu = self.v_codebook.detach().float().cpu()  # (8, k, 128)
        u1_base = F.normalize(row_u1[r_sel], dim=-1)
        u2_base = F.normalize(col_u2[c_sel], dim=-1)
        v_col_sig = v_cb_cpu[:, c_sel, :].permute(1, 0, 2)  # (n_cryst, 8, 128)
        u_row_sig = u_cb_cpu[:, r_sel, :].permute(1, 0, 2)  # (n_cryst, 8, 128)
        v_col_ortho = F.normalize(v_col_sig - (v_col_sig * u1_base).sum(dim=-1, keepdim=True) * u1_base, dim=-1)
        u_row_ortho = F.normalize(u_row_sig - (u_row_sig * u2_base).sum(dim=-1, keepdim=True) * u2_base, dim=-1)
        # 采用 1-Jet 单纯形无符号翻转闭式能量配比: alpha^* = sqrt(2/3), ||u_perp^*|| = sqrt(1/3)
        u1_new = F.normalize(JET_NORMAL_ALPHA_STAR * u1_base + JET_TANGENT_NORM_STAR * v_col_ortho, dim=-1) * math.sqrt(float(BKP_SUBKEY_DIM))
        u2_new = F.normalize(JET_NORMAL_ALPHA_STAR * u2_base + JET_TANGENT_NORM_STAR * u_row_ortho, dim=-1) * math.sqrt(float(BKP_SUBKEY_DIM))
        u_new_full = torch.stack([u1_new, u2_new], dim=2).reshape(n_cryst, D_BKP)

        v_r_part = row_v_bar[r_sel]
        v_c_part = col_v_bar[c_sel]
        v_r_unit = F.normalize(v_r_part, dim=-1)
        proj_c_on_r = (v_c_part * v_r_unit).sum(dim=-1, keepdim=True) * v_r_unit
        v_c_ortho = v_c_part - proj_c_on_r
        v_comb_dir = F.normalize(0.5 * (v_r_part + proj_c_on_r) + v_c_ortho, dim=-1)
        target_v_norm = 0.5 * (v_r_part.norm(dim=-1, keepdim=True) + v_c_part.norm(dim=-1, keepdim=True)) * float(closure_scale_ratio)
        v_new_full = v_comb_dir * target_v_norm

        s_u_new = (u_new_full.abs().amax(dim=-1) / 127.0).clamp(min=1e-6)
        s_v_new = (v_new_full.abs().amax(dim=-1) / 127.0).clamp(min=1e-6)
        q_u_int8 = torch.round(u_new_full / s_u_new.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        q_v_int8 = torch.round(v_new_full / s_v_new.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)

        self.expert_pages_int8[flat_eids_cpu, 0, :] = q_u_int8
        self.expert_pages_int8[flat_eids_cpu, 1, :] = q_v_int8
        flat_eids_dev = flat_eids_cpu.to(dev)
        self.scales[flat_eids_dev, 0] = s_u_new.to(device=dev, dtype=torch.float16)
        self.scales[flat_eids_dev, 1] = s_v_new.to(device=dev, dtype=torch.float16)
        self.activation_counts[flat_eids_dev] = torch.clamp(self.activation_counts[flat_eids_dev], min=1)
        closure_mask_dev = torch.zeros(self.num_experts, dtype=torch.bool, device=dev)
        closure_mask_dev[flat_eids_dev] = True
        self._closure_expert_mask_dev = closure_mask_dev
        self._sparse_128_triblock_mode = None

        if self._gpu_resident_pages and self._expert_pages_dev is not None:
            self._expert_pages_dev[flat_eids_dev, 0, :] = q_u_int8.to(dev)
            self._expert_pages_dev[flat_eids_dev, 1, :] = q_v_int8.to(dev)
        if self._master_uv_fp32 is not None:
            self._master_uv_fp32.data[flat_eids_dev, 0, :] = u_new_full.to(device=dev, dtype=self._master_uv_fp32.dtype)
            self._master_uv_fp32.data[flat_eids_dev, 1, :] = v_new_full.to(device=dev, dtype=self._master_uv_fp32.dtype)

        u_new_norm = F.normalize(u_new_full, dim=-1)
        mature_eids_cpu = torch.nonzero(mature_cpu.view(-1), as_tuple=False).squeeze(-1)
        u_mat_norm = F.normalize(
            (pages_cpu[mature_eids_cpu, 0, :] * scales_cpu[mature_eids_cpu, 0:1]), dim=-1
        )
        cross_cos = torch.matmul(u_new_norm, u_mat_norm.t()).abs().mean().item()
        return {
            "crystallized_count": n_cryst,
            "mature_before": int(mature_cpu.sum().item()),
            "mature_after": int(mature_cpu.sum().item()) + n_cryst,
            "mean_operator_offdiag_cos": float(cross_cos),
        }

    @torch.no_grad()
    def expand_quadrant_2x(self, perturb_scale: float = SUBSPACE_STD_SCALE) -> None:
        """
        3.4.2 节: 四象限无损倍增扩容 (K_p x K_p -> 2K_p x 2K_p, 对应 (a+b)^2 = a^2 + 2ab + b^2)
        左上象限 E_11 原位保留全部存量微专家；其余三个新增象限 (E_12, E_21, E_22) 的 v_val 严格置 0，
        并通过 codebook_pre_centered 锁死存量子键单位球面坐标，确保扩容瞬间模型前向输出误差严格为 0.0000 (<1e-6)。
        """
        self._sync_device_pages_to_cpu_buffer()
        old_k = self.grid_k
        new_k = old_k * 2
        assert new_k in STANDARD_GRID_SIZES, f"扩容目标 {new_k} 超出标准档位上限"
        new_m = new_k * new_k
        dev = self.u_codebook.device

        # 1. 扩展子键码本 U_new, V_new (球面最近邻测地中点细分 Geodesic Midpoint Mesh Subdivision)
        u_old, v_old = self.normalized_codebooks()
        eye_mask = torch.eye(old_k, dtype=torch.bool, device=dev).unsqueeze(0)
        sim_u = torch.matmul(u_old, u_old.transpose(-1, -2)).masked_fill(eye_mask, -2.0)
        sim_v = torch.matmul(v_old, v_old.transpose(-1, -2)).masked_fill(eye_mask, -2.0)
        nn_u_idx = sim_u.argmax(dim=-1, keepdim=True).expand(-1, -1, BKP_SUBKEY_DIM)
        nn_v_idx = sim_v.argmax(dim=-1, keepdim=True).expand(-1, -1, BKP_SUBKEY_DIM)
        u_nn = torch.gather(u_old, dim=1, index=nn_u_idx)
        v_nn = torch.gather(v_old, dim=1, index=nn_v_idx)
        u_mid = F.normalize(0.5 * (u_old + u_nn), dim=-1)
        v_mid = F.normalize(0.5 * (v_old + v_nn), dim=-1)
        u_perturb = F.normalize(u_mid + perturb_scale * torch.randn_like(u_mid), dim=-1)
        v_perturb = F.normalize(v_mid + perturb_scale * torch.randn_like(v_mid), dim=-1)
        self.codebook_pre_centered.fill_(1)
        self._codebook_pre_centered_cached = 1
        self.u_codebook = nn.Parameter(
            torch.cat([u_old, u_perturb], dim=1), requires_grad=False
        )
        self.v_codebook = nn.Parameter(
            torch.cat([v_old, v_perturb], dim=1), requires_grad=False
        )

        # 2. 构造 2K_p x 2K_p 四象限微专家网格，将旧 K_p x K_p 拷入左上象限 E_11
        old_pages_2d = self.expert_pages_int8.view(old_k, old_k, 2, D_BKP)
        new_pages_2d = torch.zeros((new_k, new_k, 2, D_BKP), dtype=torch.int8)
        new_pages_2d[:, :, 0, :] = torch.randint(-64, 64, (new_k, new_k, D_BKP), dtype=torch.int8)
        new_pages_2d[:old_k, :old_k] = old_pages_2d

        old_scales_2d = self.scales.view(old_k, old_k, 2)
        new_scales_2d = torch.zeros((new_k, new_k, 2), dtype=torch.float16, device=dev)
        new_scales_2d[:, :, 0] = 1.0 / math.sqrt(D_BKP)
        new_scales_2d[:old_k, :old_k] = old_scales_2d

        old_counts_2d = self.activation_counts.view(old_k, old_k)
        new_counts_2d = torch.zeros((new_k, new_k), dtype=self.activation_counts.dtype, device=dev)
        new_counts_2d[:old_k, :old_k] = old_counts_2d

        old_purity_2d = self.purity_r.view(old_k, old_k)
        new_purity_2d = torch.ones((new_k, new_k), dtype=torch.float32, device=dev)
        new_purity_2d[:old_k, :old_k] = old_purity_2d

        self.grid_k = new_k
        self.num_experts = new_m
        self.expert_pages_int8 = new_pages_2d.reshape(new_m, 2, D_BKP)
        self.scales = new_scales_2d.reshape(new_m, 2)
        self.activation_counts = new_counts_2d.reshape(new_m)
        self.purity_r = new_purity_2d.reshape(new_m)
        self._rebuild_runtime_buffers_after_topology_change()

    @torch.no_grad()
    def permute_rows_cols(self, row_perm: torch.Tensor, col_perm: torch.Tensor) -> None:
        """
        笛卡尔乘积网格的行列无损置换 (前向输出误差严格为 0.0000):
        用于谱双聚类碎片归拢 (Spectral Co-Clustering Defragmentation)。
        """
        self._sync_device_pages_to_cpu_buffer()
        k = self.grid_k
        r_cpu, c_cpu = row_perm.cpu().long(), col_perm.cpu().long()
        r_dev, c_dev = row_perm.to(self.u_codebook.device).long(), col_perm.to(self.u_codebook.device).long()

        u_norm, v_norm = self.normalized_codebooks()
        self.codebook_pre_centered.fill_(1)
        self._codebook_pre_centered_cached = 1
        self.u_codebook.data = u_norm[:, r_dev, :].contiguous()
        self.v_codebook.data = v_norm[:, c_dev, :].contiguous()

        if self._mmap_obj is not None and self.mount_mode in (BKPMountMode.RO_SYS, BKPMountMode.RO_LIB):
            self.expert_pages_int8 = self.expert_pages_int8.clone()
        pages_2d = self.expert_pages_int8.view(k, k, 2, D_BKP)[r_cpu][:, c_cpu].contiguous()
        self.expert_pages_int8.copy_(pages_2d.reshape(-1, 2, D_BKP))

        self.scales.copy_(self.scales.view(k, k, 2)[r_dev][:, c_dev].contiguous().reshape(-1, 2))
        self.activation_counts.copy_(self.activation_counts.view(k, k)[r_dev][:, c_dev].contiguous().reshape(-1))
        self.purity_r.copy_(self.purity_r.view(k, k)[r_dev][:, c_dev].contiguous().reshape(-1))
        self._rebuild_runtime_buffers_after_topology_change()

    @torch.no_grad()
    def extract_padded_subpackage(self, sub_id: str, row_idx: torch.Tensor, col_idx: torch.Tensor) -> "BKPPackage":
        """
        3.4.3 节 BKP Split 核心算子:
        从当前网格中切出 (len(row_idx) x len(col_idx)) 对角子块，并向上补齐至最近的 2^m 标准档位。
        补齐新增槽位由子块内高冲突子键施以正交微扰生成新键，且其 v_val 严格初始化为 0，
        保证分裂补齐瞬间前向输出误差严格为 0.0000。
        """
        self._sync_device_pages_to_cpu_buffer()
        sub_r, sub_c = int(row_idx.numel()), int(col_idx.numel())
        target_k = round_up_to_standard_grid(max(sub_r, sub_c))
        dev = self.u_codebook.device

        sub_pkg = BKPPackage(
            pkg_id=sub_id, grid_k=target_k, mount_mode=self.mount_mode, is_base_pkg=self.is_base_pkg, device=dev
        )
        sub_pkg._copy_manifold_stats_from(self)
        sub_pkg.codebook_pre_centered.fill_(1)
        sub_pkg._codebook_pre_centered_cached = 1
        r_dev, c_dev = row_idx.to(dev).long(), col_idx.to(dev).long()
        r_cpu, c_cpu = row_idx.cpu().long(), col_idx.cpu().long()

        # 1. 拷贝已固化单位子键，并用子块内子键的正交微扰填充向上补齐的空白行列键
        u_norm, v_norm = self.normalized_codebooks()
        u_sub = u_norm.data[:, r_dev, :]
        v_sub = v_norm.data[:, c_dev, :]
        sub_pkg.u_codebook.data[:, :sub_r, :] = u_sub
        sub_pkg.v_codebook.data[:, :sub_c, :] = v_sub

        if target_k > sub_r:
            pad_r = target_k - sub_r
            src_r = torch.arange(pad_r, device=dev) % sub_r
            sub_pkg.u_codebook.data[:, sub_r:, :] = F.normalize(
                u_sub[:, src_r, :] + 0.05 * torch.randn(BKP_NUM_HEADS, pad_r, BKP_SUBKEY_DIM, device=dev), dim=-1
            )
        if target_k > sub_c:
            pad_c = target_k - sub_c
            src_c = torch.arange(pad_c, device=dev) % sub_c
            sub_pkg.v_codebook.data[:, sub_c:, :] = F.normalize(
                v_sub[:, src_c, :] + 0.05 * torch.randn(BKP_NUM_HEADS, pad_c, BKP_SUBKEY_DIM, device=dev), dim=-1
            )

        # 2. 拷贝对角块微专家权重，补齐区保持 v_val = 0 (scale_v = 0)
        k = self.grid_k
        sub_pages = self.expert_pages_int8.view(k, k, 2, D_BKP)[r_cpu][:, c_cpu]
        dst_pages_2d = sub_pkg.expert_pages_int8.view(target_k, target_k, 2, D_BKP)
        dst_pages_2d[:sub_r, :sub_c] = sub_pages

        dst_scales_2d = sub_pkg.scales.view(target_k, target_k, 2)
        dst_scales_2d[:, :, 1] = 0.0
        dst_scales_2d[:sub_r, :sub_c] = self.scales.view(k, k, 2)[r_dev][:, c_dev]

        dst_counts_2d = sub_pkg.activation_counts.view(target_k, target_k)
        dst_counts_2d[:sub_r, :sub_c] = self.activation_counts.view(k, k)[r_dev][:, c_dev]

        dst_purity_2d = sub_pkg.purity_r.view(target_k, target_k)
        dst_purity_2d[:sub_r, :sub_c] = self.purity_r.view(k, k)[r_dev][:, c_dev]

        sub_pkg._rebuild_runtime_buffers_after_topology_change()
        return sub_pkg

    @overload
    def bkp_split_spectral(
        self,
        cross_ratio_threshold: float = 0.05,
        return_bridge: Literal[False] = False,
    ) -> Optional[Tuple["BKPPackage", "BKPPackage"]]: ...

    @overload
    def bkp_split_spectral(
        self,
        cross_ratio_threshold: float = 0.05,
        return_bridge: Literal[True] = ...,
    ) -> Optional[Tuple["BKPPackage", "BKPPackage", Optional["BKPPackage"]]]: ...

    @torch.no_grad()
    def bkp_split_spectral(
        self,
        cross_ratio_threshold: float = 0.05,
        return_bridge: bool = False,
    ) -> Optional[Union[Tuple["BKPPackage", "BKPPackage"], Tuple["BKPPackage", "BKPPackage", Optional["BKPPackage"]]]]:
        """
        3.4.3 节 BKP Split (二分图拉普拉斯谱双聚类重排与无损桥接裂变):
        若非对角交叉激活占比 <= cross_ratio_threshold，则裂变为 (pkg_a, pkg_b)；
        当 return_bridge=True 且存在非对角活跃微专家时，将 2ab 跨簇突触无损提取为 pkg_bridge。
        """
        k = self.grid_k
        c_mat = self.activation_counts.view(k, k).float()
        total_act = c_mat.sum().item()
        if total_act <= 0:
            return None

        if True:
            # 仅提取拥有实质激活的行与列子图 (剥离全零空白死区行列，杜绝零度孤立点干扰拉普拉斯谱分解与虚增子包尺寸)
            act_r_idx = torch.nonzero(c_mat.sum(dim=1) > 0, as_tuple=False).squeeze(-1)
            act_c_idx = torch.nonzero(c_mat.sum(dim=0) > 0, as_tuple=False).squeeze(-1)
            r_act_n = int(act_r_idx.numel())
            c_act_n = int(act_c_idx.numel())
            if r_act_n < 4 or c_act_n < 4:
                return None

            inact_r_idx = torch.nonzero(c_mat.sum(dim=1) <= 0, as_tuple=False).squeeze(-1)
            inact_c_idx = torch.nonzero(c_mat.sum(dim=0) <= 0, as_tuple=False).squeeze(-1)

            c_sub = c_mat[act_r_idx][:, act_c_idx]
            deg_r = c_sub.sum(dim=1, keepdim=True).clamp(min=1e-6)
            deg_c = c_sub.sum(dim=0, keepdim=True).clamp(min=1e-6)
            d_r, d_c = deg_r.rsqrt(), deg_c.rsqrt()
            norm_c = d_r * c_sub * d_c

            p_r = F.normalize(deg_r.sqrt(), p=2, dim=0)
            p_c = F.normalize(deg_c.sqrt(), p=2, dim=1).transpose(0, 1)
            proj_c = norm_c @ p_c
            proj_r = p_r.transpose(0, 1) @ norm_c
            scalar_rc = p_r.transpose(0, 1) @ proj_c
            norm_c_perp = norm_c - (proj_c @ p_c.transpose(0, 1)) - (p_r @ proj_r) + scalar_rc * (p_r @ p_c.transpose(0, 1))

            u_svd, _, v_svd_t = torch.linalg.svd(norm_c_perp, full_matrices=False)
            f_r = d_r.squeeze(1) * u_svd[:, 0]
            f_c = d_c.squeeze(0) * v_svd_t[0, :]
            sub_r_perm = torch.argsort(f_r)
            sub_c_perm = torch.argsort(f_c)

            c_cand = c_sub[sub_r_perm][:, sub_c_perm]
            mid_r, mid_c = r_act_n // 2, c_act_n // 2
            if (c_cand[:mid_r, mid_c:].sum() + c_cand[mid_r:, :mid_c].sum()) > (
                c_cand[:mid_r, :mid_c].sum() + c_cand[mid_r:, mid_c:].sum()
            ):
                sub_c_perm = sub_c_perm.flip(0)
                f_c = -f_c
                c_cand = c_sub[sub_r_perm][:, sub_c_perm]

            # 非对称谱双聚类切分 (Asymmetric Cheeger / Conductance Minimum Sweep):
            # 在剥离空白死区后的活跃子图 (r_act_n x c_act_n) 上扫描真实非对称切分点 (sr, sc)
            prefix = torch.zeros((r_act_n + 1, c_act_n + 1), dtype=torch.float32, device=c_mat.device)
            prefix[1:, 1:] = c_cand.cumsum(dim=0).cumsum(dim=1)
            min_r_m = max(2, r_act_n // (2 * BKP_NUM_HEADS))
            min_c_m = max(2, c_act_n // (2 * BKP_NUM_HEADS))
            min_cluster_act = (1.0 / float(STANDARD_GRID_SIZES[0])) * total_act

            r_zero = int((f_r[sub_r_perm] < 0).sum().item())
            c_zero = int((f_c[sub_c_perm] < 0).sum().item())
            step_r = max(1, r_act_n // 32)
            step_c = max(1, c_act_n // 32)
            r_cands = sorted(set(
                list(range(min_r_m, r_act_n - min_r_m + 1, step_r))
                + [max(min_r_m, min(r_act_n - min_r_m, r_zero)), r_act_n // 2]
            ))
            c_cands = sorted(set(
                list(range(min_c_m, c_act_n - min_c_m + 1, step_c))
                + [max(min_c_m, min(c_act_n - min_c_m, c_zero)), c_act_n // 2]
            ))

            best_score = float("inf")
            best_cross_ratio = 1.0
            best_sr, best_sc = r_act_n // 2, c_act_n // 2
            for sr in r_cands:
                for sc in c_cands:
                    act_aa = float(prefix[sr, sc].item())
                    act_bb = float((prefix[r_act_n, c_act_n] - prefix[sr, c_act_n] - prefix[r_act_n, sc] + prefix[sr, sc]).item())
                    if act_aa < min_cluster_act or act_bb < min_cluster_act:
                        continue
                    diag_cur = act_aa + act_bb
                    cross_cur = max(0.0, float(total_act - diag_cur))
                    cr_cur = cross_cur / total_act
                    ncut = cross_cur * (1.0 / act_aa + 1.0 / act_bb)
                    if ncut < best_score:
                        best_score = ncut
                        best_cross_ratio = cr_cur
                        best_sr, best_sc = sr, sc

            split_r, split_c = best_sr, best_sc
            end_r, end_c = r_act_n, c_act_n
            q_r = max(1, min(split_r, end_r - split_r) // 2)
            q_c = max(1, min(split_c, end_c - split_c) // 2)
            cross_ratio = best_cross_ratio
            if cross_ratio > cross_ratio_threshold:
                return None

            ord_act_r = act_r_idx[sub_r_perm]
            ord_act_c = act_c_idx[sub_c_perm]
            r_a, r_b = ord_act_r[:split_r], ord_act_r[split_r:]
            c_a, c_b = ord_act_c[:split_c], ord_act_c[split_c:]
            ra_cross = c_mat[r_a][:, c_b].sum(dim=1)
            rb_cross = c_mat[r_b][:, c_a].sum(dim=1)
            ca_cross = c_mat[r_b][:, c_a].sum(dim=0)
            cb_cross = c_mat[r_a][:, c_b].sum(dim=0)
            if float((ra_cross.sum() + rb_cross.sum()).item()) > 0.0:
                ord_act_r = torch.cat([r_a[torch.argsort(ra_cross, descending=True)], r_b[torch.argsort(rb_cross, descending=True)]], dim=0)
                ord_act_c = torch.cat([c_a[torch.argsort(ca_cross, descending=True)], c_b[torch.argsort(cb_cross, descending=True)]], dim=0)

            row_perm = torch.cat([ord_act_r, inact_r_idx], dim=0) if inact_r_idx.numel() > 0 else ord_act_r
            col_perm = torch.cat([ord_act_c, inact_c_idx], dim=0) if inact_c_idx.numel() > 0 else ord_act_c
            self.permute_rows_cols(row_perm, col_perm)

        end_r = locals().get("end_r", k)
        end_c = locals().get("end_c", k)
        pkg_a = self.extract_padded_subpackage(f"{self.pkg_id}_partA", torch.arange(0, split_r), torch.arange(0, split_c))
        pkg_b = self.extract_padded_subpackage(f"{self.pkg_id}_partB", torch.arange(split_r, end_r), torch.arange(split_c, end_c))

        if not return_bridge:
            if cross_ratio > 1e-5:
                scales_2d = self.scales.view(k, k, 2)
                scales_2d[:split_r, split_c:, 1] = 0.0
                scales_2d[split_r:, :split_c, 1] = 0.0
                counts_2d = self.activation_counts.view(k, k)
                counts_2d[:split_r, split_c:] = 0
                counts_2d[split_r:, :split_c] = 0
                self._rebuild_runtime_buffers_after_topology_change()
            return pkg_a, pkg_b

        woken_2d = (self.activation_counts.view(k, k) > 0) & (self.scales[:, 1].view(k, k) > 2e-5)
        n_cross_woken = int((woken_2d[:split_r, split_c:].sum() + woken_2d[split_r:, :split_c].sum()).item())
        pkg_bridge: Optional["BKPPackage"] = None
        if n_cross_woken > 0 and k >= 128:
            br_rows = torch.cat([torch.arange(0, q_r), torch.arange(split_r, split_r + q_r)], dim=0)
            br_cols = torch.cat([torch.arange(0, q_c), torch.arange(split_c, split_c + q_c)], dim=0)
            pkg_bridge = self.extract_padded_subpackage(f"{self.pkg_id}_bridge", br_rows, br_cols)
            pkg_bridge.is_bridge_pkg = True
            pkg_bridge.is_base_pkg = False
            br_k = pkg_bridge.grid_k
            br_scales_2d = pkg_bridge.scales.view(br_k, br_k, 2)
            br_scales_2d[:q_r, :q_c, 1] = 0.0
            br_scales_2d[q_r:, q_c:, 1] = 0.0
            br_counts_2d = pkg_bridge.activation_counts.view(br_k, br_k)
            br_counts_2d[:q_r, :q_c] = 0
            br_counts_2d[q_r:, q_c:] = 0
            pkg_bridge._rebuild_runtime_buffers_after_topology_change()

        return pkg_a, pkg_b, pkg_bridge

    @staticmethod
    @torch.no_grad()
    def bkp_merge_direct_sum(pkg_a: "BKPPackage", pkg_b: "BKPPackage", merged_id: str) -> "BKPPackage":
        """
        3.4.3 节 BKP Merge (主对角线直和拼接 + 2ab 交叉耦合区开辟 + 2^m 向上补齐):
        将 pkg_a (K_A x K_A) 与 pkg_b (K_B x K_B) 沿主对角线拼接，原位辟出非对角 2ab 交叉象限，
        并将总边长 K_A + K_B 向上补齐至最近的 2^m 标准协议档位，新增与交叉区 v_val 严格置 0 (scale_v=0)。
        """
        pkg_a._sync_device_pages_to_cpu_buffer()
        pkg_b._sync_device_pages_to_cpu_buffer()
        ka, kb = pkg_a.grid_k, pkg_b.grid_k
        target_k = round_up_to_standard_grid(ka + kb)
        dev = pkg_a.u_codebook.device

        merged = BKPPackage(
            pkg_id=merged_id,
            grid_k=target_k,
            mount_mode=pkg_a.mount_mode,
            is_base_pkg=bool(pkg_a.is_base_pkg or pkg_b.is_base_pkg),
            device=dev,
        )
        merged._copy_manifold_stats_from(pkg_a)
        merged.codebook_pre_centered.fill_(1)
        merged._codebook_pre_centered_cached = 1
        ua, va = pkg_a.normalized_codebooks()
        ub, vb = pkg_b.normalized_codebooks()

        merged.u_codebook.data[:, :ka, :] = ua
        merged.u_codebook.data[:, ka:ka + kb, :] = ub
        merged.v_codebook.data[:, :ka, :] = va
        merged.v_codebook.data[:, ka:ka + kb, :] = vb

        if target_k > ka + kb:
            pad = target_k - (ka + kb)
            src_u = merged.u_codebook.data[:, :ka + kb, :]
            src_v = merged.v_codebook.data[:, :ka + kb, :]
            idx = torch.arange(pad, device=dev) % (ka + kb)
            merged.u_codebook.data[:, ka + kb:, :] = F.normalize(
                src_u[:, idx, :] + SUBSPACE_STD_SCALE * torch.randn(BKP_NUM_HEADS, pad, BKP_SUBKEY_DIM, device=dev), dim=-1
            )
            merged.v_codebook.data[:, ka + kb:, :] = F.normalize(
                src_v[:, idx, :] + SUBSPACE_STD_SCALE * torch.randn(BKP_NUM_HEADS, pad, BKP_SUBKEY_DIM, device=dev), dim=-1
            )

        # 主对角线填入 E_A 与 E_B，非对角 2ab 象限与补齐区保持 v_val = 0 (scale_v = 0)
        m_pages = merged.expert_pages_int8.view(target_k, target_k, 2, D_BKP)
        m_pages[:ka, :ka] = pkg_a.expert_pages_int8.view(ka, ka, 2, D_BKP)
        m_pages[ka:ka + kb, ka:ka + kb] = pkg_b.expert_pages_int8.view(kb, kb, 2, D_BKP)

        m_scales = merged.scales.view(target_k, target_k, 2)
        m_scales[:, :, 1] = 0.0
        m_scales[:ka, :ka] = pkg_a.scales.view(ka, ka, 2)
        m_scales[ka:ka + kb, ka:ka + kb] = pkg_b.scales.view(kb, kb, 2)

        m_counts = merged.activation_counts.view(target_k, target_k)
        m_counts[:ka, :ka] = pkg_a.activation_counts.view(ka, ka)
        m_counts[ka:ka + kb, ka:ka + kb] = pkg_b.activation_counts.view(kb, kb)

        m_purity = merged.purity_r.view(target_k, target_k)
        m_purity[:ka, :ka] = pkg_a.purity_r.view(ka, ka)
        m_purity[ka:ka + kb, ka:ka + kb] = pkg_b.purity_r.view(kb, kb)

        merged._rebuild_runtime_buffers_after_topology_change()
        return merged

    @torch.no_grad()
    def nucleate_tangential_conjugate_mitosis(
        self,
        max_mitosis_slots: int = 256,
    ) -> Dict[str, Union[int, float]]:
        """
        切向复共轭叶状分裂算子 (Tangential Conjugate Mitosis, Insight 7A & 7B):
        当微专家满足零超参数瑞利多义干涉判据 (H_m * R_m^2 > ln(N_grid) 且 R_m < 1/sqrt(2)) 时，
        绝不通过原地衰减 dv_perp 丢弃正交训练细节，而是将多义母微专家 m = (r, c) 沿切向复共轭分裂到同行的空白伴随格点 m' = (r, c'):
          1. 母微专家保留正向切向偏导 u_m = alpha^* c_m + u_{m,perp}；
          2. 伴随子微专家初始化为复共轭切向对偶态 u_{m'} = alpha^* c_{m'} - u_{m,perp}，
             并承载正交奇宇称载荷分量，从而在切丛上无损分离正反两个子概念分支！
        """
        self._sync_device_pages_to_cpu_buffer()
        k = self.grid_k
        dev = self.scales.device
        poly_mask_1d = self.detect_polysemy_interference_mask()
        sv_1d = self.scales[:, 1].float()
        empty_2d = (sv_1d <= 2e-5).view(k, k)
        poly_eids = torch.nonzero(poly_mask_1d & (sv_1d > 2e-5), as_tuple=False).squeeze(-1)
        if poly_eids.numel() == 0 or int(empty_2d.sum().item()) == 0:
            return {"mitosis_count": 0, "mean_conjugate_cos": 0.0}

        if poly_eids.numel() > max_mitosis_slots:
            scores = self.activation_counts[poly_eids].float() * (1.0 - self.purity_r[poly_eids].float())
            _, top_idx = torch.topk(scores, k=int(max_mitosis_slots), largest=True)
            poly_eids = poly_eids[top_idx]

        u_cb_norm, v_cb_norm = self.normalized_codebooks()
        pages_cpu = self.expert_pages_int8.float()
        scales_cpu = self.scales.float().cpu()

        parent_list = []
        child_list = []
        used_empty = torch.zeros((k, k), dtype=torch.bool, device=dev)
        for eid_t in poly_eids:
            eid = int(eid_t.item())
            r_i = eid // k
            avail_cols = torch.nonzero(empty_2d[r_i] & (~used_empty[r_i]), as_tuple=False).squeeze(-1)
            if avail_cols.numel() == 0:
                continue
            c_child = int(avail_cols[0].item())
            used_empty[r_i, c_child] = True
            parent_list.append(eid)
            child_list.append(r_i * k + c_child)

        if len(parent_list) == 0:
            return {"mitosis_count": 0, "mean_conjugate_cos": 0.0}

        p_eids_cpu = torch.tensor(parent_list, dtype=torch.long)
        ch_eids_cpu = torch.tensor(child_list, dtype=torch.long)
        p_eids_dev = p_eids_cpu.to(dev)
        ch_eids_dev = ch_eids_cpu.to(dev)

        u_parent = (pages_cpu[p_eids_cpu, 0, :] * scales_cpu[p_eids_cpu, 0:1]).to(dev)
        v_parent = (pages_cpu[p_eids_cpu, 1, :] * scales_cpu[p_eids_cpu, 1:2]).to(dev)

        r_p = torch.div(p_eids_dev, k, rounding_mode="floor")
        c_p = p_eids_dev % k
        c_ch = ch_eids_dev % k

        c_anchor_p = torch.stack(
            [u_cb_norm[:, r_p, :].permute(1, 0, 2).float(), v_cb_norm[:, c_p, :].permute(1, 0, 2).float()], dim=2
        ).reshape(-1, D_BKP) * math.sqrt(float(BKP_SUBKEY_DIM))
        c_anchor_ch = torch.stack(
            [u_cb_norm[:, r_p, :].permute(1, 0, 2).float(), v_cb_norm[:, c_ch, :].permute(1, 0, 2).float()], dim=2
        ).reshape(-1, D_BKP) * math.sqrt(float(BKP_SUBKEY_DIM))

        alpha_p = (u_parent * c_anchor_p).sum(dim=-1, keepdim=True) / float(D_BKP)
        u_perp = u_parent - alpha_p * c_anchor_p
        u_perp_unit = F.normalize(u_perp, dim=-1) * (JET_TANGENT_NORM_STAR * math.sqrt(float(D_BKP)))

        u_parent_new = F.normalize(JET_NORMAL_ALPHA_STAR * c_anchor_p + u_perp_unit, dim=-1) * math.sqrt(float(D_BKP))
        u_child_new = F.normalize(JET_NORMAL_ALPHA_STAR * c_anchor_ch - u_perp_unit, dim=-1) * math.sqrt(float(D_BKP))

        v_p_unit = F.normalize(v_parent, dim=-1)
        v_odd_dir = F.normalize(u_perp_unit - (u_perp_unit * v_p_unit).sum(dim=-1, keepdim=True) * v_p_unit, dim=-1)
        v_odd = v_odd_dir * (0.25 * v_parent.norm(p=2, dim=-1, keepdim=True))
        v_parent_new = v_parent + v_odd
        v_child_new = v_parent - v_odd

        self.write_experts_to_mmap_pages(p_eids_dev, u_parent_new, v_parent_new)
        self.write_experts_to_mmap_pages(ch_eids_dev, u_child_new, v_child_new)
        self.purity_r[p_eids_dev] = SIMPLEX_2_GEODESIC_PROJ
        self.purity_r[ch_eids_dev] = SIMPLEX_2_GEODESIC_PROJ
        self.activation_counts[ch_eids_dev] = torch.clamp(self.activation_counts[p_eids_dev] // 2, min=1)
        self._sparse_128_triblock_mode = None

        conj_cos = F.cosine_similarity(u_perp_unit, -u_perp_unit, dim=-1).mean().item()
        return {"mitosis_count": int(p_eids_dev.numel()), "mean_conjugate_cos": float(conj_cos)}

    @staticmethod
    @torch.no_grad()
    def bkp_merge_triadic_bundle(pkg_a: "BKPPackage", pkg_b: "BKPPackage", merged_id: str) -> "BKPPackage":
        """
        同领域双包三元正交流形合成算子 ((A ∩ B) ⊕ (A \\ B) ⊕ (B \\ A), Insight 7A / 7B / 7C):
        合并两个独立训练的同领域知识包 (如 Literature_A.bkp 与 Literature_B.bkp):
          1. 利用直和网格开辟完整容量，零丢弃保留两包各自独有的微专家分支 (A \\ B 与 B \\ A)；
          2. 对两包同源对应的微专家对 (v_A, v_B)，执行偶/奇宇称三元正交分解:
             - 偶宇称共性核 v_even = 0.5 * (v_A + v_B) 锚定交集共性 (A ∩ B)；
             - 奇宇称差值轴 ±v_odd = ±0.5 * (v_A - v_B) 挂载在复共轭切向键 (±u_perp) 上，
               确保合并包在 A 域、B 域与交集域上同时无损保持全部几何细节！
        """
        merged = BKPPackage.bkp_merge_direct_sum(pkg_a, pkg_b, merged_id=merged_id)
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
    def compute_helmholtz_hodge_spectrum(self) -> Dict[str, float]:
        """
        切丛光滑矢量场的亥姆霍兹-霍奇谱分解算子 (Helmholtz-Hodge Decomposition, Insight 8A):
        单个微专家 m 的局部雅可比算子 J_m = v_m u_m^T 正交分解为:
          - 散度 (Divergence): div V_m = Tr(J_m) = <u_m, v_m>
          - 旋度 2-形式 (Curl / Vorticity): Omega_m = 0.5 * (v_m u_m^T - u_m v_m^T) = 0.5 * (u_m ^ v_m)
        定义归一化散度能量比 eta_div = cos^2(u_m, v_m) 与旋度能量比 eta_curl = sin^2(u_m, v_m) = 1 - eta_div:
          - 指令微专家 (Instruction Operators): u_m 与 v_m 趋向正交 (高旋度无散流 eta_curl -> 1, 负责状态机旋转推进);
          - 数据微专家 (Data Fact Wells): u_m 与 v_m 具备显著共线耦合 (高散度无旋势阱, 负责静态事实吸引/注入).
        """
        self._sync_device_pages_to_cpu_buffer()
        active_mask = (self.scales[:, 0] > 0) & (self.scales[:, 1] > 0)
        active_eids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        if active_eids.numel() == 0:
            return {
                "active_experts": 0,
                "mean_eta_div": 0.0,
                "mean_eta_curl": 1.0,
                "curl_dominant_Instruction_ratio": 0.0,
                "div_enriched_Data_ratio": 0.0,
            }
        eids_cpu = active_eids.cpu()
        u_mat = self.expert_pages_int8[eids_cpu, 0, :].float() * self.scales[active_eids, 0:1].cpu().float()
        v_mat = self.expert_pages_int8[eids_cpu, 1, :].float() * self.scales[active_eids, 1:2].cpu().float()

        u_norm = u_mat.norm(dim=-1).clamp_min(1e-8)
        v_norm = v_mat.norm(dim=-1).clamp_min(1e-8)
        div_val = (u_mat * v_mat).sum(dim=-1)
        cos_uv = (div_val / (u_norm * v_norm)).clamp(-1.0, 1.0)
        eta_div = cos_uv.pow(2)
        eta_curl = 1.0 - eta_div

        thresh_data_div = 4.0 / float(D_BKP)
        div_enriched_ratio = (eta_div > thresh_data_div).float().mean().item()
        curl_dominant_ratio = (eta_curl >= (1.0 - 1.0 / float(D_BKP))).float().mean().item()

        return {
            "active_experts": int(active_eids.numel()),
            "mean_eta_div": float(eta_div.mean().item()),
            "mean_eta_curl": float(eta_curl.mean().item()),
            "max_eta_div": float(eta_div.max().item()),
            "curl_dominant_Instruction_ratio": float(curl_dominant_ratio),
            "div_enriched_Data_ratio": float(div_enriched_ratio),
        }

    @staticmethod
    @torch.no_grad()
    def compute_non_abelian_lie_commutator(
        u1: torch.Tensor, v1: torch.Tensor, u2: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8
    ) -> Dict[str, torch.Tensor]:
        """
        旋度场的非交换李括号与因果不可逆性闭式算子 (Non-Abelian Lie Bracket, Insight 8B):
        对两个局部秩-1 雅可比算子 J_1 = v_1 u_1^T, J_2 = v_2 u_2^T，其李括号为:
          [J_1, J_2] = J_1 J_2 - J_2 J_1 = c_{2->1} v_1 u_2^T - c_{1->2} v_2 u_1^T
        其中因果传递耦合系数 c_{1->2} = <u_2, v_1>, c_{2->1} = <u_1, v_2>.
        """
        c_1_to_2 = (u2 * v1).sum(dim=-1)
        c_2_to_1 = (u1 * v2).sum(dim=-1)

        u1_sq = (u1 * u1).sum(dim=-1)
        v1_sq = (v1 * v1).sum(dim=-1)
        u2_sq = (u2 * u2).sum(dim=-1)
        v2_sq = (v2 * v2).sum(dim=-1)
        u1_u2 = (u1 * u2).sum(dim=-1)
        v1_v2 = (v1 * v2).sum(dim=-1)

        comm_norm_sq = (
            c_2_to_1.pow(2) * v1_sq * u2_sq
            + c_1_to_2.pow(2) * v2_sq * u1_sq
            - 2.0 * c_1_to_2 * c_2_to_1 * v1_v2 * u1_u2
        ).clamp_min(0.0)
        comm_norm = comm_norm_sq.sqrt()
        j1_norm = (u1_sq * v1_sq).sqrt()
        j2_norm = (u2_sq * v2_sq).sqrt()

        i_non_abelian = comm_norm / (j1_norm * j2_norm + eps)
        causal_asymmetry = c_1_to_2.abs() / (c_2_to_1.abs() + eps)
        return {
            "c_1_to_2": c_1_to_2,
            "c_2_to_1": c_2_to_1,
            "commutator_fro_norm": comm_norm,
            "non_abelian_index": i_non_abelian,
            "causal_asymmetry_ratio": causal_asymmetry,
        }

    @torch.no_grad()
    def compile_cot_trajectory_shortcut(
        self,
        q_traj: torch.Tensor,
        delta_traj: torch.Tensor,
        transport_weights: Optional[torch.Tensor] = None,
        target_eid: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        斯托克斯定理与切丛平行移动线积分：多步 CoT 闭式编译单步捷径微专家 (Stokes CoT Shortcut Compiler, Insight 8C):
        给定 T 步思维链轨迹 q_traj: [T, D_BKP] 与沿途微专家注入增量 delta_traj: [T, D_BKP]:
          1. 利用超球面 Levi-Civita 测地线平行移动算子 Gamma_{q_t -> q_0} 将每一步切向量拉回起点切空间 T_{q_0} S^{d-1}:
               Gamma_{q_t -> q_0}(v_t) = v_t - (<q_0, v_t> / (d + <q_t, q_0>)) * (q_t + q_0)
          2. 闭式计算斯托克斯拉回线积分:
               I_total = sum_{t=0}^{T-1} gamma_transport^{(t)} * Gamma_{q_t -> q_0}(delta_traj[t])
          3. 构造起点触发键 u_shortcut = alpha^* c_0 + u_perp，并反解一步直达写入载荷:
               v_shortcut = I_total / (w_0 * phi(<q_0, u_shortcut> / d))
        """
        assert q_traj.dim() == 2 and delta_traj.dim() == 2 and q_traj.shape == delta_traj.shape
        t_steps, d_dim = q_traj.shape
        dev = self.scales.device
        q_traj = q_traj.to(dev).float()
        delta_traj = delta_traj.to(dev).float()

        q0 = F.normalize(q_traj[0], dim=-1) * math.sqrt(float(d_dim))
        if t_steps > 1:
            tangent_dir = q_traj[1] - ((q_traj[1] * q0).sum() / float(d_dim)) * q0
        else:
            tangent_dir = delta_traj[0] - ((delta_traj[0] * q0).sum() / float(d_dim)) * q0
        u_perp = F.normalize(tangent_dir, dim=-1) * (JET_TANGENT_NORM_STAR * math.sqrt(float(d_dim)))
        u_shortcut = JET_NORMAL_ALPHA_STAR * q0 + u_perp

        if transport_weights is None:
            gamma_t = torch.ones((t_steps, 1), device=dev, dtype=torch.float32)
        else:
            gamma_t = transport_weights.to(dev).float().view(t_steps, 1)

        qt_normed = F.normalize(q_traj, dim=-1) * math.sqrt(float(d_dim))
        dot_qt_q0 = (qt_normed * q0.unsqueeze(0)).sum(dim=-1, keepdim=True)
        dot_q0_vt = (delta_traj * q0.unsqueeze(0)).sum(dim=-1, keepdim=True)
        denom = (float(d_dim) + dot_qt_q0).clamp_min(1e-4)
        pulled_back_deltas = delta_traj - (dot_q0_vt / denom) * (qt_normed + q0.unsqueeze(0))

        line_integral_vec = (gamma_t * pulled_back_deltas).sum(dim=0)

        sc0 = (q0 * q0).sum() / float(d_dim)
        phi_0 = F.silu(sc0).clamp_min(1e-4)
        v_shortcut = line_integral_vec / phi_0

        if target_eid is None:
            empty_mask = (self.scales[:, 0] == 0) & (self.scales[:, 1] == 0)
            empty_eids = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1)
            if empty_eids.numel() > 0:
                target_eid = int(empty_eids[0].item())
            else:
                target_eid = int(torch.argmin(self.activation_counts).item())

        eid_tensor = torch.tensor([target_eid], device=dev, dtype=torch.long)
        self.write_experts_to_mmap_pages(eid_tensor, u_shortcut.unsqueeze(0), v_shortcut.unsqueeze(0))
        self.activation_counts[eid_tensor] = max(int(self.activation_counts[eid_tensor].item()), t_steps)
        self.purity_r[eid_tensor] = 1.0
        self._sparse_128_triblock_mode = None

        u_quant, v_quant = self.read_experts_from_mmap_pages(eid_tensor)
        shortcut_out = phi_0 * v_quant.squeeze(0)
        cos_sim = F.cosine_similarity(shortcut_out.unsqueeze(0), line_integral_vec.unsqueeze(0)).item()

        return {
            "shortcut_eid": int(target_eid),
            "compressed_steps": int(t_steps),
            "step_compression_ratio": float(t_steps),
            "u_shortcut": u_shortcut,
            "v_shortcut": v_shortcut,
            "line_integral_vec": line_integral_vec,
            "shortcut_output_cos": float(cos_sim),
        }


# ==============================================================================
# 3. GPU 显存只读微专家 LRU 缓存池 (512MB VRAM Read-Only Expert Cache)
# ==============================================================================
class VRAMReadOnlyExpertCache(nn.Module):
    """
    GPU 显存 512MB 纯只读微专家缓存 (131,072 个 4KB 页槽位):
    - 采用全向量化 GPU 页表直接寻址 (零 Python 循环，零 .tolist() 同步开销)。
    - 命中 (Hit): 直接从显存槽位读取，零 PCIe 传输。
    - 未命中 (Miss): 从 System RAM mmap 页批量拉取缺失的 4KB 页并直接覆盖最冷 LRU 槽位。
    """
    def __init__(self, max_slots: int = 131072, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.max_slots = max_slots
        self.register_buffer(
            "cache_pages_int8", torch.zeros((max_slots, 2, D_BKP), dtype=torch.int8, device=device), persistent=False
        )
        self.register_buffer(
            "cache_scales_fp16", torch.zeros((max_slots, 2), dtype=torch.float16, device=device), persistent=False
        )
        self._slot_last_used = torch.zeros(max_slots, dtype=torch.int64, device=device)
        self._slot_owner_pkg = torch.full((max_slots,), -1, dtype=torch.int32, device=device)
        self._slot_owner_eid = torch.full((max_slots,), -1, dtype=torch.long, device=device)
        self._pkg_name_to_id: Dict[str, int] = {}
        self._id_to_pkg_table: Dict[int, torch.Tensor] = {}
        self._clock: int = 0
        self.total_queries: int = 0
        self.total_hits: int = 0

    def _get_pkg_table(self, pkg: BKPPackage, dev: torch.device) -> Tuple[int, torch.Tensor]:
        pid = self._pkg_name_to_id.get(pkg.pkg_id)
        if pid is None:
            pid = len(self._pkg_name_to_id)
            self._pkg_name_to_id[pkg.pkg_id] = pid
        tbl = self._id_to_pkg_table.get(pid)
        if tbl is None or tbl.numel() != pkg.num_experts or tbl.device != dev:
            tbl = torch.full((pkg.num_experts,), -1, dtype=torch.long, device=dev)
            self._id_to_pkg_table[pid] = tbl
        return pid, tbl

    @torch.no_grad()
    def fetch_experts(
        self, pkg: BKPPackage, unique_expert_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入去重后的微专家页号 unique_expert_ids: (U,)
        全向量化 GPU 页表寻址，返回显存中的 (pages_int8: (U, 2, 2048), scales_fp16: (U, 2))
        """
        self._clock += 1
        dev = self.cache_pages_int8.device
        eids_dev = unique_expert_ids.to(device=dev, dtype=torch.long)
        u_count = int(eids_dev.numel())
        self.total_queries += u_count

        pid, pkg_table = self._get_pkg_table(pkg, dev)
        slot_tensor = pkg_table[eids_dev]
        miss_mask = slot_tensor < 0

        if bool(miss_mask.any().item()):
            miss_pos = torch.where(miss_mask)[0]
            miss_eids_dev = eids_dev[miss_pos]
            n_miss = int(miss_eids_dev.numel())
            self.total_hits += (u_count - n_miss)

            if n_miss > self.max_slots:
                # 当单步未命中唯一专家页数超过 LRU 缓存容量 (如 512x512 首步冷启动 >16384 页) 时:
                # 直接从 CPU 主存拉取当前步所需全部微专家页，并将前 max_slots 个填入缓存槽位，彻底杜绝 topk 越界
                all_eids_cpu = eids_dev.cpu()
                out_pages = pkg.expert_pages_int8[all_eids_cpu].to(dev, non_blocking=True)
                out_scales = pkg.scales[eids_dev.to(pkg.scales.device)].to(dev, non_blocking=True)

                k_fill = self.max_slots
                fill_eids_dev = miss_eids_dev[:k_fill]
                fill_eids_cpu = fill_eids_dev.cpu()
                for active_tbl in self._id_to_pkg_table.values():
                    active_tbl.fill_(-1)
                all_slots = torch.arange(k_fill, dtype=torch.long, device=dev)
                self.cache_pages_int8[all_slots] = pkg.expert_pages_int8[fill_eids_cpu].to(dev, non_blocking=True)
                self.cache_scales_fp16[all_slots] = pkg.scales[fill_eids_dev.to(pkg.scales.device)].to(dev, non_blocking=True)
                pkg_table[fill_eids_dev] = all_slots
                self._slot_owner_pkg.fill_(pid)
                self._slot_owner_eid.copy_(fill_eids_dev)
                self._slot_last_used.fill_(self._clock)
                return out_pages, out_scales

            # 选出最久未使用的 n_miss 个冷槽位直接覆盖
            _, victim_slots = torch.topk(self._slot_last_used, k=n_miss, largest=False)

            # 向量化解除被淘汰冷槽位在原所属包页表中的映射
            old_pids = self._slot_owner_pkg[victim_slots]
            old_eids = self._slot_owner_eid[victim_slots]
            valid_old = old_pids >= 0
            if bool(valid_old.any().item()):
                for active_pid, tbl in self._id_to_pkg_table.items():
                    m_p = valid_old & (old_pids == active_pid)
                    if bool(m_p.any().item()):
                        evicted_eids = old_eids[m_p]
                        valid_range = evicted_eids < tbl.numel()
                        if bool(valid_range.any().item()):
                            tbl[evicted_eids[valid_range]] = -1

            miss_eids_cpu = miss_eids_dev.cpu()
            self.cache_pages_int8[victim_slots] = pkg.expert_pages_int8[miss_eids_cpu].to(dev, non_blocking=True)
            self.cache_scales_fp16[victim_slots] = pkg.scales[miss_eids_dev.to(pkg.scales.device)].to(dev, non_blocking=True)

            pkg_table[miss_eids_dev] = victim_slots
            self._slot_owner_pkg[victim_slots] = pid
            self._slot_owner_eid[victim_slots] = miss_eids_dev
            slot_tensor[miss_pos] = victim_slots
        else:
            self.total_hits += u_count

        self._slot_last_used[slot_tensor] = self._clock
        return self.cache_pages_int8[slot_tensor], self.cache_scales_fp16[slot_tensor]

    @torch.no_grad()
    def invalidate_or_refresh(self, pkg: BKPPackage, expert_ids: torch.Tensor) -> None:
        """当 CPU 端修改了 mmap 页后，全向量化刷新显存只读缓存中的对应槽位 (零 Python 循环)"""
        dev = self.cache_pages_int8.device
        _, pkg_table = self._get_pkg_table(pkg, dev)
        eids_dev = expert_ids.to(device=dev, dtype=torch.long)
        slots = pkg_table[eids_dev]
        hit_mask = slots >= 0
        if bool(hit_mask.any().item()):
            hit_slots = slots[hit_mask]
            hit_eids_dev = eids_dev[hit_mask]
            hit_eids_cpu = hit_eids_dev.cpu()
            self.cache_pages_int8[hit_slots] = pkg.expert_pages_int8[hit_eids_cpu].to(dev, non_blocking=True)
            self.cache_scales_fp16[hit_slots] = pkg.scales[hit_eids_dev.to(pkg.scales.device)].to(dev, non_blocking=True)

    @torch.no_grad()
    def evict_package(self, pkg_id: str) -> None:
        """当某个知识包被卸载或重挂载时，使该包在 GPU LRU 缓存池中的所有页表槽位立即失效"""
        pid = self._pkg_name_to_id.get(pkg_id)
        if pid is None:
            return
        tbl = self._id_to_pkg_table.get(pid)
        if tbl is not None:
            tbl.fill_(-1)
        owned_mask = self._slot_owner_pkg == pid
        if bool(owned_mask.any().item()):
            self._slot_owner_pkg[owned_mask] = -1
            self._slot_owner_eid[owned_mask] = -1
            self._slot_last_used[owned_mask] = 0

    @torch.no_grad()
    def clear(self) -> None:
        """清空全部 GPU LRU 微专家缓存槽位与命中统计"""
        for tbl in self._id_to_pkg_table.values():
            tbl.fill_(-1)
        self._slot_owner_pkg.fill_(-1)
        self._slot_owner_eid.fill_(-1)
        self._slot_last_used.zero_()
        self._clock = 0
        self.total_queries = 0
        self.total_hits = 0

    @property
    def hit_rate(self) -> float:
        return float(self.total_hits) / float(max(1, self.total_queries))


# ==============================================================================
# 4. BKPManager: 两级精确笛卡尔路由、闭式水位线 tau_p^eff、稀疏训练与向量外科手术
# ==============================================================================
class _ChunkedExpertInjectFn(torch.autograd.Function):
    """
    零中间激活留存的分块流式微专家注入算子:
    前向仅缓存 (M_hits, 1) 标量内积 (<512KB)，反向预计算 Token 级梯度 RMS 并采用原生 FP32 原子累加，
    在不留存任何 (M_hits, 2048) 展开张量的前提下大幅降低显存带宽与原子锁争用。
    """

    @staticmethod
    def forward(
        ctx,
        q_rms: torch.Tensor,
        u_dequant: torch.Tensor,
        v_dequant: torch.Tensor,
        w_typed: torch.Tensor,
        token_indices: torch.Tensor,
        inverse_idx: torch.Tensor,
        scale_factor: float,
        chunk_size: int = 32768,
    ) -> torch.Tensor:
        N, D = q_rms.shape
        delta_pkg = torch.zeros((N, D), dtype=q_rms.dtype, device=q_rms.device)
        m_hits = token_indices.numel()
        s_all = torch.empty((m_hits, 1), dtype=torch.float32, device=q_rms.device)
        eff_chunk = min(max(chunk_size, 4096), 8192)

        for start in range(0, m_hits, eff_chunk):
            end = min(start + eff_chunk, m_hits)
            tok_c = token_indices[start:end]
            inv_c = inverse_idx[start:end]
            q_c = q_rms[tok_c]
            u_c = u_dequant[inv_c]
            v_c = v_dequant[inv_c]
            s_c = (q_c * u_c).sum(dim=-1, keepdim=True, dtype=torch.float32) * scale_factor
            s_all[start:end] = s_c
            gate_c = F.silu(s_c).to(q_rms.dtype)
            out_c = (gate_c * w_typed[start:end]) * v_c
            delta_pkg.index_add_(0, tok_c, out_c)

        ctx.save_for_backward(q_rms, u_dequant, v_dequant, w_typed, token_indices, inverse_idx, s_all)
        ctx.scale_factor = scale_factor
        ctx.chunk_size = eff_chunk
        return delta_pkg

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_rms, u_dequant, v_dequant, w_typed, token_indices, inverse_idx, s_all = ctx.saved_tensors
        scale_factor: float = ctx.scale_factor
        chunk_size: int = ctx.chunk_size
        m_hits = token_indices.numel()

        need_q = ctx.needs_input_grad[0]
        need_u = ctx.needs_input_grad[1]
        need_v = ctx.needs_input_grad[2]
        need_w = ctx.needs_input_grad[3]

        grad_q_fp32 = torch.zeros(q_rms.shape, dtype=torch.float32, device=q_rms.device) if need_q else None
        grad_u_fp32 = torch.zeros(u_dequant.shape, dtype=torch.float32, device=u_dequant.device) if need_u else None
        grad_v_fp32 = torch.zeros(v_dequant.shape, dtype=torch.float32, device=v_dequant.device) if need_v else None
        grad_w = torch.empty_like(w_typed) if need_w else None

        for start in range(0, m_hits, chunk_size):
            end = min(start + chunk_size, m_hits)
            tok_c = token_indices[start:end]
            inv_c = inverse_idx[start:end]

            dz_c = grad_out[tok_c]
            v_c = v_dequant[inv_c]
            w_c = w_typed[start:end].float()
            s_c = s_all[start:end]

            sig_c = torch.sigmoid(s_c)
            gate_c = s_c * sig_c
            d_silu = sig_c * (1.0 + s_c * (1.0 - sig_c))

            if need_v:
                # 严格保持前向一阶切平面调制门控 gate_c = SiLU(s_c) 的真实线性变化，
                # 避免常数截断 (如 max(gate_c, 0.25)) 抹平一阶切向导数 <delta_q_perp, u_{m,perp}> 的协方差梯度！
                gate_v = torch.where(
                    gate_c.abs() > 0.02,
                    gate_c,
                    torch.where(s_c >= 0.0, torch.full_like(gate_c, 0.02), -torch.full_like(gate_c, 0.02)),
                )
                gv_c = (gate_v * w_c) * dz_c.float()
                grad_v_fp32.index_add_(0, inv_c, gv_c)

            if need_w or need_q or need_u:
                d_av = (dz_c * v_c).sum(dim=-1, keepdim=True, dtype=torch.float32)
                if need_w:
                    # 模长均衡化路由梯度: 扶持新生微专家 (max=4.0)，同时保底 min=1.0 严防成熟微专家在长程训练中路由梯度被掐断
                    v_rms = torch.sqrt(v_c.float().pow(2).mean(dim=-1, keepdim=True) + 1e-8)
                    norm_scale = (0.20 / v_rms.clamp(min=0.05)).clamp(min=1.0, max=4.0)
                    grad_w[start:end] = (d_av * norm_scale * gate_c).to(grad_w.dtype)
                if need_q or need_u:
                    d_s = (d_av * w_c * d_silu) * scale_factor
                    if need_q:
                        u_c = u_dequant[inv_c]
                        grad_q_fp32.index_add_(0, tok_c, d_s * u_c.float())
                    if need_u:
                        q_c = q_rms[tok_c].float()
                        grad_u_fp32.index_add_(0, inv_c, d_s * q_c)

        grad_q = grad_q_fp32.to(q_rms.dtype) if grad_q_fp32 is not None else None
        grad_u = grad_u_fp32.to(u_dequant.dtype) if grad_u_fp32 is not None else None
        grad_v = grad_v_fp32.to(v_dequant.dtype) if grad_v_fp32 is not None else None
        return grad_q, grad_u, grad_v, grad_w, None, None, None, None


class BKPManager(nn.Module):
    """
    对接 BicameraKernel 腰部总线 BKPWaistBus 的核心管理器:
    - 实现 query_and_inject(q_bkp) 与 forward(q_bkp)
    - 管理 GPU L1 路由快表 (Top-32 活跃包) 与 CPU L2 全局主页表后台巡检
    - 执行冈贝尔极值闭式水位线 tau_p^eff 门控、零激活短路、多头页号去重与零锚定 Softmax_1 注入
    """
    @staticmethod
    def _mod_key(pkg_id: str) -> str:
        return pkg_id.replace(".", "_")
    
    def __init__(
        self,
        config: Optional[BKPRoutingConfig] = None,
        cache_slots: int = 4096,  # 单元测试默认分配 4096 页 (16MB)，生产环境传入 131072 (512MB)
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.cfg = config or BKPRoutingConfig(vram_cache_slots=cache_slots)
        self.vram_cache = VRAMReadOnlyExpertCache(max_slots=self.cfg.vram_cache_slots, device=device)
        self.packages: nn.ModuleDict = nn.ModuleDict()
        self.l1_active_pkg_ids: List[str] = []
        self.base_pkg_id: Optional[str] = None

        # 记录最近一次前向激活的微专家轨迹，供稀疏梯度回写与 4.2.2 节故障归因外科手术使用
        self.last_active_trace: Dict[str, Tuple[torch.Tensor, ...]] = {}
        # 记录最近一次前向各挂载包的激活槽位数与权重和，供多包并发热插拔遥测使用
        self.last_pkg_slot_hits: Dict[str, int] = {}
        self.last_pkg_weight_sum: Dict[str, float] = {}
        self.last_avg_active_slots: float = 0.0
        # 轨迹级"以时间换空间"遥测: 单条序列累计去重唤醒微专家数 & 相邻思考步微专家切换率 (Jaccard Distance)
        self.last_unique_experts_per_seq: float = 0.0
        self.last_hop_jaccard_dist: float = 0.0
        # 轨迹内驻留疲劳不应期系数 (默认 0.0 严格保持逐点路由不变性；在长 CoT 蒸馏期可设为 >0 以惩罚原地踏步、激励跨步唤醒新微专家)
        self.trajectory_fatigue_beta: float = 0.0
        self.trajectory_fatigue_chunk: int = 32
        # 跨包同 Token 共激活累计矩阵 & 低命中率球面查询蓄水池 (供 AutoBKPForestController 执行 Auto-Merge / Auto-Spawn)
        self.pkg_coactivation_counts: Dict[Tuple[str, str], float] = {}
        self.pkg_token_hit_counts: Dict[str, float] = {}
        self._low_hit_q1_reservoir: Optional[torch.Tensor] = None
        self._low_hit_q2_reservoir: Optional[torch.Tensor] = None
        self._low_hit_periodic_queries: int = 0
        self._total_periodic_queries: int = 0
        # 暂存多步梯度累积 (grad_accum > 1) 期间各 micro-batch 的激活轨迹
        self._pending_grad_traces: Dict[str, List[Tuple[torch.Tensor, ...]]] = {}

    @torch.no_grad()
    def stash_microbatch_trace(self) -> None:
        """在每个 micro-batch 的 backward() 结束后立即调用: 零 PCIe 同步阻塞，直接在 GPU 上暂存紧凑稀疏梯度切片 (<32MB)"""
        for pkg_id, trace_item in list(self.last_active_trace.items()):
            eids, u_var, v_var = trace_item[0], trace_item[1], trace_item[2]
            hit_counts = trace_item[3] if len(trace_item) > 3 else None
            pkg: BKPPackage = self.packages[self._mod_key(pkg_id)]
            if u_var.grad is not None and v_var.grad is not None:
                eids_det = eids.detach().long()
                g_uv = torch.stack([u_var.grad.detach(), v_var.grad.detach()], dim=1).to(dtype=torch.bfloat16)
                hits_t = (
                    hit_counts.detach().to(dtype=torch.float32)
                    if hit_counts is not None
                    else torch.ones_like(eids_det, dtype=torch.float32)
                )
                self._pending_grad_traces.setdefault(pkg_id, []).append((eids_det, g_uv, hits_t))
                if pkg._grad_dirty_mask.numel() > 0:
                    pkg._grad_dirty_mask[eids_det.to(pkg._grad_dirty_mask.device)] = True
            u_var.grad = None
            v_var.grad = None
        self.last_active_trace.clear()

    def clear_grad_traces(self) -> None:
        """清空未提交的激活轨迹"""
        for pkg in self.packages.values():
            if pkg._grad_dirty_mask.numel() > 0:
                pkg._grad_dirty_mask.zero_()
        self._pending_grad_traces.clear()
        self.last_active_trace.clear()

    def mount_package(self, pkg: BKPPackage, pin_to_l1: bool = True) -> None:
        self.vram_cache.evict_package(pkg.pkg_id)
        ref_pkg = self._get_reference_manifold_pkg()
        if ref_pkg is not None and float(ref_pkg.q_ema_steps.item()) > float(pkg.q_ema_steps.item()):
            pkg._copy_manifold_stats_from(ref_pkg)
        self.packages[self._mod_key(pkg.pkg_id)] = pkg
        if pkg.is_base_pkg and not pkg.is_bridge_pkg and self.base_pkg_id is None:
            self.base_pkg_id = pkg.pkg_id
        if pin_to_l1 and pkg.pkg_id not in self.l1_active_pkg_ids:
            if len(self.l1_active_pkg_ids) >= self.cfg.l1_tlb_max_packages:
                # 淘汰非 Base 的最旧 L1 快表项至 L2 主页表
                for old_id in list(self.l1_active_pkg_ids):
                    if old_id != self.base_pkg_id:
                        self.l1_active_pkg_ids.remove(old_id)
                        break
            self.l1_active_pkg_ids.append(pkg.pkg_id)

    def unmount_package(self, pkg_id: str) -> None:
        self.vram_cache.evict_package(pkg_id)
        k = self._mod_key(pkg_id)
        if k in self.packages:
            self.packages[k].close_mmap()
            del self.packages[k]
        if pkg_id in self.l1_active_pkg_ids:
            self.l1_active_pkg_ids.remove(pkg_id)
        if self.base_pkg_id == pkg_id:
            self.base_pkg_id = None
            for pid in self.l1_active_pkg_ids:
                mk = self._mod_key(pid)
                if mk in self.packages and self.packages[mk].is_base_pkg and not self.packages[mk].is_bridge_pkg:
                    self.base_pkg_id = pid
                    break

    def _get_reference_manifold_pkg(self) -> Optional[BKPPackage]:
        """获取提供查询锥中心与球面度规的基准包: 优先取最新 EMA 步数的活跃包，严防卸载分裂时回退到旧快照"""
        best_pkg: Optional[BKPPackage] = None
        best_steps: float = -1.0
        for pid in self.l1_active_pkg_ids:
            k = self._mod_key(pid)
            if k in self.packages:
                st = float(self.packages[k].q_ema_steps.item())
                if st > best_steps:
                    best_steps = st
                    best_pkg = self.packages[k]
        if best_pkg is not None and best_steps > 0:
            return best_pkg
        if self.base_pkg_id is not None:
            base_key = self._mod_key(self.base_pkg_id)
            if base_key in self.packages:
                return self.packages[base_key]
        for pid in self.l1_active_pkg_ids:
            k = self._mod_key(pid)
            if k in self.packages:
                return self.packages[k]
        return None

    def _standardize_subqueries(
        self,
        q_bkp: torch.Tensor,
        update_center: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 2048 维查询向量重排为 8 个正交检索头 x 2 个子空间 x 128 维，并在每个子空间内独立标准化。
        严格保持逐 Token 因果独立性 (Pointwise Causal Independence):
        - 特征维标准化只沿 dim=-1 (单个 Token 自身内部) 进行，绝不跨时间步/跨样本聚合；
        - 全语料静态表征锥 (Anisotropic Cone / Rogue Dimensions) 的去共模改用知识包内持久化的历史
          EMA 背景中心 q_ema_mean (BatchNorm running-stats 思想)：前向只用更新前的统计量，
          因此 B=8 x T=1024 训练与 B=1 单步自回归推理数值 100% 恒等，
          既消除"全体 Token 撞向同几个垄断子键"的热门坍缩，又彻底杜绝预知未来的因果数据泄露。
        - 二级球面去均值 q_post_ema_mean 同理只读历史固化值，且在最后一次线性内积之前完成，
          因此不破坏上述逐 Token 因果独立性 (实测 B=8x32 vs B=1x1 差异 < 1e-7)。
        """
        N = q_bkp.shape[0]
        q_heads = q_bkp.view(N, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM).float()
        mean = q_heads.mean(dim=-1, keepdim=True)
        std = q_heads.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-6)
        q_norm = (q_heads - mean) / std

        center_pkg: Optional[BKPPackage] = self._get_reference_manifold_pkg()

        if center_pkg is not None:
            # 冻结更新前的历史中心 (避免读到本批次刚写入的新中心，保持前向严格因果)
            center = center_pkg.query_cone_center().detach().reshape(1, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM).clone()
            if update_center:
                # 暂存当前批次均值，待本步全部路由 (_route_single_package / normalized_codebooks) 完成后再更新 EMA
                center_pkg._pending_cone_batch_mean = q_norm.detach().mean(dim=0)
            # 100% 扣除全语料静态共模中心，将共有元特征留在 Kernel 主干，仅将特异语义偏差送入 .bkp
            q_centered = q_norm - center.to(dtype=q_norm.dtype)
            if update_center:
                q_c_det = q_centered.detach()
                center_pkg._pending_batch_cov = torch.einsum("nhsd,nhse->hsde", q_c_det, q_c_det) / float(max(1, N))

            # 2b. 正向流形球面化变换 T: M -> S^{127}
            #     乘以历史固化的对称 ZCA 白化算子 q_whiten_mat，将各向异性流形密集锥拉伸铺展到标准球面上
            w_mat = center_pkg.q_whiten_mat.detach().to(dtype=q_centered.dtype).clone()
            q_norm = torch.einsum("nhsd,hsde->nhse", q_centered, w_mat)
            mean2 = q_norm.mean(dim=-1, keepdim=True)
            std2 = q_norm.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-6)
            q_norm = (q_norm - mean2) / std2

            # 2c. 二级球面去均值 (Post-Normalization Spherical De-Biasing): 封杀"反去均值偏度作弊"
            #     上面 /std2 是逐 Token 非线性缩放: 减完一级中心后 E_t[diff] == 0，但除以各自模长后
            #     E_t[diff/|diff|] 会重新非零 (实测 full_v14 的 Head 5 上从 0.47 反弹到 2.75/11.31)。
            #     由于子键被库仑排斥压成近乎正交标架 (MaxCos=0.058)，只可能有一个子键对齐该偏度轴，
            #     它便永久白嫖 +1.64 sigma 的静态起跑底分，形成他人无法竞争的护城河 (单专家 1.16 亿次激活)。
            #     这里在最后一次线性内积 einsum("nhd,hkd->nhk", q1, u_keys) 之前只做一次纯线性平移:
            #     由双线性性 E_t[<q_t, u_k>] = <E_t[q_t], u_k> = 0 恒等成立，任何子键都无法凭静态偏置过线。
            #     注意: 之后绝不能再接任何逐 Token 归一化/除法，否则会再次非线性地造出新的偏度均值。
            post_center = (
                center_pkg.query_post_cone_center()
                .detach()
                .reshape(1, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM)
                .clone()
            )
            if update_center:
                center_pkg._pending_post_cone_batch_mean = q_norm.detach().mean(dim=0)
            q_norm = q_norm - post_center.to(dtype=q_norm.dtype)

        return q_norm[:, :, 0, :], q_norm[:, :, 1, :]  # q1, q2: (N, 8, 128)

    def _precompute_pkg_subkey_scores(
        self,
        pkg: BKPPackage,
        q1: torch.Tensor,
        q2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算单个知识包在 8 个正交子空间上的行列子键内积 s1, s2，并标记哪些行/列拥有至少一个非零有效微专家页。
        用于同空间多包联合统计环境背景标尺，彻底消除文件拆分/合并带来的标尺割裂；
        若包内同时存在成熟微专家 (scale_v > 5e-4) 与刚解锁的初生种子页 (scale_v <= 5e-4)，
        背景标尺统计严格锚定于成熟微专家行/列，防止初生空白子键扰动存量流形的 z-score 标尺。
        """
        k_p = pkg.grid_k
        u_keys, v_keys = pkg.normalized_codebooks()  # (8, K_p, 128)
        s1 = torch.einsum("nhd,hkd->nhk", q1.float(), u_keys.float())
        s2 = torch.einsum("nhd,hkd->nhk", q2.float(), v_keys.float())
        sv_2d = pkg.scales[:, 1].view(k_p, k_p)
        mature_2d = sv_2d > 2e-5
        vr_mat = mature_2d.any(dim=1)
        vc_mat = mature_2d.any(dim=0)
        base_64_mask = torch.zeros(k_p, dtype=torch.bool, device=q1.device)
        base_64_mask[: min(64, k_p)] = True
        # 当成熟行/列 >= 8 时严格锚定成熟行/列；当刚写入少量微专家 (< 8) 时锚定前 64 个基础正交子键，
        # 既避免 1~2 行自减均值归零，又保证 64->128 四象限扩容瞬间背景 mu/std/tau_z 100% 不变 (扩容零误差 = 0.0000)！
        valid_r = torch.where(vr_mat.sum() >= 8, vr_mat, base_64_mask | vr_mat)
        valid_c = torch.where(vc_mat.sum() >= 8, vc_mat, base_64_mask | vc_mat)
        return s1, s2, valid_r, valid_c

    def _route_tile_block(
        self,
        pkg: BKPPackage,
        s1_tile: torch.Tensor,
        s2_tile: torch.Tensor,
        row_offset_or_map: Union[int, torch.Tensor],
        col_offset_or_map: Union[int, torch.Tensor],
        mu_r: torch.Tensor,
        mu_c: torch.Tensor,
        std_r: torch.Tensor,
        std_c: torch.Tensor,
        tau_z_sub: Union[float, torch.Tensor],
        is_pkg_rw_train: bool,
        is_bridge_tile: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, k_t, dev = s1_tile.shape[0], s1_tile.shape[-1], s1_tile.device
        r_idx = (
            torch.arange(k_t, device=dev, dtype=torch.long) + row_offset_or_map
            if isinstance(row_offset_or_map, int)
            else row_offset_or_map.to(device=dev, dtype=torch.long)
        )
        c_idx = (
            torch.arange(k_t, device=dev, dtype=torch.long) + col_offset_or_map
            if isinstance(col_offset_or_map, int)
            else col_offset_or_map.to(device=dev, dtype=torch.long)
        )

        tile_sv_2d = pkg.scales[:, 1].view(pkg.grid_k, pkg.grid_k)[r_idx][:, c_idx]
        if is_bridge_tile:
            h_t = k_t // 2
            off_diag = torch.ones((k_t, k_t), dtype=torch.bool, device=dev)
            off_diag[:h_t, :h_t] = False
            off_diag[h_t:, h_t:] = False
            tile_sv_2d = torch.where(off_diag, tile_sv_2d, torch.zeros_like(tile_sv_2d))

        active_sv_2d = tile_sv_2d > 0
        has_any_active = active_sv_2d.any()
        allow_cold_start = (
            is_pkg_rw_train
            and pkg.is_base_pkg
            and (len(self.l1_active_pkg_ids) <= 1)
            and (not is_bridge_tile)
            and (~has_any_active)
        )
        route_vr = active_sv_2d.any(dim=1) | allow_cold_start
        route_vc = active_sv_2d.any(dim=0) | allow_cold_start
        s1_tile = s1_tile.masked_fill(~route_vr.view(1, 1, k_t), -1e4)
        s2_tile = s2_tile.masked_fill(~route_vc.view(1, 1, k_t), -1e4)

        k_sub = min(self.cfg.sub_top_k, k_t)
        if is_bridge_tile and k_t >= 4:
            h_t = k_t // 2
            k_half = max(1, min(k_sub // 2, h_t))
            s1_a, i_a = torch.topk(s1_tile[:, :, :h_t], k=k_half, dim=-1)
            s1_b, i_b = torch.topk(s1_tile[:, :, h_t:], k=k_half, dim=-1)
            top_s1, top_i_loc = torch.cat([s1_a, s1_b], dim=-1), torch.cat([i_a, i_b + h_t], dim=-1)
            s2_a, j_a = torch.topk(s2_tile[:, :, :h_t], k=k_half, dim=-1)
            s2_b, j_b = torch.topk(s2_tile[:, :, h_t:], k=k_half, dim=-1)
            top_s2, top_j_loc = torch.cat([s2_a, s2_b], dim=-1), torch.cat([j_a, j_b + h_t], dim=-1)
            k_sub = top_i_loc.shape[-1]
        else:
            top_s1, top_i_loc = torch.topk(s1_tile, k=k_sub, dim=-1)
            top_s2, top_j_loc = torch.topk(s2_tile, k=k_sub, dim=-1)

        # 定律 1: 零超参数球面三角剖分单纯形切子空间投影能量 (Zero-Hyperparameter Simplex Tangent Subspace Energy)
        # - 软滤波下界: 3-单纯形 (最小球面三角剖分面元) 重心投影衰减比 1/sqrt(3) * tau_z^* ≈ 0.5774 * tau_z^*
        # - 软滤波上界: 2-测地线 (球面大圆边) 中点投影衰减比 1/sqrt(2) * tau_z^* ≈ 0.7071 * tau_z^*
        # - 通过三次厄米特 C^1 平滑步阶函数 H(u) = u^2*(3 - 2u) 滤除低于 1/sqrt(3)*tau_z^* 的背景正交噪声轴后，
        #   无需人为指定或截断子空间维数 d_s，直接对测地圆内全部非零响应轴求全子空间 L2 范数 Z_simp = sqrt(sum_k z_tilde_k^2)！
        zr = (top_s1 - mu_r) / std_r
        zc = (top_s2 - mu_c) / std_c
        tau_floor = SIMPLEX_3_BARYCENTER_PROJ * tau_z_sub
        tau_span_raw = (SIMPLEX_2_GEODESIC_PROJ - SIMPLEX_3_BARYCENTER_PROJ) * tau_z_sub
        tau_span = tau_span_raw if not isinstance(tau_span_raw, float) else max(1e-6, tau_span_raw)
        ur_gate = ((zr - tau_floor) / tau_span).clamp(0.0, 1.0)
        uc_gate = ((zc - tau_floor) / tau_span).clamp(0.0, 1.0)
        zr_tilde = zr * (ur_gate * ur_gate * (3.0 - 2.0 * ur_gate))
        zc_tilde = zc * (uc_gate * uc_gate * (3.0 - 2.0 * uc_gate))
        z_simp_r = torch.sqrt(zr_tilde.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        z_simp_c = torch.sqrt(zc_tilde.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        excess_zr = torch.maximum(
            F.relu(zr - tau_z_sub),
            F.relu(z_simp_r - tau_z_sub) * (zr_tilde / (z_simp_r + 1e-12)),
        )
        excess_zc = torch.maximum(
            F.relu(zc - tau_z_sub),
            F.relu(z_simp_c - tau_z_sub) * (zc_tilde / (z_simp_c + 1e-12)),
        )

        prod_grid = excess_zr.unsqueeze(-1) * excess_zc.unsqueeze(-2)
        grid_excess = torch.where(prod_grid > 0.0, torch.sqrt(prod_grid.clamp(min=1e-12)), torch.zeros_like(prod_grid))
        tile_grid_ids = top_i_loc.unsqueeze(-1) * k_t + top_j_loc.unsqueeze(-2)
        tile_sv_flat = tile_sv_2d.reshape(-1)
        page_sv = tile_sv_flat[tile_grid_ids].to(grid_excess.dtype)
        page_eligible = (page_sv > 0.0) | allow_cold_start
        grid_excess = grid_excess * page_eligible.to(grid_excess.dtype)

        max_sv_t = tile_sv_flat.max().clamp(min=1e-12)
        ref_sv_t = (0.20 * max_sv_t).clamp(min=5e-5)
        if is_pkg_rw_train:
            mat_gate = torch.where(
                max_sv_t <= 2e-5,
                torch.where(page_eligible, torch.full_like(page_sv, 0.92), torch.zeros_like(page_sv)),
                torch.where(page_sv > 0.0, (page_sv / ref_sv_t).clamp(min=0.92, max=1.0), torch.zeros_like(page_sv)),
            )
        else:
            mat_gate = (page_sv > 2e-5).to(grid_excess.dtype)
        grid_excess = grid_excess * mat_gate

        # 三次厄米特 C^1 平滑面片包络 (Cubic Hermite Smoothstep Envelope):
        # 废除破坏连续性的硬性布尔峰值截断 is_principal，改用 H(t) = t^2 * (3 - 2*t), H(0)=H'(0)=0, H(1)=1, H'(1)=0，
        # 既保留局部切平面内所有强协同支撑微专家，又使滑出邻域的边缘格点以 O(t^3) 平滑归零，严格保证流形场 C^1 一阶连续可微！
        e_max = grid_excess.amax(dim=(-2, -1), keepdim=True)
        t_env = (grid_excess / (e_max + 1e-12)).clamp(0.0, 1.0)
        grid_excess = grid_excess * (t_env * t_env * (3.0 - 2.0 * t_env))

        # 未知语境自适应测地圆软化闭包 (Adaptive Geodesic Closure Interpolation):
        soft_gamma = float(getattr(self.cfg, "geodesic_closure_soft_gamma", 0.0))
        if soft_gamma > 0.0 and soft_gamma < 1.0:
            tau_z_soft = tau_z_sub * soft_gamma
            exc_zr_soft = F.relu(zr - tau_z_soft)
            exc_zc_soft = F.relu(zc - tau_z_soft)
            # 双边平衡门控: 要求行子键与列子键同时显著过线 (min/max >= 0.25)，杜绝单边高频特征误唤醒跨流形闭包格点
            bal_ratio = torch.minimum(exc_zr_soft.unsqueeze(-1), exc_zc_soft.unsqueeze(-2)) / (
                torch.maximum(exc_zr_soft.unsqueeze(-1), exc_zc_soft.unsqueeze(-2)).clamp(min=1e-6)
            )
            prod_soft = exc_zr_soft.unsqueeze(-1) * exc_zc_soft.unsqueeze(-2) * (bal_ratio >= 0.10).to(exc_zr_soft.dtype)
            grid_soft = torch.where(prod_soft > 0.0, torch.sqrt(prod_soft.clamp(min=1e-12)), torch.zeros_like(prod_soft))
            grid_soft = grid_soft * page_eligible.to(grid_soft.dtype) * mat_gate * float(getattr(self.cfg, "geodesic_closure_damping", 0.45))
            e_max_s = grid_soft.amax(dim=(-2, -1), keepdim=True)
            t_env_s = (grid_soft / (e_max_s + 1e-12)).clamp(0.0, 1.0)
            grid_soft = grid_soft * (t_env_s * t_env_s * (3.0 - 2.0 * t_env_s))
            head_has_strict = (grid_excess > 0.0).any(dim=-1, keepdim=True).any(dim=-2, keepdim=True)
            closure_mask_dev = getattr(pkg, "_closure_expert_mask_dev", None)
            if closure_mask_dev is not None and closure_mask_dev.numel() == pkg.num_experts:
                glob_ids_tmp = r_idx[top_i_loc].unsqueeze(-1) * pkg.grid_k + c_idx[top_j_loc].unsqueeze(-2)
                is_closure_tile = closure_mask_dev[glob_ids_tmp]
                in_closure_cone = getattr(pkg, "_cur_in_closure_cone", None)
                if in_closure_cone is not None:
                    grid_strict_parent = torch.where(~is_closure_tile, grid_excess, torch.zeros_like(grid_excess))
                    grid_closure_all = torch.where(head_has_strict, grid_excess, grid_soft)
                    grid_excess = torch.where(in_closure_cone, grid_closure_all, grid_strict_parent)
                else:
                    grid_excess = torch.where(head_has_strict, grid_excess, grid_soft)
            else:
                grid_excess = torch.where(head_has_strict, grid_excess, grid_soft)

        glob_grid_ids = r_idx[top_i_loc].unsqueeze(-1) * pkg.grid_k + c_idx[top_j_loc].unsqueeze(-2)
        flat_excess = grid_excess.view(N, BKP_NUM_HEADS, k_sub * k_sub)
        flat_ids = glob_grid_ids.view(N, BKP_NUM_HEADS, k_sub * k_sub)
        k_head = min(self.cfg.head_top_k, k_sub * k_sub)
        gated_excess, best_pos = torch.topk(flat_excess, k=k_head, dim=-1)
        best_ids = torch.gather(flat_ids, dim=-1, index=best_pos)
        return best_ids, gated_excess

    def _route_single_package(
        self,
        pkg: BKPPackage,
        q1: torch.Tensor,
        q2: torch.Tensor,
        precomputed_scores: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ambient_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单个知识包在统一环境空间测地圆标尺下执行 O(K_p) 寻址与局部峰值包容:
        返回:
        - cand_ids: (N, 8, k_cand) 候选微专家一维页号 m = i * K_p + j
        - cand_excess: (N, 8, k_cand) 超额无偏标准化得分 (相对于冈贝尔极值底线 tau_z_eff)
        """
        N = q1.shape[0]
        k_p = pkg.grid_k
        closure_anchor_q = getattr(pkg, "_closure_query_anchors", None)
        if closure_anchor_q is not None:
            with torch.no_grad():
                q_cur_sph = F.normalize(torch.cat([q1, q2], dim=-1).reshape(N, D_BKP), dim=-1)
                pkg._cur_in_closure_cone = (torch.matmul(q_cur_sph, closure_anchor_q.t()).max(dim=-1).values > 0.88).view(N, 1, 1, 1)
        else:
            pkg._cur_in_closure_cone = None
        if precomputed_scores is not None:
            s1, s2, valid_r, valid_c = precomputed_scores
        else:
            s1, s2, valid_r, valid_c = self._precompute_pkg_subkey_scores(pkg, q1, q2)

        if ambient_stats is not None:
            mu_r, mu_c, std_r, std_c, tau_z_eff = ambient_stats
        else:
            q1_std = q1.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5).detach()
            q2_std = q2.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5).detach()
            vr_f = valid_r.to(s1.dtype).view(1, 1, -1)
            vc_f = valid_c.to(s2.dtype).view(1, 1, -1)
            cnt_r = vr_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            cnt_c = vc_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mu_r = ((s1 * vr_f).sum(dim=-1, keepdim=True) / cnt_r).detach()
            mu_c = ((s2 * vc_f).sum(dim=-1, keepdim=True) / cnt_c).detach()
            var_r = (((s1 - mu_r) * vr_f).pow(2).sum(dim=-1, keepdim=True) / cnt_r).clamp(min=1e-10)
            var_c = (((s2 - mu_c) * vc_f).pow(2).sum(dim=-1, keepdim=True) / cnt_c).clamp(min=1e-10)
            std_r = (torch.sqrt(var_r) / q1_std).clamp(min=1e-5).detach()
            std_c = (torch.sqrt(var_c) / q2_std).clamp(min=1e-5).detach()

            q_idx = 1.0 if pkg.is_base_pkg else float(getattr(pkg, "_quality_q_cached", 1.0))
            alpha_p = max(1e-9, self.cfg.alpha_0 * q_idx)
            # 冈贝尔极值上确界闭式门控水位线 tau_z^*(N_grid) = mu_Gumbel(N_grid) + sigma_Gumbel(N_grid):
            # 由当前包有效子键网格密度 k_p 与孔径因子 alpha_p 直接解析导出，零经验魔数
            n_grid_eff = max(4.0, float(k_p) / alpha_p)
            tau_z_eff = closed_form_gumbel_threshold(n_grid_eff)

        tau_z_sub = tau_z_eff if isinstance(tau_z_eff, float) else tau_z_eff
        is_pkg_rw_train = (
            self.training
            and torch.is_grad_enabled()
            and pkg.mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE)
        )

        if pkg.is_bridge_pkg:
            return self._route_tile_block(
                pkg, s1, s2, 0, 0, mu_r, mu_c, std_r, std_c, tau_z_sub, is_pkg_rw_train, is_bridge_tile=True
            )

        if k_p == 128:
            sv_2d = pkg.scales[:, 1].view(128, 128)
            has_quad234 = bool((sv_2d[64:, :] > 0).any() or (sv_2d[:, 64:] > 0).any())
            if not has_quad234:
                # 四象限倍增扩容后新增的三个象限仍全为空白 (scale_v == 0)：
                # 严格原位路由左上存量象限 [:64, :64]，保证扩容瞬间前向输出误差严格为 0.000000！
                return self._route_tile_block(
                    pkg, s1[:, :, :64], s2[:, :, :64], 0, 0, mu_r, mu_c, std_r, std_c, tau_z_sub, is_pkg_rw_train
                )
            if pkg._sparse_128_triblock_mode is None:
                outside_br = torch.ones((128, 128), dtype=torch.bool, device=q1.device)
                outside_br[:64, :64] = False
                outside_br[64:, 64:] = False
                outside_br[:32, 64:96] = False
                outside_br[64:96, :32] = False
                pkg._sparse_128_triblock_mode = not bool((sv_2d > 0)[outside_br].any().item())
            if pkg._sparse_128_triblock_mode:
                id_aa, ex_aa = self._route_tile_block(
                    pkg, s1[:, :, :64], s2[:, :, :64], 0, 0, mu_r, mu_c, std_r, std_c, tau_z_sub, is_pkg_rw_train
                )
                id_bb, ex_bb = self._route_tile_block(
                    pkg, s1[:, :, 64:], s2[:, :, 64:], 64, 64, mu_r, mu_c, std_r, std_c, tau_z_sub, is_pkg_rw_train
                )
                br_map = torch.cat([torch.arange(0, 32, device=q1.device), torch.arange(64, 96, device=q1.device)])
                id_br, ex_br = self._route_tile_block(
                    pkg,
                    s1[:, :, br_map],
                    s2[:, :, br_map],
                    br_map,
                    br_map,
                    mu_r,
                    mu_c,
                    std_r,
                    std_c,
                    tau_z_sub,
                    is_pkg_rw_train,
                    is_bridge_tile=True,
                )
                return torch.cat([id_aa, id_bb, id_br], dim=-1), torch.cat([ex_aa, ex_bb, ex_br], dim=-1)

        return self._route_tile_block(
            pkg, s1, s2, 0, 0, mu_r, mu_c, std_r, std_c, tau_z_sub, is_pkg_rw_train, is_bridge_tile=False
        )

    @torch.no_grad()
    def step_l2_page_walker(self, q_bkp: torch.Tensor) -> List[str]:
        """
        3.1.1 节 CPU L2 全局主页表后台巡检器 (Background Page Walker):
        扫描未在 L1 快表中的冷门知识包，利用加性可分离恒等式 s_max^(p) = max(S1) + max(S2)
        一步判定是否跨过有效唤醒水位线；若过线则将其页表热切入 GPU L1 快表。
        """
        q_flat = q_bkp.reshape(-1, D_BKP)
        q1, q2 = self._standardize_subqueries(q_flat)
        promoted: List[str] = []

        for _, pkg in self.packages.items():
            if pkg.pkg_id in self.l1_active_pkg_ids:
                continue
            _, cand_excess = self._route_single_package(pkg, q1, q2)
            if bool((cand_excess > 0).any().item()):
                self.mount_package(pkg, pin_to_l1=True)
                promoted.append(pkg.pkg_id)
        return promoted

    def query_and_inject(
        self,
        q_bkp: torch.Tensor,
        puncture_dropout_prob: float = 0.0,
        puncture_expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        腰部总线核心调用接口:
        输入 q_bkp: (B, T, 2048) -> 输出注入残差流的知识脉冲 delta_z_bkp: (B, T, 2048)
        同时原生产出基于测地圆半径与 L2 能量守恒单位分解的亥姆霍兹/霍奇三分解场诊断缓存 self._last_helmholtz_cache。
        """
        if q_bkp.dim() == 2:
            B, T = q_bkp.shape[0], 1
        else:
            B, T = q_bkp.shape[0], q_bkp.shape[1]
        N = B * T
        dev = q_bkp.device

        if len(self.l1_active_pkg_ids) == 0:
            zero_scalar = torch.zeros((B, T), device=dev, dtype=torch.float32)
            self._last_helmholtz_cache = {
                "div_v": zero_scalar,
                "eta_curl": torch.ones_like(zero_scalar),
                "eta_harmonic": zero_scalar,
                "active_experts": torch.zeros((B, T, 1), device=dev, dtype=torch.long),
                "active_weights": zero_scalar.unsqueeze(-1),
                "max_resonance": zero_scalar,
            }
            return torch.zeros_like(q_bkp)

        q_flat = q_bkp.reshape(N, D_BKP)

        # 仅在"可写训练态 + 梯度开启 + 码本可训练 (寻址坐标系构建期)"时收集查询锥 EMA 统计量：
        # 一旦读取层与码本冻结 (u_codebook.requires_grad=False)，查询锥中心同步锁死，严防读取信号漂移
        base_pkg: Optional[BKPPackage] = self._get_reference_manifold_pkg()
        collect_center_stats = bool(
            self.training
            and N >= 16
            and base_pkg is not None
            and torch.is_grad_enabled()
            and base_pkg.u_codebook.requires_grad
            and base_pkg.mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE)
        )
        q1, q2 = self._standardize_subqueries(q_flat, update_center=collect_center_stats)

        # 1. 仅扫描驻留在 L1 快表 (l1_active_pkg_ids) 中的活跃坐标卡，绝不扫描未进 L1 的冷包
        ordered_pids: List[str] = []
        if self.base_pkg_id and self.base_pkg_id in self.l1_active_pkg_ids:
            ordered_pids.append(self.base_pkg_id)
        for pid in self.l1_active_pkg_ids:
            if pid not in ordered_pids and self._mod_key(pid) in self.packages:
                ordered_pids.append(pid)

        pkg_routing_results: List[Tuple[int, BKPPackage, torch.Tensor, torch.Tensor]] = []

        # 同空间统一测地圆标尺 (Unified Ambient Geodesic Ball Metric) + 多包码本单次融合投影:
        # 所有挂载的 .bkp 共同组成同一环境向量空间的外置智能，不应因文件被 Split 拆分或 Merge 合并而改变距离标尺。
        precomputed_list = []
        valid_s1_chunks = []
        valid_s2_chunks = []
        if len(ordered_pids) == 1:
            pkg = self.packages[self._mod_key(ordered_pids[0])]
            s1_p, s2_p, vr_p, vc_p = self._precompute_pkg_subkey_scores(pkg, q1, q2)
            precomputed_list.append((pkg, s1_p, s2_p, vr_p, vc_p))
            if not pkg.is_bridge_pkg:
                valid_s1_chunks.append((s1_p, vr_p))
                valid_s2_chunks.append((s2_p, vc_p))
        elif len(ordered_pids) > 1:
            pkgs_ord = [self.packages[self._mod_key(pid)] for pid in ordered_pids]
            k_sizes = [p.grid_k for p in pkgs_ord]
            u_list, v_list = [], []
            for p in pkgs_ord:
                uk, vk = p.normalized_codebooks()
                u_list.append(uk.float())
                v_list.append(vk.float())
            u_cat = torch.cat(u_list, dim=1)  # (8, sum_K, 128)
            v_cat = torch.cat(v_list, dim=1)  # (8, sum_K, 128)
            s1_cat = torch.einsum("nhd,hkd->nhk", q1, u_cat)
            s2_cat = torch.einsum("nhd,hkd->nhk", q2, v_cat)
            s1_splits = torch.split(s1_cat, k_sizes, dim=-1)
            s2_splits = torch.split(s2_cat, k_sizes, dim=-1)
            for p, s1_p, s2_p in zip(pkgs_ord, s1_splits, s2_splits):
                k_p = p.grid_k
                sv_2d = p.scales[:, 1].view(k_p, k_p)
                mature_2d = sv_2d > 2e-5
                vr_mat = mature_2d.any(dim=1)
                vc_mat = mature_2d.any(dim=0)
                base_64_p = torch.zeros(k_p, dtype=torch.bool, device=q1.device)
                base_64_p[: min(64, k_p)] = True
                vr_p = torch.where(vr_mat.sum() >= 8, vr_mat, base_64_p | vr_mat)
                vc_p = torch.where(vc_mat.sum() >= 8, vc_mat, base_64_p | vc_mat)
                precomputed_list.append((p, s1_p, s2_p, vr_p, vc_p))
                if not p.is_bridge_pkg:
                    valid_s1_chunks.append((s1_p, vr_p))
                    valid_s2_chunks.append((s2_p, vc_p))

        ambient_stats = None
        if len(valid_s1_chunks) > 0:
            s1_amb_all = torch.cat([c[0] for c in valid_s1_chunks], dim=-1)
            vr_amb_all = torch.cat([c[1] for c in valid_s1_chunks], dim=-1)
            s2_amb_all = torch.cat([c[0] for c in valid_s2_chunks], dim=-1)
            vc_amb_all = torch.cat([c[1] for c in valid_s2_chunks], dim=-1)

            vr_f = vr_amb_all.to(s1_amb_all.dtype).view(1, 1, -1)
            vc_f = vc_amb_all.to(s2_amb_all.dtype).view(1, 1, -1)
            cnt_r = vr_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            cnt_c = vc_f.sum(dim=-1, keepdim=True).clamp(min=1.0)

            q1_std = q1.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5).detach()
            q2_std = q2.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5).detach()
            mu_r_amb = ((s1_amb_all * vr_f).sum(dim=-1, keepdim=True) / cnt_r).detach()
            mu_c_amb = ((s2_amb_all * vc_f).sum(dim=-1, keepdim=True) / cnt_c).detach()
            var_r_amb = (((s1_amb_all - mu_r_amb) * vr_f).pow(2).sum(dim=-1, keepdim=True) / cnt_r).clamp(min=1e-10)
            var_c_amb = (((s2_amb_all - mu_c_amb) * vc_f).pow(2).sum(dim=-1, keepdim=True) / cnt_c).clamp(min=1e-10)
            std_r_amb = (torch.sqrt(var_r_amb) / q1_std).clamp(min=1e-5).detach()
            std_c_amb = (torch.sqrt(var_c_amb) / q2_std).clamp(min=1e-5).detach()
            eff_k_amb = float(max(cnt_r.max().item(), cnt_c.max().item(), 64.0))
            n_grid_amb = max(4.0, eff_k_amb / max(1e-9, self.cfg.alpha_0))
            tau_z_amb = closed_form_gumbel_threshold(n_grid_amb)
            ambient_stats = (mu_r_amb, mu_c_amb, std_r_amb, std_c_amb, tau_z_amb)

        for p_idx, (pkg, s1_p, s2_p, vr_p, vc_p) in enumerate(precomputed_list):
            c_ids, c_excess = self._route_single_package(
                pkg,
                q1,
                q2,
                precomputed_scores=(s1_p, s2_p, vr_p, vc_p),
                ambient_stats=ambient_stats,
            )
            if puncture_expert_mask is not None and puncture_expert_mask.numel() == pkg.num_experts:
                p_mask_dev = puncture_expert_mask.to(c_ids.device)
                c_excess = c_excess.masked_fill(p_mask_dev[c_ids.clamp(0, pkg.num_experts - 1)], -1e9)
            if puncture_dropout_prob > 0.0:
                drop_m = torch.rand_like(c_excess) < float(puncture_dropout_prob)
                c_excess = c_excess.masked_fill(drop_m, -1e9)
            # 包级零激活短路秒筛: 若该包没有任何微专家跨过冈贝尔极值底线 tau_z_eff，直接旁路跳过，0 页拉取！
            if bool((c_excess > 0.0).any()):
                pkg_routing_results.append((p_idx, pkg, c_ids, c_excess))

        # 若所有包均未过线 (Zero-Activation Short-Circuit)，直接返回零张量，零访存开销
        if len(pkg_routing_results) == 0:
            if collect_center_stats and base_pkg is not None:
                if getattr(base_pkg, "_pending_cone_batch_mean", None) is not None:
                    base_pkg.update_query_cone_center(base_pkg._pending_cone_batch_mean)
                    base_pkg._pending_cone_batch_mean = None
                if getattr(base_pkg, "_pending_batch_cov", None) is not None:
                    base_pkg.update_query_manifold_metric(base_pkg._pending_batch_cov)
                    base_pkg._pending_batch_cov = None
                if getattr(base_pkg, "_pending_post_cone_batch_mean", None) is not None:
                    base_pkg.update_query_post_cone_center(base_pkg._pending_post_cone_batch_mean)
                    base_pkg._pending_post_cone_batch_mean = None
            zero_scalar = torch.zeros((B, T), device=dev, dtype=torch.float32)
            self._last_helmholtz_cache = {
                "div_v": zero_scalar,
                "eta_curl": torch.ones_like(zero_scalar),
                "eta_harmonic": zero_scalar,
                "active_experts": torch.zeros((B, T, 1), device=dev, dtype=torch.long),
                "active_weights": zero_scalar.unsqueeze(-1),
                "max_resonance": zero_scalar,
            }
            return torch.zeros_like(q_bkp)

        # 2. 汇总过线候选并执行零点锚定动态归一化 (Zero-Anchored Softmax_1)
        #    将 (p_idx, expert_id) 编码为全局唯一 64 位复合键: global_id = (p_idx << 32) | expert_id
        #    由于不同知识包 p_idx 的高 32 位天然互异，重复页号仅可能出现在单包内部的 H*K_c 个候选之间；
        #    因此在单包内部先行完成跨头去重并截取前 k_global 名，再跨包汇总取全局 Top-K (数学严格恒等，消除 O((sum C_p)^2) 大矩阵显存峰值)
        all_global_ids = []
        all_excess = []
        k_cap = self.cfg.global_top_k_max
        for p_idx, pkg, c_ids, c_excess in pkg_routing_results:
            H, K_c = c_ids.shape[1], c_ids.shape[2]
            flat_c_ids = c_ids.reshape(N, H * K_c)
            flat_c_exc = c_excess.reshape(N, H * K_c)
            s_exc, s_idx = torch.sort(flat_c_exc, dim=-1, descending=True)
            s_ids = torch.gather(flat_c_ids, dim=-1, index=s_idx)
            dup_m = (s_ids.unsqueeze(-1) == s_ids.unsqueeze(-2)).triu(diagonal=1).any(dim=-2)
            dedup_exc = s_exc.masked_fill(dup_m, -1e9)
            if H * K_c > k_cap:
                p_exc, p_pos = torch.topk(dedup_exc, k=k_cap, dim=-1)
                p_ids = torch.gather(s_ids, dim=-1, index=p_pos)
            else:
                p_exc, p_ids = dedup_exc, s_ids
            all_global_ids.append((int(p_idx) << 32) | p_ids)
            all_excess.append(p_exc)

        if len(all_global_ids) == 1:
            cat_gids = all_global_ids[0]
            cat_excess = all_excess[0]
        else:
            cat_gids = torch.cat(all_global_ids, dim=-1)
            cat_excess = torch.cat(all_excess, dim=-1)

        k_global = min(k_cap, cat_excess.shape[-1])
        top_excess, top_pos = torch.topk(cat_excess, k=k_global, dim=-1)  # (N, k_global)
        top_gids = torch.gather(cat_gids, dim=-1, index=top_pos)

        # 可选: 轨迹内分块因果驻留疲劳调制 (Causal Trajectory Residence Fatigue)
        # 当长 CoT 训练开启 trajectory_fatigue_beta > 0 且 T > chunk 时，若某微专家在前序思考块中已被高频霸占，
        # 对其当前块的超额得分施加因果不应期衰减，逼迫模型随着 CoT 步数推进主动跳向球面流形新区域唤醒不同微专家！
        if self.training and self.trajectory_fatigue_beta > 0.0 and T > self.trajectory_fatigue_chunk:
            with torch.no_grad():
                c_len = self.trajectory_fatigue_chunk
                n_chunks = (T + c_len - 1) // c_len
                gids_bt = top_gids.view(B, T, k_global)
                exc_bt_det = top_excess.detach().view(B, T, k_global)
                fatigue_scale_bt = torch.ones((B, T, k_global), dtype=top_excess.dtype, device=top_excess.device)
                for c_idx in range(1, n_chunks):
                    s_t = c_idx * c_len
                    e_t = min(T, (c_idx + 1) * c_len)
                    prev_s = max(0, (c_idx - 2) * c_len)
                    prev_gids = gids_bt[:, prev_s:s_t, :].reshape(B, -1)
                    prev_act = (exc_bt_det[:, prev_s:s_t, :] > 0.0).reshape(B, -1)
                    prev_valid_gids = torch.where(prev_act, prev_gids, torch.full_like(prev_gids, -999999))
                    cur_gids = gids_bt[:, s_t:e_t, :]  # (B, cur_len, k_global)
                    repeat_hits = (
                        cur_gids.unsqueeze(-1) == prev_valid_gids.unsqueeze(1).unsqueeze(2)
                    ).float().sum(dim=-1)
                    fatigue_scale_bt[:, s_t:e_t, :] = torch.rsqrt(
                        1.0 + self.trajectory_fatigue_beta * repeat_hits
                    ).to(top_excess.dtype)
            top_excess = top_excess * fatigue_scale_bt.view(N, k_global)

        active_mask = top_excess > 0.0
        slot_counts_per_tok = active_mask.float().sum(dim=-1)
        self._fwd_telemetry_step = getattr(self, "_fwd_telemetry_step", 0) + 1
        do_sync_telemetry = (not self.training) or (self._fwd_telemetry_step <= 4) or (self._fwd_telemetry_step % 8 == 0)

        if do_sync_telemetry:
            # 统计单序列累计去重唤醒微专家数 (UniqueExpertsPerSeq) 与跨步流形跳变率 (HopJaccardDistance)
            with torch.no_grad():
                self.last_avg_active_slots = float(slot_counts_per_tok.mean().item())
                valid_gids_bt = torch.where(active_mask, top_gids, torch.full_like(top_gids, -1)).view(B, T, k_global)
                if T > 1:
                    # 二维排序相邻差分单次 kernel 计算序列去重微专家数 (消除逐序列 torch.unique 循环同步)
                    b_sample = min(B, 4)
                    flat_seq = valid_gids_bt[:b_sample].reshape(b_sample, -1)
                    sorted_seq, _ = torch.sort(flat_seq, dim=-1)
                    is_val = sorted_seq >= 0
                    is_new = torch.ones_like(is_val)
                    is_new[:, 1:] = sorted_seq[:, 1:] != sorted_seq[:, :-1]
                    uniq_per_row = (is_val & is_new).float().sum(dim=-1)
                    self.last_unique_experts_per_seq = float(uniq_per_row.mean().item())

                    # 相邻思考步 (步距 stride=8 tokens) 唤醒微专家集合的切换率 (1 - Jaccard Similarity)
                    hop_stride = min(8, T - 1)
                    g_prev = valid_gids_bt[:, :-hop_stride, :]  # (B, T-s, K)
                    g_next = valid_gids_bt[:, hop_stride:, :]   # (B, T-s, K)
                    m_prev = g_prev >= 0
                    m_next = g_next >= 0
                    match_mat = (g_prev.unsqueeze(-1) == g_next.unsqueeze(-2)) & m_prev.unsqueeze(-1) & m_next.unsqueeze(-2)
                    inter_cnt = match_mat.any(dim=-1).float().sum(dim=-1)  # (B, T-s)
                    union_cnt = (m_prev.float().sum(dim=-1) + m_next.float().sum(dim=-1) - inter_cnt).clamp(min=1.0)
                    both_nonempty = (m_prev.any(dim=-1) & m_next.any(dim=-1)).float()
                    jacc_dist = (1.0 - inter_cnt / union_cnt) * both_nonempty
                    self.last_hop_jaccard_dist = float(
                        (jacc_dist.sum() / both_nonempty.sum().clamp(min=1.0)).item()
                    )
                else:
                    self.last_unique_experts_per_seq = self.last_avg_active_slots
                    self.last_hop_jaccard_dist = 0.0

                # 收集低槽位命中 (< 4.0 槽位) 的未知流形球面查询样本，供 Auto-Spawn 孵化新专家包
                if self.training:
                    low_hit_mask = slot_counts_per_tok < 4.0
                    all_low_idx = torch.where(low_hit_mask)[0]
                    self._total_periodic_queries += int(slot_counts_per_tok.numel())
                    self._low_hit_periodic_queries += int(all_low_idx.numel())
                    idx_low = all_low_idx[:128]
                    if idx_low.numel() > 0:
                        q1_low = q1[idx_low].detach()
                        q2_low = q2[idx_low].detach()
                        if self._low_hit_q1_reservoir is None:
                            self._low_hit_q1_reservoir = q1_low
                            self._low_hit_q2_reservoir = q2_low
                        else:
                            self._low_hit_q1_reservoir = torch.cat([self._low_hit_q1_reservoir, q1_low], dim=0)[-512:]
                            self._low_hit_q2_reservoir = torch.cat([self._low_hit_q2_reservoir, q2_low], dim=0)[-512:]

                    # 统计跨包同 Token 共激活频次 (单次批量矩阵乘汇总，消除双重循环 .item() 同步)
                    if len(pkg_routing_results) >= 2:
                        p_indices = torch.tensor(
                            [p_idx for p_idx, _, _, _ in pkg_routing_results],
                            device=top_gids.device,
                            dtype=top_gids.dtype,
                        ).view(1, 1, -1)
                        tok_pkg_mat = (
                            active_mask.unsqueeze(-1) & ((top_gids >> 32).unsqueeze(-1) == p_indices)
                        ).any(dim=1).float()  # (N, P)
                        co_mat = torch.matmul(tok_pkg_mat.t(), tok_pkg_mat).cpu().tolist()
                        pids_now = [pkg.pkg_id for _, pkg, _, _ in pkg_routing_results]
                        for i_a, pa in enumerate(pids_now):
                            self.pkg_token_hit_counts[pa] = self.pkg_token_hit_counts.get(pa, 0.0) + float(co_mat[i_a][i_a])
                            for i_b in range(i_a + 1, len(pids_now)):
                                pb = pids_now[i_b]
                                pair_key = (pa, pb) if pa < pb else (pb, pa)
                                co_val = max(float(co_mat[i_a][i_b]), min(float(co_mat[i_a][i_a]), float(co_mat[i_b][i_b])))
                                self.pkg_coactivation_counts[pair_key] = (
                                    self.pkg_coactivation_counts.get(pair_key, 0.0) + co_val
                                )

        # 定律 3: 零点锚定 Softmax_1 + 严格 L2 能量守恒路由权重 (L2 Energy-Conserving Partition of Unity):
        # 局部流形上的多个正交微专家协同张成连续切平面时，纯 L1 归一化会导致 K_eff 个等权顶点的合成范数按 1/sqrt(K_eff) 塌陷；
        # 严格按 S_1 / S_2 = (sum p_m) / sqrt(sum p_m^2 + eps) 补偿，使球面测地中点与顶点的激活能量严格守恒 (sum w_m^2 = S_1^2)！
        exp_scores = torch.exp(top_excess.clamp(max=20.0)) * active_mask.float()
        denom = 1.0 + exp_scores.sum(dim=-1, keepdim=True)
        w_l1 = exp_scores / denom  # (N, k_global)
        w_sum = w_l1.sum(dim=-1, keepdim=True)
        w_l2 = torch.sqrt(w_l1.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        ortho_gain = (w_sum / w_l2).clamp(min=1.0, max=math.sqrt(float(max(1, k_global))))
        routing_weights = w_l1 * ortho_gain

        # 3. 按包分组执行多头页号去重、VRAM LRU 缓存拉取与 Rank-1 微专家计算
        delta_z = torch.zeros((N, D_BKP), dtype=q_bkp.dtype, device=q_bkp.device)
        div_accum = torch.zeros(N, dtype=torch.float32, device=dev)
        eta_div_accum = torch.zeros(N, dtype=torch.float32, device=dev)
        eta_harm_accum = torch.zeros(N, dtype=torch.float32, device=dev)
        w2_accum = torch.zeros(N, dtype=torch.float32, device=dev)
        max_res_accum = torch.zeros(N, dtype=torch.float32, device=dev)
        self.last_active_trace.clear()
        if do_sync_telemetry:
            self.last_pkg_slot_hits = {pkg.pkg_id: 0 for _, pkg, _, _ in pkg_routing_results}
            self.last_pkg_weight_sum = {pkg.pkg_id: 0.0 for _, pkg, _, _ in pkg_routing_results}
        scale_factor = 1.0 / math.sqrt(D_BKP)

        # 微专家键值点积专用的单位 RMS 球面查询 (保留方向软梯度耦合，仅截断方差缩放分母):
        # 让微专家内部键 u_m 与前段球面查询方向形成双向协同对齐，同时对 rsqrt 分母 detach 杜绝模长膨胀作弊。
        q_sphere_full = torch.stack([q1, q2], dim=2).reshape(N, D_BKP)
        q_sphere_coupled = 0.25 * q_sphere_full + 0.75 * q_sphere_full.detach()
        q_inv_rms = torch.rsqrt(q_sphere_full.detach().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        q_rms = (q_sphere_coupled * q_inv_rms).to(q_bkp.dtype)

        for p_idx, pkg, _, _ in pkg_routing_results:
            pkg_mask = active_mask & ((top_gids >> 32) == p_idx)
            token_indices, slot_positions = torch.where(pkg_mask)
            if token_indices.numel() == 0:
                continue
            local_eids = (top_gids[token_indices, slot_positions] & 0xFFFFFFFF).long()
            local_weights = routing_weights[token_indices, slot_positions]
            if do_sync_telemetry:
                self.last_pkg_slot_hits[pkg.pkg_id] = int(local_eids.numel())
                self.last_pkg_weight_sum[pkg.pkg_id] = float(local_weights.detach().float().sum().item())

            is_pkg_trainable = (
                self.training
                and torch.is_grad_enabled()
                and pkg.mount_mode in (BKPMountMode.RW_SHARED, BKPMountMode.COW_PRIVATE)
            )

            # 按微专家页号去重 (Multi-Head Consensus Deduplication)，同一微专家在单步内仅从缓存拉取一次
            unique_eids, inverse_idx = torch.unique(local_eids, return_inverse=True)
            if is_pkg_trainable:
                with torch.no_grad():
                    # 按单 Token 行维度计算连续流形有效支撑度 (Manifold Support Degree K_eff = exp(H(p))):
                    # 以球面三角剖分最小非退化单形维数 d_s^* == 3 作为 100% 满支撑清晰度定标基准
                    w_pkg = torch.where(pkg_mask, routing_weights, torch.zeros_like(routing_weights))
                    w_row_sum = w_pkg.sum(dim=-1, keepdim=True)
                    act_rows_f = (w_row_sum.squeeze(-1) > 1e-8).float()
                    n_act_rows = act_rows_f.sum().clamp(min=1.0)
                    p_row = w_pkg / w_row_sum.clamp(min=1e-8)
                    ent_per_row = -(p_row * torch.log(p_row.clamp(min=1e-12))).sum(dim=-1)
                    ent_row = (ent_per_row * act_rows_f).sum() / n_act_rows
                    k_eff_support = torch.exp(ent_row)
                    clarity = (k_eff_support / 3.0).clamp(min=0.10, max=1.0)
                    has_act = (act_rows_f.sum() > 0.0).to(pkg.ema_route_clarity.dtype)
                    pkg.ema_route_clarity.mul_(1.0 - 0.05 * has_act).add_(clarity.to(pkg.ema_route_clarity.dtype) * (0.05 * has_act))
                    ones = torch.ones_like(local_eids, dtype=pkg.activation_counts.dtype)
                    pkg.activation_counts.scatter_add_(0, local_eids, ones)

            # 若开启 GPU 页池常驻 (含 RW 与 RO_SYS) 则直接从设备原位微专家池零开销切片，否则走只读 LRU 缓存拉取
            if pkg._gpu_resident_pages and pkg._expert_pages_dev is not None:
                pages_int8 = pkg._expert_pages_dev[unique_eids]
                scales_fp16 = pkg.scales[unique_eids]
            else:
                pages_int8, scales_fp16 = self.vram_cache.fetch_experts(pkg, unique_eids)
            u_dequant = pages_int8[:, 0, :].to(q_bkp.dtype) * scales_fp16[:, 0:1].to(q_bkp.dtype)
            v_dequant = pages_int8[:, 1, :].to(q_bkp.dtype) * scales_fp16[:, 1:2].to(q_bkp.dtype)

            # 原生累加当前激活包在测地圆半径内的亥姆霍兹/霍奇三分解统计量 (0 额外调页，100% 复用 u_dequant, v_dequant 与 L2 路由权重)
            with torch.no_grad():
                u_f = F.normalize(u_dequant.detach().float(), dim=-1)
                v_f = v_dequant.detach().float()
                uv_dot_u = (u_f * v_f).sum(dim=-1)  # (U_uniq,)
                v_norm_u = v_f.norm(dim=-1).clamp_min(1e-6)
                cos2_uv_u = (uv_dot_u / v_norm_u).clamp(-1.0, 1.0).pow(2)
                v_perp_u = (v_f - uv_dot_u.unsqueeze(-1) * u_f).norm(dim=-1)
                wormhole_u = ((v_perp_u > 1.8) | bool(pkg.is_bridge_pkg)).float()

                w_f = local_weights.detach().float()
                w2_f = w_f * w_f
                div_accum.index_add_(0, token_indices, w_f * uv_dot_u[inverse_idx])
                eta_div_accum.index_add_(0, token_indices, w2_f * cos2_uv_u[inverse_idx])
                eta_harm_accum.index_add_(0, token_indices, w2_f * wormhole_u[inverse_idx])
                w2_accum.index_add_(0, token_indices, w2_f)

                q_tok_u = F.normalize(q_flat[token_indices].detach().float(), dim=-1)
                sim_tok = (q_tok_u * u_f[inverse_idx]).sum(dim=-1).clamp(0.0, 1.0)
                max_res_accum.scatter_reduce_(0, token_indices, sim_tok, reduce="amax", include_self=True)

            # 若该包处于可写训练态，对当前步被唤醒的稀疏微专家开启梯度追踪以支持局部悬置微调；
            # 若该包处于 RO_SYS 只读态但上游 Kernel 查询 q_bkp 处于训练态 (如跨核母语流形锚定)，
            # 则仅冻结 u_dequant/v_dequant，保留 local_weights 对 q_bkp 的路由梯度流！
            if is_pkg_trainable:
                u_dequant = u_dequant.detach().requires_grad_(True)
                v_dequant = v_dequant.detach().requires_grad_(True)
                hit_counts = torch.bincount(inverse_idx, minlength=unique_eids.numel()).to(dtype=torch.float32)
                self.last_active_trace[pkg.pkg_id] = (unique_eids, u_dequant, v_dequant, hit_counts)
                w_typed = local_weights.to(q_bkp.dtype).unsqueeze(-1)
            elif q_bkp.requires_grad:
                w_typed = local_weights.to(q_bkp.dtype).unsqueeze(-1)
            else:
                w_typed = local_weights.detach().to(q_bkp.dtype).unsqueeze(-1)

            # q_rms 已在循环外一次性构造完成 (同一查询向量与各知识包共享)

            # 零中间激活留存的分块流式微专家聚合算子 (chunk_size=8192 对齐 A100 40MB L2 Cache，彻底消除数 GB 反向 FP32 峰值)
            delta_pkg = _ChunkedExpertInjectFn.apply(
                q_rms, u_dequant, v_dequant, w_typed, token_indices, inverse_idx, scale_factor, 8192
            )
            delta_z.add_(delta_pkg)

        with torch.no_grad():
            w2_norm = w2_accum.clamp_min(1e-8)
            eta_div_norm = (eta_div_accum / w2_norm).clamp(0.0, 1.0)
            eta_curl_norm = 1.0 - eta_div_norm
            eta_harm_norm = (eta_harm_accum / w2_norm).clamp(0.0, 1.0)
            k_diag = min(8, top_gids.shape[-1])
            self._last_helmholtz_cache = {
                "div_v": div_accum.view(B, T),
                "eta_curl": eta_curl_norm.view(B, T),
                "eta_harmonic": eta_harm_norm.view(B, T),
                "active_experts": (top_gids[:, :k_diag] & 0xFFFFFFFF).view(B, T, k_diag),
                "active_weights": routing_weights[:, :k_diag].detach().float().view(B, T, k_diag),
                "max_resonance": max_res_accum.view(B, T),
            }

        # 逆流形同胚变换 T^{-1}: S -> M，将标准球面上的微专家合成响应映射回原语料流形的各向异性几何空间
        # 注意: 保幅比例因子 (pre_rms / post_rms) 采用 detach() 作为纯标量常数，防止 1/post_rms 求导抵消微专家方向梯度！
        if base_pkg is not None:
            dew_mat = base_pkg.q_dewhiten_mat.detach().to(dtype=delta_z.dtype).clone()
            dz_heads = delta_z.view(N, BKP_NUM_HEADS, 2, BKP_SUBKEY_DIM)
            dz_manifold = torch.einsum("nhsd,hsde->nhse", dz_heads, dew_mat)
            with torch.no_grad():
                pre_rms = torch.sqrt(dz_heads.float().pow(2).mean(dim=-1, keepdim=True) + 1e-8)
                post_rms = torch.sqrt(dz_manifold.float().pow(2).mean(dim=-1, keepdim=True) + 1e-8)
                rms_gain = (pre_rms / post_rms).to(dz_manifold.dtype)
            delta_z = (dz_manifold * rms_gain).reshape(N, D_BKP)

        # 严格后置更新: 待前向路由、微专家点积及逆流形变换全部完成后再更新 EMA 中心与流形度规，确保当前步前向 100% 只读历史统计量
        if collect_center_stats and base_pkg is not None:
            if getattr(base_pkg, "_pending_cone_batch_mean", None) is not None:
                base_pkg.update_query_cone_center(base_pkg._pending_cone_batch_mean)
                base_pkg._pending_cone_batch_mean = None
            if getattr(base_pkg, "_pending_batch_cov", None) is not None:
                base_pkg.update_query_manifold_metric(base_pkg._pending_batch_cov)
                base_pkg._pending_batch_cov = None
            if getattr(base_pkg, "_pending_post_cone_batch_mean", None) is not None:
                base_pkg.update_query_post_cone_center(base_pkg._pending_post_cone_batch_mean)
                base_pkg._pending_post_cone_batch_mean = None
            for other_pkg in self.packages.values():
                if other_pkg is not base_pkg:
                    other_pkg._copy_manifold_stats_from(base_pkg)

        return delta_z.view(B, T, D_BKP)

    def forward(self, q_bkp: torch.Tensor) -> torch.Tensor:
        return self.query_and_inject(q_bkp)

    # --------------------------------------------------------------------------
    # 稀疏微专家原位优化器步进 (含单义相干纯度 R_{a,b} 动量更新与脏数据批次熔断)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def step_sparse_bkp_update(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        purity_drop_abort_threshold: float = 0.4,
        grad_scale: float = 1.0,
        collinear_core_lock_strength: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        在累积梯度 backward() 完成后调用:
        - GPU 驻留训练模式 (K_p <= 512): 全程 0 PCIe 传输原位完成向量化 AdamW 更新与 INT8 随机舍入量化 (<5ms/步)
        - 超大知识包流式模式 (K_p >= 1024): 按 4096 页分块流式送入 GPU 更新后写回 OS mmap 页
        """
        if len(self.last_active_trace) > 0:
            self.stash_microbatch_trace()

        commit_status: Dict[str, bool] = {}
        inv_scale = 1.0 / max(grad_scale, 1e-30)
        chunk_sz = 4096

        for pkg in self.packages.values():
            pkg_id = pkg.pkg_id
            traces = self._pending_grad_traces.get(pkg_id, [])
            if len(traces) == 0:
                if pkg._grad_dirty_mask.numel() > 0:
                    pkg._grad_dirty_mask.zero_()
                commit_status[pkg_id] = False
                continue

            dev = pkg.scales.device
            if len(traces) == 1:
                unique_eids_dev, g_all_dev, hits_all_dev = traces[0]
                g_all_dev = g_all_dev.to(device=dev, dtype=torch.float32)
                hits_all_dev = hits_all_dev.to(device=dev, dtype=torch.float32)
            else:
                cat_eids = torch.cat([t[0].to(dev) for t in traces], dim=0)
                cat_g = torch.cat([t[1].to(device=dev, dtype=torch.float32) for t in traces], dim=0)
                cat_hits = torch.cat([t[2].to(device=dev, dtype=torch.float32) for t in traces], dim=0)
                unique_eids_dev, inv_u = torch.unique(cat_eids, return_inverse=True)
                n_u_all = unique_eids_dev.numel()
                g_all_dev = torch.zeros((n_u_all, 2, D_BKP), dtype=torch.float32, device=dev)
                hits_all_dev = torch.zeros(n_u_all, dtype=torch.float32, device=dev)
                g_all_dev.index_add_(0, inv_u, cat_g)
                hits_all_dev.index_add_(0, inv_u, cat_hits)

            if pkg._frozen_expert_mask_cpu is not None and pkg._frozen_expert_mask_cpu.numel() == pkg.num_experts:
                frozen_dev = pkg._frozen_expert_mask_cpu.to(dev)
                keep_dev = ~frozen_dev[unique_eids_dev]
                unique_eids_dev = unique_eids_dev[keep_dev]
                if unique_eids_dev.numel() == 0:
                    if pkg._grad_dirty_mask.numel() > 0:
                        pkg._grad_dirty_mask.zero_()
                    commit_status[pkg_id] = False
                    continue
                g_all_dev = g_all_dev[keep_dev]
                hits_all_dev = hits_all_dev[keep_dev]

            num_u = unique_eids_dev.numel()
            buf_dev = pkg._sparse_m1_cpu.device
            has_master = pkg._master_uv_fp32 is not None and pkg._master_uv_fp32.numel() > 0
            need_cpu_ids = (buf_dev.type == "cpu") or (
                has_master and pkg._master_uv_fp32.device.type == "cpu"
            ) or (not has_master and pkg._expert_pages_dev is None)
            unique_eids_cpu = unique_eids_dev.cpu() if need_cpu_ids else None
            aborted = False

            for s_idx in range(0, num_u, chunk_sz):
                c_ids_dev = unique_eids_dev[s_idx : s_idx + chunk_sz]
                c_ids_cpu = unique_eids_cpu[s_idx : s_idx + chunk_sz] if unique_eids_cpu is not None else None
                c_ids = c_ids_cpu if buf_dev.type == "cpu" else c_ids_dev

                g_c = g_all_dev[s_idx : s_idx + chunk_sz]
                hits_c = hits_all_dev[s_idx : s_idx + chunk_sz].clamp(min=1.0).view(-1, 1, 1)
                if grad_scale != 1.0:
                    g_c.mul_(inv_scale)
                # 1. 频次脱钩均值归一化 (Frequency-Decoupled Mean Update) + 零同步数值安全清洗:
                #    将微专家累积梯度严格除以当前步命中次数 H_m，防止高频重复语料拉偏微专家方向或引爆模长
                g_c = torch.nan_to_num(g_c / hits_c, nan=0.0, posinf=0.0, neginf=0.0)

                prev_t = torch.index_select(pkg._sparse_steps_cpu, 0, c_ids) + 1
                pkg._sparse_steps_cpu.index_copy_(0, c_ids, prev_t)
                t_f = prev_t.to(device=dev, dtype=torch.float32).view(-1, 1, 1)

                m1 = torch.index_select(pkg._sparse_m1_cpu, 0, c_ids).to(device=dev, dtype=torch.float32)
                m2 = torch.index_select(pkg._sparse_m2_cpu, 0, c_ids).to(device=dev, dtype=torch.float32)

                m1_next = beta1 * m1 + (1.0 - beta1) * g_c
                m2_next = beta2 * m2 + (1.0 - beta2) * g_c.pow(2)

                pkg._sparse_m1_cpu.index_copy_(0, c_ids, m1_next.to(device=buf_dev, dtype=pkg._sparse_m1_cpu.dtype))
                pkg._sparse_m2_cpu.index_copy_(0, c_ids, m2_next.to(device=buf_dev, dtype=pkg._sparse_m2_cpu.dtype))

                m1_hat = m1_next / (1.0 - (beta1 ** t_f))
                m2_hat = m2_next / (1.0 - (beta2 ** t_f))

                # 2. 球面方向相干纯度 R_{a,b} 与二阶张力保护 (First-Order Coherence + Second-Order Tension)
                g_flat = g_c.flatten(1)
                m1_flat = m1_next.flatten(1)
                cos_coh = (g_flat * m1_flat).sum(dim=1) / (
                    g_flat.norm(p=2, dim=1) * m1_flat.norm(p=2, dim=1) + 1e-12
                )
                inst_purity = ((cos_coh + 1.0) * 0.5).clamp(0.01, 1.0)
                pkg.purity_r[c_ids_dev] = 0.85 * pkg.purity_r[c_ids_dev] + 0.15 * inst_purity

                has_master = pkg._master_uv_fp32 is not None and pkg._master_uv_fp32.numel() > 0
                if has_master:
                    m_dev = pkg._master_uv_fp32.device
                    m_ids = c_ids_cpu if m_dev.type == "cpu" else c_ids_dev.to(device=m_dev)
                    w_c = torch.index_select(pkg._master_uv_fp32, 0, m_ids).to(device=dev, dtype=torch.float32)
                elif pkg._expert_pages_dev is not None:
                    w_int8 = pkg._expert_pages_dev[c_ids_dev].float()
                    w_scale = pkg.scales[c_ids_dev].float().unsqueeze(-1)
                    w_c = w_int8 * w_scale
                else:
                    w_int8 = pkg.expert_pages_int8[c_ids_cpu].to(device=dev, dtype=torch.float32)
                    w_scale = pkg.scales[c_ids_dev].float().unsqueeze(-1)
                    w_c = w_int8 * w_scale

                # 3. 零超参数后验预测信用凋亡判据 (Closed-Form t-Statistic Apoptosis):
                #    C_m = - <g_v, v_val> / (||g_v|| * ||v_val||)
                #    当显著性统计量 sqrt(H_m) * C_m < -tau_z^*(N_grid) 且处于多义失相干态 (R_m < 1/sqrt(2)) 时，
                #    按网格分辨率闭式生存因子 (1 - 1/N_grid) 原位收缩其载荷，零人工经验魔数！
                gv_cur = g_c[:, 1, :]
                vv_cur = w_c[:, 1, :]
                credit_cos = -(gv_cur * vv_cur).sum(dim=-1) / (
                    gv_cur.norm(p=2, dim=-1) * vv_cur.norm(p=2, dim=-1) + 1e-12
                )
                h_m_cur = pkg.activation_counts[c_ids_dev].float().clamp(min=1.0)
                r_m_cur = pkg.purity_r[c_ids_dev].float()
                tau_z_pkg = closed_form_gumbel_threshold(float(pkg.grid_k))
                bad_wake = (torch.sqrt(h_m_cur) * credit_cos < -tau_z_pkg) & (r_m_cur < SIMPLEX_2_GEODESIC_PROJ)
                closed_decay_val = 1.0 - (1.0 / max(4.0, float(pkg.grid_k)))
                apoptosis_decay = torch.where(
                    bad_wake, torch.full_like(credit_cos, closed_decay_val), torch.ones_like(credit_cos)
                )

                rms_m2 = torch.sqrt(m2_hat.mean(dim=-1, keepdim=True) + 1e-16)
                step_denom = 0.5 * torch.sqrt(m2_hat + 1e-16) + 0.5 * rms_m2 + 1e-8
                warmup_scale = (t_f / 2.0).clamp(max=1.0)
                step_delta = (lr * warmup_scale) * (m1_hat / step_denom)

                # 4. 零超参数球面 James-Stein 闭式共线主轴保护 (Spherical James-Stein Collinear Shrinkage):
                #    根据高维球面随机游走瑞利分布，微专家历史梯度的纯正交噪声期望满足 E[H_m * R_m^2] = 1；
                #    因此主轴信号占总变异的无偏 James-Stein 最优收缩锁定强度精确为:
                #        gamma_m^* = max(0, (H_m * R_m^2 - 1) / (H_m - 1)) in [0, 1]
                #    新微专家 (H_m <= 1) gamma_m^* = 0 完全自由探索；成熟单义微专家自动锁定主轴并将正交微扰分流给伴随微专家！
                vv_norm = vv_cur.norm(p=2, dim=-1, keepdim=True)
                v_hat = vv_cur / (vv_norm + 1e-12)
                dv = step_delta[:, 1, :]
                dv_para = (dv * v_hat).sum(dim=-1, keepdim=True) * v_hat
                dv_perp = dv - dv_para
                h_js = h_m_cur.unsqueeze(-1)
                r_js = r_m_cur.unsqueeze(-1)
                js_numer = (h_js * r_js.pow(2) - 1.0).clamp(min=0.0)
                js_denom = (h_js - 1.0).clamp(min=1.0)
                gamma_js = (js_numer / js_denom).clamp(0.0, 1.0) * (h_js > 1.0).float() * (vv_norm > 1e-3).float()
                if collinear_core_lock_strength is not None:
                    lock_scale = min(1.0, max(0.0, float(collinear_core_lock_strength)))
                else:
                    attr_lock = getattr(self, "collinear_core_lock_strength", None)
                    # 默认联合演化模式下 lock_scale = 0.0 (零正交衰减，由切向复共轭叶状分裂 Auto-Mitosis 无损承载分叉细节)
                    lock_scale = min(1.0, max(0.0, float(attr_lock))) if attr_lock is not None else 0.0
                gamma_lock = gamma_js * lock_scale
                step_delta[:, 1, :] = dv_para + (1.0 - gamma_lock) * dv_perp

                updated_w = w_c - step_delta

                # 5. 定律 2: 1-Jet 三元微专家法向/切向能量守恒闭式分解 (u_m = alpha_m * c_m + u_{m,perp})
                #    在 3-单纯形面片 (cos(theta_edge) = 1/sqrt(3)) 内保证一阶切向倾角绝不翻转零阶法向门控符号的临界条件为:
                #        alpha_m / ||u_{m,perp}||_2 >= sqrt(2)  <==>  alpha_m^* = sqrt(2/3), ||u_{m,perp}^*||_2 <= sqrt(1/3)
                u_raw = F.normalize(updated_w[:, 0, :], dim=-1) * math.sqrt(D_BKP)
                u_cb_norm, v_cb_norm = pkg.normalized_codebooks()
                r_m = torch.div(c_ids_dev, pkg.grid_k, rounding_mode="floor")
                c_m_idx = c_ids_dev % pkg.grid_k
                c_u1 = u_cb_norm[:, r_m, :].permute(1, 0, 2).float()
                c_v2 = v_cb_norm[:, c_m_idx, :].permute(1, 0, 2).float()
                c_anchor = torch.stack([c_u1, c_v2], dim=2).reshape(-1, D_BKP) * math.sqrt(float(BKP_SUBKEY_DIM))
                alpha_m = (u_raw * c_anchor).sum(dim=-1, keepdim=True) / float(D_BKP)
                u_perp = u_raw - alpha_m * c_anchor
                alpha_anchored = torch.clamp(alpha_m, min=JET_NORMAL_ALPHA_STAR)
                max_perp_norm = (JET_TANGENT_NORM_STAR / JET_NORMAL_ALPHA_STAR) * alpha_anchored * math.sqrt(float(D_BKP))
                perp_norm = u_perp.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                u_perp_bounded = torch.where(perp_norm > max_perp_norm, u_perp * (max_perp_norm / perp_norm), u_perp)
                u_new = F.normalize(alpha_anchored * c_anchor + u_perp_bounded, dim=-1) * math.sqrt(D_BKP)

                v_new = updated_w[:, 1, :] * apoptosis_decay.unsqueeze(-1)
                v_norm = v_new.norm(p=2, dim=-1, keepdim=True)
                max_v_norm = 0.30 * math.sqrt(D_BKP)
                v_new = torch.where(v_norm > max_v_norm, v_new * (max_v_norm / (v_norm + 1e-12)), v_new)

                pkg.write_experts_to_mmap_pages(c_ids_dev, u_new, v_new)

            if pkg._grad_dirty_mask.numel() > 0:
                pkg._grad_dirty_mask.zero_()

            if aborted:
                commit_status[pkg_id] = False
                continue

            if not pkg._gpu_resident_pages:
                self.vram_cache.invalidate_or_refresh(pkg, unique_eids_dev)
            self._step_counter = getattr(self, "_step_counter", 0) + 1
            if self._step_counter == 1 or (self._step_counter % 50 == 0):
                pkg.update_quality_index()
            commit_status[pkg_id] = True

        self._pending_grad_traces.clear()
        self.last_active_trace.clear()
        return commit_status

    # --------------------------------------------------------------------------
    # 4.2.2 节: 微专家向量的三种直接外科手术修改算子 (Direct Vector Editing)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def surgical_silence_expert(self, pkg_id: str, expert_id: int) -> None:
        """算子 1: 定点静默与擦除 (Targeted Silencing) —— 直接将目标槽位输出向量 v_val 置 0"""
        pkg: BKPPackage = self.packages[self._mod_key(pkg_id)]
        pkg._sync_device_pages_to_cpu_buffer()
        eid_tensor = torch.tensor([expert_id], dtype=torch.long, device=pkg.scales.device)
        u_cur = pkg.expert_pages_int8[expert_id, 0:1, :].float() * pkg.scales[expert_id, 0].cpu().float()
        v_zero = torch.zeros_like(u_cur)
        pkg.write_experts_to_mmap_pages(eid_tensor, u_cur, v_zero)
        self.vram_cache.invalidate_or_refresh(pkg, eid_tensor)

    @torch.no_grad()
    def surgical_inject_rank1_expert(
        self,
        pkg_id: str,
        row_a: int,
        col_b: int,
        q_trigger: torch.Tensor,
        delta_h_target: torch.Tensor,
    ) -> int:
        """算子 2: 闭式秩-1 键值直接写入 (Closed-Form Rank-1 Injection)"""
        pkg: BKPPackage = self.packages[self._mod_key(pkg_id)]
        pkg._sync_device_pages_to_cpu_buffer()
        expert_id = row_a * pkg.grid_k + col_b
        q_flat = q_trigger.view(1, D_BKP)
        q1, q2 = self._standardize_subqueries(q_flat)

        # 将目标槽位的行列子键对齐至触发查询方向
        pkg.u_codebook.data[:, row_a, :] = F.normalize(q1.squeeze(0).to(pkg.u_codebook.device), dim=-1)
        pkg.v_codebook.data[:, col_b, :] = F.normalize(q2.squeeze(0).to(pkg.v_codebook.device), dim=-1)

        u_vec = F.normalize(q_flat.float(), dim=-1) * math.sqrt(D_BKP)
        v_vec = delta_h_target.view(1, D_BKP).float()
        eid_tensor = torch.tensor([expert_id], dtype=torch.long, device=pkg.scales.device)
        pkg.write_experts_to_mmap_pages(eid_tensor, u_vec, v_vec)
        self.vram_cache.invalidate_or_refresh(pkg, eid_tensor)
        return expert_id

    @torch.no_grad()
    def surgical_debias_orthogonal(self, pkg_id: str, expert_id: int, bias_dir: torch.Tensor) -> None:
        """算子 3: 正交子空间去偏投影 (Orthogonal Subspace De-biasing)"""
        pkg: BKPPackage = self.packages[self._mod_key(pkg_id)]
        pkg._sync_device_pages_to_cpu_buffer()
        d_hat = F.normalize(bias_dir.view(1, D_BKP).float().cpu(), dim=-1)
        u_cur = pkg.expert_pages_int8[expert_id, 0:1, :].float() * pkg.scales[expert_id, 0].cpu().float()
        v_cur = pkg.expert_pages_int8[expert_id, 1:2, :].float() * pkg.scales[expert_id, 1].cpu().float()

        # v_new = v - (v . d_hat) * d_hat
        proj = (v_cur * d_hat).sum(dim=-1, keepdim=True) * d_hat
        v_debiased = v_cur - proj

        eid_tensor = torch.tensor([expert_id], dtype=torch.long, device=pkg.scales.device)
        pkg.write_experts_to_mmap_pages(eid_tensor, u_cur, v_debiased)
        self.vram_cache.invalidate_or_refresh(pkg, eid_tensor)

    @torch.no_grad()
    def reset_coactivation_stats(self) -> None:
        """清空周期性跨包共激活计数与低命中查询蓄水池，确保下一轮拓扑巡检仅反映最新窗口的流形状态"""
        self.pkg_coactivation_counts.clear()
        self.pkg_token_hit_counts.clear()
        self._low_hit_q1_reservoir = None
        self._low_hit_q2_reservoir = None
        self._low_hit_periodic_queries = 0
        self._total_periodic_queries = 0

    @torch.no_grad()
    def spawn_specialist_package_from_queries(
        self,
        new_pkg_id: str,
        grid_k: int = 64,
        mmap_file_path: Optional[str] = None,
        mount_mode: BKPMountMode = BKPMountMode.RW_SHARED,
        init_scale_v: float = 1e-5,
    ) -> BKPPackage:
        """
        自治拓扑算子 Auto-Spawn (球面未知流形新包孵化):
        从收集到的低槽位命中球面查询池 (_low_hit_q1_reservoir, _low_hit_q2_reservoir) 中，
        通过最远点球面分散采样提取 grid_k 个最具代表性的正交方向作为新包的初始子键码本，
        并继承基准包的查询锥中心与球面度规，初生页 v_val=0 (零步零扰动)，开启 scale_v=init_scale_v 准备吸收新领域残差！
        """
        ref_pkg = self._get_reference_manifold_pkg()
        dev = ref_pkg.u_codebook.device if ref_pkg is not None else self.vram_cache.cache_pages_int8.device

        new_pkg = BKPPackage(
            pkg_id=new_pkg_id,
            grid_k=grid_k,
            mmap_file_path=mmap_file_path,
            mount_mode=mount_mode,
            is_base_pkg=(len(self.packages) == 0),
            device=dev,
        )
        if ref_pkg is not None:
            new_pkg._copy_manifold_stats_from(ref_pkg)

        if (
            self._low_hit_q1_reservoir is not None
            and self._low_hit_q2_reservoir is not None
            and self._low_hit_q1_reservoir.shape[0] >= 8
        ):
            q1_pool = F.normalize(self._low_hit_q1_reservoir.to(dev).float(), dim=-1)  # (M, 8, 128)
            q2_pool = F.normalize(self._low_hit_q2_reservoir.to(dev).float(), dim=-1)  # (M, 8, 128)
            m_pool = q1_pool.shape[0]
            # 对每个头采用贪心最远点球面采样 (Farthest Point Sampling on S^{127}) 选出覆盖最广的 grid_k 个方向
            for h_i in range(BKP_NUM_HEADS):
                q1_h = q1_pool[:, h_i, :]
                q2_h = q2_pool[:, h_i, :]
                if m_pool >= grid_k:
                    perm = torch.randperm(m_pool, device=dev)[:grid_k]
                    u_sel = q1_h[perm]
                    v_sel = q2_h[perm]
                else:
                    rep = (grid_k + m_pool - 1) // m_pool
                    u_sel = F.normalize(
                        q1_h.repeat(rep, 1)[:grid_k] + 0.08 * torch.randn(grid_k, BKP_SUBKEY_DIM, device=dev),
                        dim=-1,
                    )
                    v_sel = F.normalize(
                        q2_h.repeat(rep, 1)[:grid_k] + 0.08 * torch.randn(grid_k, BKP_SUBKEY_DIM, device=dev),
                        dim=-1,
                    )
                u_centered = F.normalize(u_sel - u_sel.mean(dim=0, keepdim=True), dim=-1)
                v_centered = F.normalize(v_sel - v_sel.mean(dim=0, keepdim=True), dim=-1)
                new_pkg.u_codebook.data[h_i].copy_(u_centered)
                new_pkg.v_codebook.data[h_i].copy_(v_centered)
            new_pkg.codebook_pre_centered.fill_(1)
            new_pkg._codebook_pre_centered_cached = 1
            self._low_hit_q1_reservoir = None
            self._low_hit_q2_reservoir = None
            self._low_hit_periodic_queries = 0
            self._total_periodic_queries = 0

        # 激活微专家页供反向传播写入 (v_val 初始为 0，挂载瞬间零抖动)
        new_pkg.enable_empty_pages_for_training(init_scale_v=init_scale_v, freeze_existing_pages=False)
        self.mount_package(new_pkg, pin_to_l1=True)
        return new_pkg


# ==============================================================================
# 5. 自检测试 (验证 4KB 对齐、闭式极值路由、VRAM 只读缓存与零误差拓扑扩容/合并)
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    manager = BKPManager(cache_slots=1024)

    # 1. 创建常驻 Base.bkp (64x64) 与可写会话包 Context.bkp (64x64)
    base_pkg = BKPPackage("Base.bkp", grid_k=64, mount_mode=BKPMountMode.RO_SYS, is_base_pkg=True)
    ctx_pkg = BKPPackage("Context.bkp", grid_k=64, mount_mode=BKPMountMode.RW_SHARED, is_base_pkg=False)
    assert base_pkg.expert_pages_int8.shape == (4096, 2, 2048)
    assert base_pkg.expert_pages_int8[0].numel() * base_pkg.expert_pages_int8.element_size() == PAGE_SIZE

    manager.mount_package(base_pkg)
    manager.mount_package(ctx_pkg)

    # 2. 用外科手术算子向 Context.bkp 注入一条触发直觉，验证前向检索、反向稀疏梯度与 mmap 原位写回
    q_sample = torch.randn(2, 4, D_BKP)
    target_impulse = torch.randn(D_BKP) * 0.5
    injected_eid = manager.surgical_inject_rank1_expert("Context.bkp", 3, 7, q_sample[0, 0], target_impulse)

    manager.train()
    out_delta = manager.query_and_inject(q_sample)
    loss = F.mse_loss(out_delta[0, 0], target_impulse)
    loss.backward()
    commit_res = manager.step_sparse_bkp_update(lr=1e-2)

    # 3. 验证四象限 (a+b)^2 倍增扩容前向误差严格为 0.0000
    manager.eval()
    with torch.no_grad():
        out_before = manager.query_and_inject(q_sample)
        ctx_pkg.expand_quadrant_2x()
        assert ctx_pkg.grid_k == 128
        out_after_expand = manager.query_and_inject(q_sample)
        expand_diff = (out_before - out_after_expand).abs().max().item()

    print(f"[PASS] 4KB Page Size = {PAGE_SIZE}B | Injected Expert ID = {injected_eid}")
    print(f"[PASS] Sparse Update Committed = {commit_res} | Quality Q_p = {ctx_pkg.quality_q.item():.4f}")
    print(f"[PASS] VRAM Cache Hit Rate (2nd Pass) = {manager.vram_cache.total_hits}/{manager.vram_cache.total_queries}")
    print(f"[PASS] 4-Quadrant (a+b)^2 Expansion Max Forward Diff = {expand_diff:.6f}")