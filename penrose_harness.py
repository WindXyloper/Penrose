"""
Penrose v1.0 — Penrose Harness (`penrose_harness.py`)
=====================================================
流形块调度器、全息波包 I/O 换能总线（Holographic Wavepacket Transducer）、
硬件级三态总线分支控制器（分支 A 旋度静默闭环 / 分支 B 负散度坍缩吐字 / 分支 C 球面并行训练）
以及免清洗拖拽式 Kaczmarz 训练与自治森林演化引擎。
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from penrose_pmc import (
    D_BKP,
    BKP_NUM_HEADS,
    JET_NORMAL_ALPHA_STAR,
    AMBIENT_NOISE_FLOOR,
    GUMBEL_SINK_DIV_STAR,
    FUNNEL_RESONANCE_APERTURE,
    closed_form_gumbel_threshold,
    PMCMountMode,
    PMCRoutingConfig,
    PMCPackage,
    PMCManager,
    HelmholtzFieldDiagnostics,
)
from penrose_mpu import (
    PenroseMPUConfig,
    PenroseMPU,
    MPUStepOutput,
)


def build_walsh_hadamard_matrix(n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    递归构造精确的 n x n 阶标准正交沃尔什-阿达玛矩阵 H_n in {-1/sqrt(n), +1/sqrt(n)}^{n x n}，
    满足 H_n^T @ H_n = I_n（0 可学习参数，任意两不同行内积严格为 0）。
    要求 n 为 2 的整次幂（如 256 或 2048）。
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} 必须为 2 的整次幂"
    h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    curr = 1
    while curr < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        curr *= 2
    return h / math.sqrt(float(n))


class BusBranchState(Enum):
    """Module 5 出口的硬件级三态总线分支状态"""
    BRANCH_A_INTERNAL_CURL_LOOP = "branch_a_curl_loop"      # 内部旋度闭环 / 无声推理潜航
    BRANCH_B_NEGATIVE_DIV_EMIT = "branch_b_sink_emit"       # 外部负散度坍缩 / 符号波包吐出
    BRANCH_C_PARALLEL_TRAINING = "branch_c_parallel_train"  # 球面 ResNet 并行训练态（零解码直算弦长 Loss）


class UnicodePhaseClass(Enum):
    """第一层零参数物理相变边界类别"""
    LATIN_ALNUM = 1     # 拉丁字母、数字、下划线 (a-z, A-Z, 0-9, _)
    CJK_IDEOGRAPH = 2   # CJK 表意文字连续流
    SYNTAX_PUNCT = 3    # 语法算子与标点边界 ({, }, :, ", (, ), ., 。, ， 等)
    WHITESPACE = 4      # 空白与控制符


class HolographicWavepacketTransducer(nn.Module):
    """
    Penrose Harness 外围 I/O 总线换能器（0 可学习参数）：
    1. 微观层面（2048 = 8 x 256 沃尔什-阿达玛正交谐波折叠 + 非阿贝尔位置相位旋转）：
       - 单字节 b in {0..255} 映射为 H_256 的第 b 行向量 h_256(b) in R^{256}（任意两不同字节内积恒为 0）；
       - 8 个 256 维谐波槽位配合非阿贝尔黄金角 Givens 相位旋转 R_pos^{(i)} 与全局 H_2048 混频：
             u^{(i)}(b^{(i)}) = H_2048 * R_pos^{(i)} * (e_{slot(i mod 8)} (x) h_256(b^{(i)}))
             q(tau) = (1 / sqrt(k)) * sum_{i=1}^k u^{(i)}(b^{(i)}) in S^{2047}
       - 解码时利用正交逆变换与公理 2 的 Gumbel 极值孔径阈值 tau_z*(256) 自动检测有效字节数 k，
         无需任何结束符或词表矩阵，100% 零串扰还原 UTF-8 字节流！
    2. 宏观层面（双层物理相变 + PMC 负散度漏斗动态捕获）：
       - 第一层物理相变门自动切开 JSON/代码格式符（如 `{text: "`）；
       - 第二层实时试探 PMC 负散度吸引盆，将多字词自动吸入同一个 2048 维波包漏斗。
    """

    def __init__(self, max_bytes_per_packet: int = 16, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.max_slots = int(max_bytes_per_packet)
        self.num_base_channels = 8
        self.channel_dim = 256
        assert self.num_base_channels * self.channel_dim == D_BKP

        # 1. 构造 256x256 单字节正交阿达玛基 H_256 与 8x8 谐波槽位混频阿达玛基 H_8
        #    注：H_2048 = H_8 (x) H_256，使用克罗内克结构可在 O(d) 内完成全局 2048 维全息混频！
        h_256 = build_walsh_hadamard_matrix(256, device=device)  # (256, 256)
        h_8 = build_walsh_hadamard_matrix(8, device=device)      # (8, 8)
        self.register_buffer("h_256", h_256, persistent=False)
        self.register_buffer("h_8", h_8, persistent=False)

        # 2. 构造非阿贝尔位置相位旋转算子 R_pos^{(i)}（当字节数超过 8 时通过黄金角正交 Givens 旋转区分周期）
        #    每个槽位 i in [0, max_slots) 在 128 对二维平面上施加黄金角 theta_i = (i // 8) * pi * (sqrt(5)-1)/2
        golden_angle = math.pi * (math.sqrt(5.0) - 1.0) * 0.5
        cos_list, sin_list = [], []
        for slot_i in range(self.max_slots):
            cycle = slot_i // 8
            if cycle == 0:
                cos_list.append(torch.ones(128, dtype=torch.float32, device=device))
                sin_list.append(torch.zeros(128, dtype=torch.float32, device=device))
            else:
                freq = torch.arange(1, 129, dtype=torch.float32, device=device) * golden_angle * float(cycle)
                cos_list.append(torch.cos(freq))
                sin_list.append(torch.sin(freq))
        self.register_buffer("pos_cos", torch.stack(cos_list, dim=0), persistent=False)  # (max_slots, 128)
        self.register_buffer("pos_sin", torch.stack(sin_list, dim=0), persistent=False)  # (max_slots, 128)

        # 3. 公理 2：256 维正交字节基的 Gumbel 极值噪声孔径阈值
        #    随机单位向量在 256 个正交基上的最大投影标准差为 1/sqrt(256) = 1/16，
        #    故信噪比门限为 tau_slot* = tau_z*(256) / sqrt(256) ≈ 3.405 / 16 ≈ 0.2128
        self.gumbel_tau_256: float = closed_form_gumbel_threshold(256.0) / math.sqrt(float(D_BKP))


    @staticmethod
    def _apply_givens_256(vec_256: torch.Tensor, cos_128: torch.Tensor, sin_128: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        x1 = vec_256[..., :128]
        x2 = vec_256[..., 128:]
        if not inverse:
            y1 = x1 * cos_128 - x2 * sin_128
            y2 = x2 * cos_128 + x1 * sin_128
        else:
            y1 = x1 * cos_128 + x2 * sin_128
            y2 = x2 * cos_128 - x1 * sin_128
        return torch.cat([y1, y2], dim=-1)

    @torch.no_grad()
    def slot_unit_vector(self, slot_idx: int, byte_val: int) -> torch.Tensor:
        """
        构造第 slot_idx 个字节槽位取字节值 byte_val in {0..255} 时的 2048 维单位基向量 u^{(i)}(b)：
            u^{(i)}(b) = (H_8 (x) I_256) * (e_{slot mod 8} (x) R_pos^{(i)} h_256(b)) in S^{2047}
        因为 H_8 的每一列元素全为 ±1/sqrt(8)，该字节的能量严格均匀铺满全部 8 个 256 维注意力头！
        """
        b_idx = int(byte_val) & 0xFF
        s_idx = int(slot_idx) % self.max_slots
        ch = s_idx % 8
        h_vec = self.h_256[b_idx]  # (256,)
        rot_vec = self._apply_givens_256(h_vec, self.pos_cos[s_idx], self.pos_sin[s_idx], inverse=False)
        # 外积乘上 H_8 的第 ch 行，使 256 维谐波均匀分布到 8 个头 (8, 256) -> (2048,)
        h8_row = self.h_8[ch].unsqueeze(-1)  # (8, 1)
        u_2048 = (h8_row * rot_vec.unsqueeze(0)).reshape(D_BKP)
        return u_2048

    @torch.no_grad()
    def encode_bytes_to_wavepacket(self, raw_bytes: bytes) -> torch.Tensor:
        """
        将 1 ~ k 个 UTF-8 字节无损正交折叠为单个 2048 维球面单位波包向量：
            q(tau) = Normalize( sum_{i=1}^k u^{(i)}(b^{(i)}) ) in S^{2047}
        """
        dev = self.h_256.device
        if len(raw_bytes) == 0:
            return self.slot_unit_vector(0, 0)
        k = min(len(raw_bytes), self.max_slots)
        acc = torch.zeros(D_BKP, dtype=torch.float32, device=dev)
        for i in range(k):
            acc = acc + self.slot_unit_vector(i, raw_bytes[i])
        return F.normalize(acc, dim=-1)

    @torch.no_grad()
    def encode_string_to_wavepacket(self, text: str) -> torch.Tensor:
        return self.encode_bytes_to_wavepacket(text.encode("utf-8"))

    @torch.no_grad()
    def decode_wavepacket_to_bytes(self, q_vec: torch.Tensor) -> Tuple[bytes, List[float]]:
        """
        全息逆变换直接解码（0 参数，无需 LM Head 词表，无需结束符）：
        对 q_vec in S^{2047} 先左乘 H_8^T 解混频还原 8 个正交谐波通道，
        再对每个位置槽位 i 施加逆 Givens 旋转与 H_256^T 内积，
        用公理 2 的 Gumbel 极值孔径阈值 tau_z*(256) 自动检测有效字节长度 k！
        """
        q_unit = F.normalize(q_vec.detach().float().view(D_BKP), dim=-1)
        # 1. H_8^T 解混频: (8, 8) @ (8, 256) -> (8, 256)
        q_channels = torch.matmul(self.h_8.t(), q_unit.view(8, 256))

        decoded_bytes = bytearray()
        slot_peaks: List[float] = []

        # 公理 2 闭式自适应孔径：满载波包最低有效槽位振幅为 1/sqrt(max_slots) = 0.25，
        # 空槽位背景极值噪声上确界为 tau_z*(256)/sqrt(2048) ≈ 0.07525，取两者精确算术中点 (零经验魔数)
        min_signal_amp = 1.0 / math.sqrt(float(self.max_slots))
        tau_aperture = 0.5 * (min_signal_amp + self.gumbel_tau_256)

        for i in range(self.max_slots):
            ch = i % 8
            ch_vec = q_channels[ch]
            # 逆相位旋转 R_pos^{(i)T}
            unrot_vec = self._apply_givens_256(ch_vec, self.pos_cos[i], self.pos_sin[i], inverse=True)
            # 在 256 个正交字节基上的投影响应: (256,)
            scores = torch.matmul(self.h_256, unrot_vec)
            max_val, best_b = torch.max(scores, dim=-1)
            peak = float(max_val.item())
            if peak < tau_aperture:
                break
            decoded_bytes.append(int(best_b.item()))
            slot_peaks.append(peak)
            # 从该通道减去已解码出的正交分量，确保后续超过 8 字节的黄金角移相层无残留干扰
            q_channels[ch] = q_channels[ch] - peak * self._apply_givens_256(
                self.h_256[int(best_b.item())], self.pos_cos[i], self.pos_sin[i], inverse=False
            )

        return bytes(decoded_bytes), slot_peaks

    @torch.no_grad()
    def decode_wavepacket_to_string(self, q_vec: torch.Tensor) -> str:
        raw_b, _ = self.decode_wavepacket_to_bytes(q_vec)
        return raw_b.decode("utf-8", errors="replace")

    @staticmethod
    def classify_char_phase(ch: str) -> UnicodePhaseClass:
        """第一层零参数物理相变边界分类器"""
        if not ch:
            return UnicodePhaseClass.WHITESPACE
        code = ord(ch)
        if ch.isspace():
            return UnicodePhaseClass.WHITESPACE
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch == "_":
            return UnicodePhaseClass.LATIN_ALNUM
        # CJK 统一表意文字及扩展区
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x3040 <= code <= 0x30FF):
            return UnicodePhaseClass.CJK_IDEOGRAPH
        return UnicodePhaseClass.SYNTAX_PUNCT

    @torch.no_grad()
    def _probe_negative_divergence_funnel(
        self,
        candidate_text: str,
        pmc_manager: Optional[PMCManager],
    ) -> Tuple[bool, float, torch.Tensor]:
        """
        第二层：试探合成波包 `q_acc(candidate_text)` 是否落入 PMC 中的负散度吸引盆（Attractor Basin）。
        返回: (is_captured, div_v_value, q_wavepacket)
        """
        q_cand = self.encode_string_to_wavepacket(candidate_text)
        if pmc_manager is None or len(pmc_manager.mounted_packages) == 0:
            return False, 0.0, q_cand

        diag = pmc_manager.query_with_helmholtz_diagnostics(q_cand.view(1, 1, D_BKP))
        max_res = float(diag.max_resonance.item())
        div_val = float(diag.div_v.item())
        # 闭式漏斗捕获判据：共振强度跨过几何中点孔径 0.5*(1 + 1/sqrt(2)) ≈ 0.8536，且散度低于冈贝尔负散度门限 -tau_z*(8)/sqrt(2048)
        is_captured = (max_res > FUNNEL_RESONANCE_APERTURE) and (div_val < GUMBEL_SINK_DIV_STAR)
        effective_div = div_val if is_captured else 0.0
        return is_captured, effective_div, q_cand

    @torch.no_grad()
    def segment_text_to_wavepackets(
        self,
        text: str,
        pmc_manager: Optional[PMCManager] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        双层负散度漏斗流式切分器：
        将任意原始文本（包括带 `{text: "..."}` 外壳的脏数据、中英混排或代码）
        切分为 2048 维单位球面波包序列 `q(1), q(2), ..., q(T) in S^{2047}` 与对应的表面字符串块列表。

        1. 第一层（零参数物理相变边界）：
           - 跨物理相态或遇到语法标点时自动封口；
           - 例如 `{text: "这是一段训练语料。"}` 的前导格式符自动被隔离为独立前导波包 `["{", "text", ": \""]`，
             100% 不会混入正文波包！
        2. 第二层（PMC 负散度漏斗动态捕获）：
           - 对连续 CJK 字符流，递推试探 `c_1 -> c_1 c_2 -> c_1 c_2 c_3` 是否掉入 PMC 的负散度汇点；
           - 若 `PMC` 已凝结出 `"训练"`、`"语料"` 的负散度数据微专家，则自动将其锁进同一个波包漏斗；
           - 在第 0 步冷启动空白流形期，自动退化为单字波包，待负散度微专家结晶后自动升级为复合词波包！
        """
        dev = self.h_256.device
        if not text:
            q_empty = self.encode_bytes_to_wavepacket(b"").unsqueeze(0)
            return q_empty, [""]

        # 第一层：按物理相变切出初级块
        raw_runs: List[Tuple[UnicodePhaseClass, str]] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            ph = self.classify_char_phase(ch)
            if ph == UnicodePhaseClass.SYNTAX_PUNCT:
                # 连续的 `: "` 或 `": "` 作为格式分隔块或单标点块
                if ch == ":" and i + 2 < n and text[i + 1] == " " and text[i + 2] == '"':
                    raw_runs.append((ph, text[i : i + 3]))
                    i += 3
                    continue
                raw_runs.append((ph, ch))
                i += 1
            elif ph == UnicodePhaseClass.WHITESPACE:
                j = i + 1
                while j < n and self.classify_char_phase(text[j]) == UnicodePhaseClass.WHITESPACE:
                    j += 1
                # 纯空格若夹在英文单词间则作为相变分隔符跳过，换行符保留为控制波包
                ws_chunk = text[i:j]
                if "\n" in ws_chunk:
                    raw_runs.append((ph, "\n"))
                i = j
            else:
                j = i + 1
                while j < n and self.classify_char_phase(text[j]) == ph:
                    j += 1
                raw_runs.append((ph, text[i:j]))
                i = j

        wavepackets: List[torch.Tensor] = []
        chunks_out: List[str] = []

        # 第二层：对每个相态块执行字节容量封顶或 CJK 负散度漏斗动态捕获
        for ph, run_str in raw_runs:
            if ph != UnicodePhaseClass.CJK_IDEOGRAPH:
                # 拉丁单词/标识符/标点：按 max_slots 字节上限打包为波包
                b_run = run_str.encode("utf-8")
                for s_idx in range(0, len(b_run), self.max_slots):
                    sub_b = b_run[s_idx : s_idx + self.max_slots]
                    sub_s = sub_b.decode("utf-8", errors="ignore")
                    if sub_s:
                        wavepackets.append(self.encode_bytes_to_wavepacket(sub_b))
                        chunks_out.append(sub_s)
            else:
                # CJK 连续字符流：实时试探 PMC 负散度吸引盆漏斗（批量一次性探查所有候选子串以消除逐字 kernel 启动开销）
                pos = 0
                m_len = len(run_str)
                max_k_cap = max(1, min(4, self.max_slots // 3))
                cand_cache: Dict[str, Tuple[bool, float, torch.Tensor]] = {}
                if pmc_manager is not None and len(pmc_manager.mounted_packages) > 0 and m_len > 0:
                    unique_cands: List[str] = []
                    seen_cands = set()
                    for p_i in range(m_len):
                        for k_i in range(1, min(max_k_cap, m_len - p_i) + 1):
                            cs = run_str[p_i : p_i + k_i]
                            if cs not in seen_cands:
                                seen_cands.add(cs)
                                unique_cands.append(cs)
                    q_batch = torch.stack([self.encode_string_to_wavepacket(cs) for cs in unique_cands], dim=0)
                    diag_b = pmc_manager.query_with_helmholtz_diagnostics(q_batch.unsqueeze(1))
                    res_list = diag_b.max_resonance.reshape(-1).tolist()
                    div_list = diag_b.div_v.reshape(-1).tolist()
                    for idx_c, cs in enumerate(unique_cands):
                        cap_c = (res_list[idx_c] > FUNNEL_RESONANCE_APERTURE) and (div_list[idx_c] < GUMBEL_SINK_DIV_STAR)
                        eff_d = float(div_list[idx_c]) if cap_c else 0.0
                        cand_cache[cs] = (cap_c, eff_d, q_batch[idx_c])

                while pos < m_len:
                    best_str = run_str[pos]
                    if best_str in cand_cache:
                        _, curr_div, best_q = cand_cache[best_str]
                    else:
                        best_q = self.encode_string_to_wavepacket(best_str)
                        _, curr_div, _ = self._probe_negative_divergence_funnel(best_str, pmc_manager)

                    # 向前试探合并第 2 ~ 4 个汉字（最多不超过 max_slots // 3 个汉字）
                    max_cjk_chars = min(4, self.max_slots // 3, m_len - pos)
                    for k_try in range(2, max_cjk_chars + 1):
                        cand_str = run_str[pos : pos + k_try]
                        if cand_str in cand_cache:
                            captured, cand_div, cand_q = cand_cache[cand_str]
                        else:
                            captured, cand_div, cand_q = self._probe_negative_divergence_funnel(cand_str, pmc_manager)
                        if captured and cand_div < curr_div:
                            best_str = cand_str
                            best_q = cand_q
                            curr_div = cand_div
                        else:
                            break

                    wavepackets.append(best_q)
                    chunks_out.append(best_str)
                    pos += len(best_str)

        if len(wavepackets) == 0:
            q_empty = self.encode_bytes_to_wavepacket(b"").unsqueeze(0)
            return q_empty, [""]

        return torch.stack(wavepackets, dim=0).to(dev), chunks_out


class PenroseHarness(nn.Module):
    """
    Penrose v1.0 顶层运行时调度系统（Penrose Harness）：
    统一管理 `PenroseMPU`（纯球面状态推进器）、`PMCManager`（同质微专家流形块森林）与
    `HolographicWavepacketTransducer`（外围 I/O 换能总线），实现：
    1. Module 5 出口的硬件级三态总线分支（分支 A 内部旋度闭环 / 分支 B 外部负散度坍缩吐字 / 分支 C 球面并行训练）；
    2. 免清洗拖拽式训练（`drag_and_drop_ingest`）：支持带 `{text: "..."}` 等脏外壳文档零清洗拖入、
       无 AdamW 的 Kaczmarz + 罗德里格斯测地线单步锁死；
    3. 冷启动向心凝聚成词（`nucleate_composite_word_sink`）：在合成波包坐标处自动结晶出负散度数据微专家；
    4. 自治森林四大拓扑算子巡检（`Auto-Mitosis`, `Auto-Expand`, `Auto-Split`, `Auto-Merge`）。
    """

    def __init__(
        self,
        mpu: Optional[PenroseMPU] = None,
        pmc_manager: Optional[PMCManager] = None,
        auto_evolve_interval: int = 5,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.mpu = (mpu or PenroseMPU()).to(device)
        self.pmc_manager = pmc_manager or PMCManager(device=device)
        self.transducer = HolographicWavepacketTransducer(device=device)
        self.manifold_step: int = 0
        self.auto_evolve_interval: int = max(1, int(auto_evolve_interval))

        # 确保至少挂载一颗可写工作区流形块 Workspace.pmc
        if len(self.pmc_manager.mounted_packages) == 0:
            ws_pkg = PMCPackage(
                pkg_id="Workspace.pmc",
                grid_k=64,
                mount_mode=PMCMountMode.RW_SHARED,
                is_base_pkg=True,
                device=device,
            )
            self.pmc_manager.mount_package(ws_pkg, pin_to_l1=True)

    @torch.no_grad()
    def nucleate_composite_word_sink(
        self,
        word_str: str,
        next_word_str: Optional[str] = None,
        sink_strength: float = JET_NORMAL_ALPHA_STAR,
    ) -> int:
        """
        冷启动与流形生长闭环算子（Negative-Divergence Sink Nucleation）：
        在合成波包坐标 `q(word_str)` 处结晶出一个负散度数据微专家（满足 <u_m, v_m> < 0，即 div V < 0），
        使得下一轮 `HolographicWavepacketTransducer` 的第二层负散度漏斗扫描到该连续汉字时，
        自动将其吸入同一个复合词波包漏斗！
        """
        pkg = self.pmc_manager.get_writable_package()
        if pkg is None:
            return -1

        q_word = self.transducer.encode_string_to_wavepacket(word_str).to(self.device)
        if next_word_str is not None:
            q_next = self.transducer.encode_string_to_wavepacket(next_word_str).to(self.device)
            v_tan = q_next - (q_next * q_word).sum() * q_word
        else:
            v_tan = torch.zeros_like(q_word)

        # 构造强负散度向心载荷：v_m = -sink_strength * q_word + v_tan，保证 <u_m, v_m> = -sink_strength * d < 0
        v_sink = (-float(sink_strength) * q_word + v_tan) * math.sqrt(float(D_BKP))

        # 优先挑选整行与整列均未占用的空白坐标 (r_idx, c_idx)，避免覆盖存量微专家的子键寻址
        k_grid = pkg.grid_k
        act_2d = (pkg.scales[:, 1] > 1e-6).view(k_grid, k_grid)
        empty_rows = torch.nonzero(~act_2d.any(dim=1), as_tuple=False).squeeze(-1)
        empty_cols = torch.nonzero(~act_2d.any(dim=0), as_tuple=False).squeeze(-1)
        if empty_rows.numel() > 0 and empty_cols.numel() > 0:
            r_idx = int(empty_rows[0].item())
            c_idx = int(empty_cols[0].item())
            eid = r_idx * k_grid + c_idx
        else:
            empty_eids = torch.nonzero(pkg.scales[:, 1] <= 1e-6, as_tuple=False).squeeze(-1)
            eid = int(empty_eids[0].item()) if empty_eids.numel() > 0 else 0
            r_idx = eid // k_grid
            c_idx = eid % k_grid

        q1_w, q2_w = self.pmc_manager._standardize_subqueries(q_word.view(1, D_BKP), update_center=False)
        pkg.u_codebook.data[:, r_idx, :] = F.normalize(q1_w[0], dim=-1)
        pkg.v_codebook.data[:, c_idx, :] = F.normalize(q2_w[0], dim=-1)
        pkg._u_cb_norm_cache = None
        pkg._v_cb_norm_cache = None
        pkg.quantize_write_expert_page(eid, q_word, v_sink, hit_increment=5.0, purity_val=1.0)
        self.pmc_manager.vram_cache.invalidate_or_refresh(pkg, torch.tensor([eid], device=self.device, dtype=torch.long))
        return eid

    @torch.no_grad()
    def step_tri_state_bus(
        self,
        q_curr: torch.Tensor,
        q_prev: Optional[torch.Tensor] = None,
        max_internal_curl_ticks: int = BKP_NUM_HEADS,
        sink_div_threshold: float = -AMBIENT_NOISE_FLOOR,
    ) -> Dict[str, Any]:
        """
        Module 5 出口的硬件级三态总线分支执行器（用于推理与自治状态机演进）：
        - 分支 A（内部旋度闭环 / 隐式思考）：
          当 div V(q(tau+1)) > sink_div_threshold 且旋度 eta_curl 占优时，
          q(tau+1) 不经过任何解码器，直接经内部高速旁路跳回 Module 1 入口继续空转推进！
        - 分支 B（外部负散度坍缩 / 符号吐出）：
          一旦状态滑入负散度概念汇点（div V <= sink_div_threshold 或达到最大内部旋度步数），
          Harness 读取 q(tau+1)，执行零参数逆阿达玛变换一次性还原出 UTF-8 字节流！
        """
        self.mpu.eval()
        curr = q_curr.to(self.device)
        if curr.dim() == 1:
            curr = curr.view(1, 1, D_BKP)
        elif curr.dim() == 2:
            curr = curr.unsqueeze(0)

        prev = q_prev.to(self.device) if q_prev is not None else curr[:, :1, :]
        if prev.dim() == 1:
            prev = prev.view(1, 1, D_BKP)
        elif prev.dim() == 2:
            prev = prev.unsqueeze(0)

        branch_trace: List[str] = []
        div_trace: List[float] = []
        curl_trace: List[float] = []

        q_out = curr[:, -1:, :]
        for tick in range(max(1, int(max_internal_curl_ticks))):
            step_res = self.mpu(curr, q_prev=prev, pmc_manager=self.pmc_manager)
            q_out = step_res.q_next[:, -1:, :]
            div_last = float(step_res.helmholtz_diag.div_v[:, -1].mean().item())
            curl_last = float(step_res.helmholtz_diag.eta_curl[:, -1].mean().item())
            div_trace.append(div_last)
            curl_trace.append(curl_last)

            if div_last <= float(sink_div_threshold) or tick == max_internal_curl_ticks - 1:
                # 触发分支 B：落入负散度概念汇点，执行外围全息逆变换吐出符号
                branch_trace.append(BusBranchState.BRANCH_B_NEGATIVE_DIV_EMIT.value)
                break
            else:
                # 触发分支 A：内部高旋度无散流，不解码直接跳回 Module 1 入口！
                branch_trace.append(BusBranchState.BRANCH_A_INTERNAL_CURL_LOOP.value)
                curr = torch.cat([curr, q_out], dim=1)
                prev = torch.cat([curr[:, :1, :], curr[:, :-1, :]], dim=1)

        emitted_str = self.transducer.decode_wavepacket_to_string(q_out.squeeze())
        return {
            "q_final": q_out.squeeze(0).squeeze(0),
            "emitted_text": emitted_str,
            "internal_curl_ticks": len(branch_trace),
            "branch_trace": branch_trace,
            "div_trace": div_trace,
            "curl_trace": curl_trace,
        }

    def train_manifold_trajectory(
        self,
        raw_text_or_wavepackets: Union[str, torch.Tensor],
        snap_rate: float = 1.0,
        enable_auto_mitosis: bool = True,
        mpu_optimizer: Optional[torch.optim.Optimizer] = None,
        domain_hint: str = "general",
        auto_evolve: bool = True,
        eval_before_after: bool = False,
    ) -> Dict[str, Any]:
        """
        统一流形轨迹训练引擎（Unified Manifold Trajectory Training — 分支 C 球面并行训练 + Kaczmarz 一步更新）：
        - 默认 `eval_before_after=False`：关闭 `out_before` 和 `out_after` 的额外两次 `mpu(...)` 前向推理，
          单步仅执行 1 次 `mpu(...)` 前向，直接用 `out_train.loss` 作为当前步监控指标；
        - 当 `eval_before_after=True` 时，额外测量单步更新前后的 `loss_before -> loss_after` 弦长误差对比。
        """
        if isinstance(raw_text_or_wavepackets, str):
            q_seq, surface_chunks = self.transducer.segment_text_to_wavepackets(
                raw_text_or_wavepackets, pmc_manager=self.pmc_manager
            )
        else:
            q_seq = F.normalize(raw_text_or_wavepackets.to(self.device).float().reshape(-1, D_BKP), dim=-1)
            surface_chunks = []

        if q_seq.size(0) < 2:
            return {
                "step": int(self.manifold_step),
                "num_wavepackets": int(q_seq.size(0)),
                "surface_chunks": surface_chunks,
                "loss": 0.0,
                "loss_before": 0.0,
                "loss_after": 0.0,
                "kaczmarz_stats": {},
                "forest_events": [],
            }

        self.manifold_step += 1
        q_curr = q_seq[:-1].unsqueeze(0)  # (1, T-1, 2048)
        q_target = q_seq[1:].unsqueeze(0)  # (1, T-1, 2048)
        q_prev = torch.cat([q_curr[:, :1, :], q_curr[:, :-1, :]], dim=1)

        loss_before = 0.0
        if eval_before_after:
            with torch.no_grad():
                out_before = self.mpu(q_curr, q_prev=q_prev, pmc_manager=self.pmc_manager, q_target=q_target)
                loss_before = float(out_before.loss.detach().cpu()) if out_before.loss is not None else 1.0

        # 1. PMC 无 AdamW 闭式 Kaczmarz + 罗德里格斯大圆旋转一步批处理锁死
        self.pmc_manager.train(True)
        kaczmarz_stats = self.pmc_manager.geodesic_kaczmarz_snap_update(
            q_curr=q_curr,
            q_prev=q_prev,
            q_target=q_target,
            snap_rate=snap_rate,
            enable_auto_mitosis=enable_auto_mitosis,
        )
        # 流线偶极坍缩成词 (Streamline Dipole Nucleation): 对已锁死流线直接复用 nucleate_composite_word_sink 结晶复合词漏斗
        for d_idx in kaczmarz_stats.get("locked_dipole_indices", []):
            if d_idx + 1 < len(surface_chunks):
                w1, w2 = surface_chunks[d_idx], surface_chunks[d_idx + 1]
                if (
                    len(w1) + len(w2) <= 4
                    and self.transducer.classify_char_phase(w1[0]) == UnicodePhaseClass.CJK_IDEOGRAPH
                    and self.transducer.classify_char_phase(w2[0]) == UnicodePhaseClass.CJK_IDEOGRAPH
                ):
                    next_w = surface_chunks[d_idx + 2] if (d_idx + 2 < len(surface_chunks)) else None
                    self.nucleate_composite_word_sink(w1 + w2, next_w)

        # 2. 若处于 MPU 联合训练阶段（分支 C 并行训练），单步仅执行 1 次 mpu(...) 前向与反向传播
        train_loss: Optional[float] = None
        if mpu_optimizer is not None:
            self.mpu.train()
            mpu_optimizer.zero_grad(set_to_none=True)
            out_train = self.mpu(q_curr, q_prev=q_prev, pmc_manager=self.pmc_manager, q_target=q_target)
            assert out_train.loss is not None
            out_train.loss.backward()
            mpu_optimizer.step()
            # 若使用 Agile 降维配置 (d_model < 2048)，自动执行 Stiefel 正交流形重投影；原生 2048 维下为空操作
            self.mpu.project_stiefel_frames()
            for layer in self.mpu.module2_lower_stack:
                layer.moe.update_expert_bias()
            for layer in self.mpu.module4_upper_stack:
                layer.moe.update_expert_bias()
            train_loss = float(out_train.loss.detach().cpu())

        if eval_before_after:
            with torch.no_grad():
                self.mpu.eval()
                self.pmc_manager.train(True)
                out_after = self.mpu(q_curr, q_prev=q_prev, pmc_manager=self.pmc_manager, q_target=q_target)
                loss_after = float(out_after.loss.detach().cpu()) if out_after.loss is not None else 0.0
        else:
            if train_loss is None:
                with torch.no_grad():
                    self.mpu.eval()
                    self.pmc_manager.train(True)
                    out_single = self.mpu(q_curr, q_prev=q_prev, pmc_manager=self.pmc_manager, q_target=q_target)
                    train_loss = float(out_single.loss.detach().cpu()) if out_single.loss is not None else 0.0
            loss_before = train_loss
            loss_after = train_loss

        # 3. 按时间步或满载信号自动触发 PMCManager 内聚的六大森林自治演化算子
        forest_events: List[Dict[str, Any]] = []
        pkg_w = self.pmc_manager.get_writable_package()
        needs_urgent_expand = bool(pkg_w is not None and getattr(pkg_w, "_needs_quadrant_expand", False))
        if auto_evolve and (self.manifold_step % self.auto_evolve_interval == 0 or needs_urgent_expand):
            forest_events = self.pmc_manager.step_autonomous_forest_evolution(
                step=self.manifold_step,
                q_probe=q_curr,
                domain_hint=domain_hint,
            )

        curr_loss = train_loss if train_loss is not None else loss_after
        return {
            "step": int(self.manifold_step),
            "num_wavepackets": int(q_seq.size(0)),
            "surface_chunks": surface_chunks,
            "loss": round(curr_loss, 6),
            "loss_before": round(loss_before, 6),
            "loss_after": round(loss_after, 6),
            "kaczmarz_stats": kaczmarz_stats,
            "forest_events": forest_events,
        }

    def drag_and_drop_ingest(
        self,
        raw_text: str,
        snap_rate: float = 1.0,
        enable_auto_mitosis: bool = True,
        mpu_optimizer: Optional[torch.optim.Optimizer] = None,
        domain_hint: str = "general",
        auto_evolve: bool = True,
        eval_before_after: bool = False,
    ) -> Dict[str, Any]:
        return self.train_manifold_trajectory(
            raw_text_or_wavepackets=raw_text,
            snap_rate=snap_rate,
            enable_auto_mitosis=enable_auto_mitosis,
            mpu_optimizer=mpu_optimizer,
            domain_hint=domain_hint,
            auto_evolve=auto_evolve,
            eval_before_after=eval_before_after,
        )

    @torch.no_grad()
    def erase_manifold_trajectory(
        self,
        raw_text_or_wavepackets: Union[str, torch.Tensor],
        erase_rate: float = 1.0,
        eval_before_after: bool = False,
    ) -> Dict[str, Any]:
        """
        逆向流形轨迹擦除引擎（Inverse Manifold Trajectory Unlearning）：
        与 `train_manifold_trajectory` 严格对偶的闭式机器遗忘接口，支持：
        - 独占概念微专家触发公理 9「细胞凋亡（Cellular Apoptosis）」原位清零；
        - 共享语法主干微专家执行秩-1 平行推力正交剥离与逆罗德里格斯回退；
        - 流线邻接表 `stream_next_ptr` 拓扑断链。
        """
        if isinstance(raw_text_or_wavepackets, str):
            q_seq, surface_chunks = self.transducer.segment_text_to_wavepackets(
                raw_text_or_wavepackets, pmc_manager=self.pmc_manager
            )
        else:
            q_seq = F.normalize(raw_text_or_wavepackets.to(self.device).float().reshape(-1, D_BKP), dim=-1)
            surface_chunks = []

        if q_seq.size(0) < 2:
            return {
                "num_wavepackets": int(q_seq.size(0)),
                "surface_chunks": surface_chunks,
                "loss_before": 0.0,
                "loss_after": 0.0,
                "erase_stats": {},
            }

        self.mpu.eval()
        q_curr = q_seq[:-1].unsqueeze(0)
        q_target = q_seq[1:].unsqueeze(0)
        q_prev = torch.cat([q_curr[:, :1, :], q_curr[:, :-1, :]], dim=1)

        loss_before = 0.0
        if eval_before_after:
            out_before = self.mpu(q_curr, q_prev=q_prev, pmc_manager=self.pmc_manager, q_target=q_target)
            loss_before = float(out_before.loss.detach().cpu()) if out_before.loss is not None else 0.0

        erase_stats = self.pmc_manager.geodesic_kaczmarz_erase_downdate(
            q_curr=q_curr,
            q_prev=q_prev,
            q_target=q_target,
            erase_rate=erase_rate,
        )

        loss_after = loss_before
        if eval_before_after:
            out_after = self.mpu(q_curr, q_prev=q_prev, pmc_manager=self.pmc_manager, q_target=q_target)
            loss_after = float(out_after.loss.detach().cpu()) if out_after.loss is not None else 0.0

        return {
            "num_wavepackets": int(q_seq.size(0)),
            "surface_chunks": surface_chunks,
            "loss_before": round(loss_before, 6),
            "loss_after": round(loss_after, 6),
            "erase_stats": erase_stats,
        }

    @torch.no_grad()
    def drag_and_drop_erase(
        self,
        raw_text: str,
        erase_rate: float = 1.0,
        eval_before_after: bool = False,
    ) -> Dict[str, Any]:
        return self.erase_manifold_trajectory(
            raw_text_or_wavepackets=raw_text,
            erase_rate=erase_rate,
            eval_before_after=eval_before_after,
        )

    @torch.no_grad()
    def run_autonomous_topology_cycle(
        self,
        q_probe: Optional[torch.Tensor] = None,
        domain_hint: str = "general",
    ) -> Dict[str, Any]:
        """
        显式触发一次 `PMCManager` 的六大森林自治算子巡检并返回森林全局快照：
        - Auto-Split   : 二分图拉普拉斯谱双聚类裂变 + BridgeAB
        - Auto-Expand  : 四象限 (a+b)^2 零误差倍增扩容
        - Auto-Merge   : 跨包三元正交聚变 (A ∩ B) ⊕ (A \\ B) ⊕ (B \\ A)
        - Auto-Spawn   : 冈贝尔极值尾概率未知流形新包孵化
        - Auto-Closure : 双正交笛卡尔闭包成核
        - Auto-Mitosis : 切向复共轭有丝分裂
        """
        events = self.pmc_manager.step_autonomous_forest_evolution(
            step=self.manifold_step,
            q_probe=q_probe,
            domain_hint=domain_hint,
        )
        snap = self.pmc_manager.snapshot_forest_state(step=self.manifold_step)
        pkg = self.pmc_manager.get_writable_package()
        active_pages = int((pkg.scales[:, 1] > 1e-6).sum().item()) if pkg is not None else 0
        grid_k = int(pkg.grid_k) if pkg is not None else 0
        occupancy = active_pages / float(max(1, pkg.num_experts)) if pkg is not None else 0.0
        return {
            "step": int(self.manifold_step),
            "pkg_id": pkg.pkg_id if pkg is not None else "",
            "grid_k": grid_k,
            "active_pages": active_pages,
            "occupancy_ratio": round(occupancy, 4),
            "auto_expanded": any(e.get("op") == "Auto-Expand" for e in events),
            "total_kaczmarz_snaps": int(self.pmc_manager.total_kaczmarz_snaps),
            "total_mitosis_spawns": int(self.pmc_manager.total_mitosis_spawns),
            "events": events,
            "forest_snapshot": snap,
        }


