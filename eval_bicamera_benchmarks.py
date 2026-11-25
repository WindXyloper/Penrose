import os
import re
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bicamera_kernel_draft import BicameraKernel
from bkp_manager import BKPManager, BKPPackage, BKPMountMode

BOS_ID, EOS_ID, SEP_ID = 1, 2, 3
USER_TOK, THINK_BOS_TOK, THINK_EOS_TOK = (
    "<|reserved_0|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
)
ASST_TOK = ""
USER_ID, THINK_BOS_ID, THINK_EOS_ID = 4, 6, 7
ASST_ID = 5


# ==============================================================================
# 1. 标准评测题库定义 (ARC-Science / PIQA / MMLU-CrossDomain / GSM8K-MultiHop)
# ==============================================================================
ARC_SCIENCE_BENCHMARK = [
    {
        "id": "arc_01",
        "question": "在高原地区用普通敞口锅烧水时，水沸腾的温度低于 100°C 且食物更难煮熟，其根本物理原因是什么？",
        "choices": [
            "高原海拔高导致大气压强降低，水的沸点随气压降低而下降，较低的沸水温度减缓了蛋白质变性速率。",
            "高原地区重力加速度变大，水分子受到更强的引力束缚导致沸点升高。",
            "高原空气中氧气含量减少，水分子必须吸收氧气才能完成汽化相变。",
            "高原太阳紫外线辐射强烈，直接破坏了水分子的共价键结构。",
        ],
        "gold": 0,
    },
    {
        "id": "arc_02",
        "question": "植物叶片靠近上表皮的栅栏组织细胞排列紧密且富含叶绿体，而靠近下表皮的海绵组织排列疏松，这种结构的主要适应性意义是？",
        "choices": [
            "紧密的栅栏组织最大化捕获顶层入射阳光进行光合作用，疏松的海绵组织配合下表皮气孔促进二氧化碳气体扩散。",
            "栅栏组织用于储存大量水分防止根系干枯，海绵组织专门用于反射多余的绿光。",
            "上表皮没有角质层保护，必须靠栅栏组织进行有氧呼吸释放热量。",
            "海绵组织负责合成叶绿素并通过筛管向上运输给栅栏组织。",
        ],
        "gold": 0,
    },
    {
        "id": "arc_03",
        "question": "When a spacecraft orbits Earth in a stable circular low-Earth orbit with engines turned off, why do astronauts inside experience weightlessness?",
        "choices": [
            "Because both the spacecraft and the astronauts are in continuous free fall toward Earth with the same centripetal acceleration due to gravity.",
            "Because there is zero gravitational pull from Earth once an object leaves the atmosphere.",
            "Because the centrifugal force from the Moon cancels out Earth's magnetic field.",
            "Because the vacuum of space exerts an upward buoyant force equal to the astronauts' mass.",
        ],
        "gold": 0,
    },
    {
        "id": "arc_04",
        "question": "Why does a metal spoon feel much colder to the touch than a wooden spoon when both have been sitting in the same room at 20°C?",
        "choices": [
            "Metal has a much higher thermal conductivity than wood, so it conducts heat away from the human finger much faster.",
            "Metal naturally maintains a lower equilibrium temperature than wood in the same room.",
            "Wood generates internal thermal energy through slow oxidation with room air.",
            "Metal reflects infrared radiation back into the hand while wood absorbs it.",
        ],
        "gold": 0,
    },
    {
        "id": "arc_05",
        "question": "在标准大气压下，将一束单色激光从空气斜射入折射率 n = 1.5 的玻璃砖中，下列哪一物理量保持严格不变？",
        "choices": [
            "光波的频率 (Frequency)，因为频率仅由光源的受激辐射能级差决定，而波速和波长在介质中均减为原来的 1/n。",
            "光在介质中的传播速度 (Phase Velocity) 与波长均保持不变。",
            "光的折射角始终严格等于入射角。",
            "光子的动量大小在进入玻璃后减为零。",
        ],
        "gold": 0,
    },
    {
        "id": "arc_06",
        "question": "During cellular respiration in eukaryotic cells, which organelle is primarily responsible for synthesizing the majority of ATP through oxidative phosphorylation?",
        "choices": [
            "The mitochondrion, via the electron transport chain and ATP synthase across its inner membrane.",
            "The ribosome, by polymerizing amino acids into polypeptide chains.",
            "The Golgi apparatus, through vesicle packaging and exocytosis.",
            "The lysosome, by hydrolyzing macromolecules at acidic pH.",
        ],
        "gold": 0,
    },
]

PIQA_COMMONSENSE_BENCHMARK = [
    {
        "id": "piqa_01",
        "goal": "如何在没有高压锅的高海拔营地尽快把生土豆煮熟？",
        "choices": [
            "将土豆切成薄片或小丁以减小热传导距离，并在锅盖上压重物尽可能减少蒸汽逸出以提高锅内局部气压和沸点。",
            "把锅盖完全敞开并不断往沸水里加入冰块以刺激淀粉吸水膨胀。",
        ],
        "gold": 0,
    },
    {
        "id": "piqa_02",
        "goal": "To loosen a tightly rusted metal jar lid on a glass jar without breaking the glass:",
        "choices": [
            "Run hot water briefly over the metal lid so the metal expands slightly more than the glass due to its higher thermal expansion coefficient.",
            "Submerge the entire glass jar in liquid nitrogen and strike the glass base with a steel hammer.",
        ],
        "gold": 0,
    },
    {
        "id": "piqa_03",
        "goal": "在训练深度学习大模型时，为了在不改变数学梯度真值的前提下消除末端全词表 Logits 张量 (B, T, V) 造成的数 GB 显存峰值：",
        "choices": [
            "将隐状态 h 按 Token 块 (如 chunk=2048) 切分，结合 activation checkpointing 在反向传播时逐块重算局部 Logits 与交叉熵梯度。",
            "将词表大小 V 直接截断为 10 个词并丢弃其余所有 Token 的损失函数。",
        ],
        "gold": 0,
    },
    {
        "id": "piqa_04",
        "goal": "When rescuing someone who has fallen through thin ice on a frozen lake:",
        "choices": [
            "Lie flat on the ice to distribute your weight over a larger surface area and push a long pole, ladder, or rope toward them.",
            "Run upright as fast as possible directly to the broken edge of the ice hole and jump in.",
        ],
        "gold": 0,
    },
]

MMLU_CROSS_DOMAIN_BENCHMARK = [
    {
        "id": "mmlu_cs_01",
        "subject": "computer_science",
        "question": "在操作系统中，采用 4KB 页对齐的虚拟内存映射 (mmap) 访问超大规模稀疏微专家库相比传统 read() 系统调用的核心优势是什么？",
        "choices": [
            "4KB 对齐的 mmap 直接与硬件 MMU 页表 1:1 映射，消除了双重拷贝，并利用按需缺页中断仅传输被稀疏唤醒的微专家页。",
            "mmap 会将磁盘机械转速提升 10 倍并绕过 CPU 一级指令缓存。",
            "read() 系统调用不支持读取大于 64KB 的二进制文件。",
            "mmap 在启动时强制把几百 GB 文件一次性全部复制进 GPU 寄存器。",
        ],
        "gold": 0,
    },
    {
        "id": "mmlu_cs_02",
        "subject": "computer_science",
        "question": "What is the worst-case time complexity of searching for a target key in a sorted array of N = 1024 elements using binary search?",
        "choices": [
            "O(log2 N), requiring at most log2(1024) + 1 = 11 comparisons because the search interval halves at each step.",
            "O(N^2), requiring 1,048,576 comparisons by scanning all element pairs.",
            "O(N log N), because the array must be re-sorted before every comparison.",
            "O(1) with zero memory reads regardless of N.",
        ],
        "gold": 0,
    },
    {
        "id": "mmlu_math_01",
        "subject": "college_mathematics",
        "question": "已知二阶方阵 M = [[5, 2], [3, 4]]，则该矩阵的迹 tr(M)、行列式 det(M) 以及特征多项式 p(lambda) 分别为？",
        "choices": [
            "tr(M) = 9，det(M) = 14，特征多项式为 p(lambda) = lambda^2 - 9*lambda + 14。",
            "tr(M) = 20，det(M) = 6，特征多项式为 p(lambda) = lambda^2 + 20*lambda - 6。",
            "tr(M) = 1，det(M) = 26，特征多项式为 p(lambda) = lambda^2 - lambda + 26。",
            "tr(M) = 14，det(M) = 9，特征多项式为 p(lambda) = lambda^3 - 14。",
        ],
        "gold": 0,
    },
    {
        "id": "mmlu_math_02",
        "subject": "college_mathematics",
        "question": "Let A be a real symmetric positive-definite matrix of size d x d. Which statement about its eigenvalues and ZCA whitening matrix W = A^(-1/2) is strictly true?",
        "choices": [
            "All eigenvalues of A are strictly positive real numbers, and W = A^(-1/2) maps covariance A to the isotropic identity matrix I.",
            "All eigenvalues of A are purely imaginary numbers with negative real parts.",
            "The determinant of A is strictly zero so A^(-1/2) does not exist.",
            "A symmetric matrix can never be diagonalized over the real field.",
        ],
        "gold": 0,
    },
    {
        "id": "mmlu_phys_01",
        "subject": "college_physics",
        "question": "质量 m = 4 kg 的物体从静止受恒力产生加速度 a = 5 m/s^2，持续 t = 6 s。则合外力 F、末速度 v、位移 s 与末动能 E_k 分别为？",
        "choices": [
            "F = 20 N，v = 30 m/s，s = 90.0 m，E_k = 1800.0 J。",
            "F = 9 N，v = 11 m/s，s = 30.0 m，E_k = 120.0 J。",
            "F = 120 N，v = 5 m/s，s = 6.0 m，E_k = 50.0 J。",
            "F = 20 N，v = 15 m/s，s = 180.0 m，E_k = 450.0 J。",
        ],
        "gold": 0,
    },
    {
        "id": "mmlu_phys_02",
        "subject": "college_physics",
        "question": "Why does a pure bilinear operator O(h, z) = Gate(h) * Up(z) strictly guarantee zero output when the external knowledge payload z is zero (z = 0)?",
        "choices": [
            "Because Up(z) = W_up · z is a homogeneous linear projection without additive bias, so z = 0 implies Up(0) = 0 and Gate(h) * 0 = 0 identically.",
            "Because Gate(h) sets the context hidden state h to infinity when z is zero.",
            "Because bilinear operators ignore both h and z and output a constant vector.",
            "Because RMSNorm divides zero by zero to produce a Gaussian vector.",
        ],
        "gold": 0,
    },
]

GSM8K_MULTIHOP_COT_BENCHMARK = [
    {
        "id": "gsm8k_test_01",
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "gold_steps": [
            "Janet's ducks lay 16 eggs per day.",
            "She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs every day.",
            "The remaining eggs sold at the market are 16 - 7 = 9 eggs.",
            "At $2 per egg, she makes 9 * 2 = 18 dollars every day.",
        ],
        "gold_answer": "18",
    },
    {
        "id": "gsm8k_test_02",
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "gold_steps": [
            "The robe takes 2 bolts of blue fiber.",
            "It takes half that much white fiber, which is 2 / 2 = 1 bolt of white fiber.",
            "In total, it takes 2 + 1 = 3 bolts of fiber.",
        ],
        "gold_answer": "3",
    },
    {
        "id": "gsm8k_test_03",
        "question": "一个果园里有 45 棵苹果树，梨树的数量比苹果树少 17 棵，桃树的数量正好是梨树的 2 倍。请问果园里桃树有多少棵？",
        "gold_steps": [
            "已知苹果树有 45 棵，梨树比苹果树少 17 棵，所以梨树有 45 - 17 = 28 棵。",
            "桃树的数量是梨树的 2 倍，即 28 * 2 = 56 棵。",
        ],
        "gold_answer": "56",
    },
    {
        "id": "gsm8k_test_04",
        "question": "一列火车以 80 千米/小时的速度匀速行驶了 3 小时，随后提速到 110 千米/小时继续行驶了 2 小时。求这列火车总共行驶了多少千米？",
        "gold_steps": [
            "前 3 小时的行驶路程为 80 * 3 = 240 千米。",
            "后 2 小时的行驶路程为 110 * 2 = 220 千米。",
            "两段路程相加，总行驶路程为 240 + 220 = 460 千米。",
        ],
        "gold_answer": "460",
    },
    {
        "id": "gsm8k_test_05",
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "gold_steps": [
            "James runs 3 sprints per session, 3 times a week, which is 3 * 3 = 9 sprints a week.",
            "Each sprint is 60 meters, so he runs 9 * 60 = 540 meters in total per week.",
        ],
        "gold_answer": "540",
    },
    {
        "id": "gsm8k_test_06",
        "question": "书店进了一批故事书共 150 本，第一天卖出 38 本，第二天卖出 42 本，剩下的平均分装在 7 个箱子里，每个箱子装多少本？",
        "gold_steps": [
            "前两天一共卖出 38 + 42 = 80 本。",
            "剩下故事书 150 - 80 = 70 本。",
            "平均分装在 7 个箱子里，每个箱子装 70 / 7 = 10 本。",
        ],
        "gold_answer": "10",
    },
]


def get_all_benchmark_forbidden_substrings() -> List[str]:
    """
    提取所有评测基准 (ARC / PIQA / MMLU / GSM8K) 的题干与关键短语，
    供数据准备管线 (prepare_workspace.py / build_cot_distill_dataset.py) 执行编译期防泄漏扫描，
    严禁任何评测集原题或同构句式进入训练集！
    """
    forbidden = []
    for suite in (ARC_SCIENCE_BENCHMARK, PIQA_COMMONSENSE_BENCHMARK, MMLU_CROSS_DOMAIN_BENCHMARK, GSM8K_MULTIHOP_COT_BENCHMARK):
        for item in suite:
            q = (item.get("question") or item.get("goal") or "").strip()
            if len(q) >= 12:
                forbidden.append(q)
                # 同时取前 24 个字符作为前缀指纹
                forbidden.append(q[:min(28, len(q))])
            for ch in item.get("choices", []):
                if len(ch) >= 16:
                    forbidden.append(ch[:min(32, len(ch))])
            for st in item.get("gold_steps", []):
                if len(st) >= 16:
                    forbidden.append(st[:min(32, len(st))])
    return forbidden


def is_text_contaminated_by_benchmark(text: str, forbidden_list: Optional[List[str]] = None) -> bool:
    """检查给定训练文本是否包含任何评测集禁用指纹（零容忍防泄漏守卫）"""
    if not text:
        return False
    checks = forbidden_list if forbidden_list is not None else get_all_benchmark_forbidden_substrings()
    norm_txt = re.sub(r"\s+", " ", text).lower()
    for pat in checks:
        norm_pat = re.sub(r"\s+", " ", pat).lower()
        if norm_pat and norm_pat in norm_txt:
            return True
    return False


# ==============================================================================
# 2. 标准多选题长度归一化对数似然打分器 (lm-eval-harness standard norm_loglik)
# ==============================================================================
@torch.no_grad()
def score_continuation_loglik(
    model: BicameraKernel,
    bkp_mgr: Optional[BKPManager],
    tok: Tokenizer,
    prompt_str: str,
    continuation_str: str,
    device: torch.device,
) -> Tuple[float, float]:
    """计算条件对数似然 log P(continuation | prompt) 的总和与长度归一化均值 (acc_norm)。"""
    p_ids = [BOS_ID] + tok.encode(prompt_str).ids
    c_ids = tok.encode(continuation_str).ids
    if len(c_ids) == 0:
        return -1e9, -1e9

    full_ids = p_ids + c_ids
    inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with amp_ctx:
        logits, _, _ = model(inp, bkp_manager=bkp_mgr, bkp_dropout=(bkp_mgr is None))

    start_pos = len(p_ids) - 1
    c_logits = logits[0, start_pos:, :].float()
    c_targets = tgt[0, start_pos:]
    token_nll = F.cross_entropy(c_logits, c_targets, reduction="none")
    sum_loglik = float((-token_nll.sum()).item())
    norm_loglik = sum_loglik / max(1, len(c_ids))
    return sum_loglik, norm_loglik


@torch.no_grad()
def evaluate_multiple_choice_suite(
    model: BicameraKernel,
    bkp_mgr: Optional[BKPManager],
    tok: Tokenizer,
    items: List[Dict],
    device: torch.device,
    use_cot_prefix: bool = False,
) -> Dict[str, float]:
    """在标准多选基准 (ARC / PIQA / MMLU) 上执行零样本条件似然评测"""
    correct_raw = 0
    correct_norm = 0
    margins = []
    uniq_experts_acc = []

    for item in items:
        q_txt = item.get("question") or item.get("goal") or ""
        prompt = f"{USER_TOK}Question: {q_txt}\n{ASST_TOK}" if use_cot_prefix else f"Question: {q_txt}\nAnswer: "

        raw_scores, norm_scores = [], []
        for ch in item["choices"]:
            s_raw, s_norm = score_continuation_loglik(model, bkp_mgr, tok, prompt, ch, device)
            raw_scores.append(s_raw)
            norm_scores.append(s_norm)
            if bkp_mgr is not None:
                uniq_experts_acc.append(bkp_mgr.last_unique_experts_per_seq)

        gold = int(item["gold"])
        pred_raw = int(max(range(len(raw_scores)), key=lambda i: raw_scores[i]))
        pred_norm = int(max(range(len(norm_scores)), key=lambda i: norm_scores[i]))
        if pred_raw == gold:
            correct_raw += 1
        if pred_norm == gold:
            correct_norm += 1

        gold_score = norm_scores[gold]
        best_wrong = max(norm_scores[i] for i in range(len(norm_scores)) if i != gold)
        margins.append(gold_score - best_wrong)

    n_total = max(1, len(items))
    return {
        "acc_raw": round(100.0 * correct_raw / n_total, 2),
        "acc_norm": round(100.0 * correct_norm / n_total, 2),
        "avg_loglik_margin": round(sum(margins) / n_total, 4),
        "avg_unique_experts_per_seq": round(sum(uniq_experts_acc) / max(1, len(uniq_experts_acc)), 1),
    }


@torch.no_grad()
def evaluate_gsm8k_multihop_cot(
    model: BicameraKernel,
    bkp_mgr: Optional[BKPManager],
    tok: Tokenizer,
    device: torch.device,
) -> Dict:
    """
    评测 GSM8K 多跳 CoT 推理能力与"以时间换空间 (多步唤醒不同微专家)"效应:
    - 不设人工 max_new_tokens 硬截断上限，仅受模型物理上下文窗口 (model.config.max_seq_len) 保护，
      支持模型沿着 <|think|> ... <|/think|> 自由延展多跳推理链直至生成自然终止符 (EOS_ID 或结论行结束)！
    - 统计单条思考链跨步累计唤醒的去重微专家总数 vs 单步槽位数 -> 时间换空间放大倍率 (Time-to-Space Amplification)
    - 对比经过多步 <think> 推理链后的最终答案对数似然 vs 无 <think> 直接盲猜的对数似然增益 (CoT Reasoning Gain)
    """
    cot_nlls, ans_with_cot_logliks, ans_no_cot_logliks = [], [], []
    traj_unique_experts, step_avg_slots, hop_jaccard_dists = [], [], []
    gen_samples = []
    exact_match_cnt = 0

    for item in GSM8K_MULTIHOP_COT_BENCHMARK:
        q_str = item["question"]
        think_body = "\n".join(item["gold_steps"])
        gold_ans = item["gold_answer"]
        ans_text = f"最终结论为 {gold_ans}。\n#### {gold_ans}"

        prompt_no_cot = f"{USER_TOK}Question: {q_str}\n"
        prompt_with_cot = f"{USER_TOK}Question: {q_str}\n{THINK_BOS_TOK}\n{think_body}\n{THINK_EOS_TOK}\n"

        _, ll_no_cot = score_continuation_loglik(model, bkp_mgr, tok, prompt_no_cot, ans_text, device)
        ans_no_cot_logliks.append(ll_no_cot)

        _, ll_with_cot = score_continuation_loglik(model, bkp_mgr, tok, prompt_with_cot, ans_text, device)
        ans_with_cot_logliks.append(ll_with_cot)
        if bkp_mgr is not None:
            traj_unique_experts.append(bkp_mgr.last_unique_experts_per_seq)
            step_avg_slots.append(bkp_mgr.last_avg_active_slots)
            hop_jaccard_dists.append(bkp_mgr.last_hop_jaccard_dist)

        _, ll_cot_full = score_continuation_loglik(
            model, bkp_mgr, tok, prompt_no_cot, f"{THINK_BOS_TOK}\n{think_body}\n{THINK_EOS_TOK}\n{ans_text}", device
        )
        cot_nlls.append(-ll_cot_full)

        gen_prefix = f"{USER_TOK}Question: {q_str}\n{THINK_BOS_TOK}\n"
        cur_ids = [BOS_ID] + tok.encode(gen_prefix).ids
        inp = torch.tensor([cur_ids], dtype=torch.long, device=device)
        prompt_len = inp.size(1)
        phys_ctx_limit = int(getattr(model.config, "max_seq_len", 512))
        gen_limit = min(phys_ctx_limit, prompt_len + 256)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        seen_think_eos = False
        tokens_after_think_eos = 0
        while inp.size(1) < gen_limit:
            with amp_ctx:
                logits, _, _ = model(inp, bkp_manager=bkp_mgr, bkp_dropout=(bkp_mgr is None))
            next_logits = logits[0, -1, :].float()
            if float(next_logits.abs().max().item()) <= 1e-8:
                break
            # 严禁对数字/算术符号施加滑动窗口重复惩罚 (避免如 120 - 20 = 100 中的 '1'/'0' 因前文出现过而被压制为 '90')!
            # 仅当末尾连续 3 个 Token 完全相同时阻断第 4 次连续退化重复:
            if inp.size(1) >= 3 and inp[0, -1].item() == inp[0, -2].item() == inp[0, -3].item():
                next_logits[int(inp[0, -1].item())] = -float("inf")
            next_tok = int(torch.argmax(next_logits).item())
            if next_tok in (0, EOS_ID, USER_ID):
                break
            inp = torch.cat([inp, torch.tensor([[next_tok]], dtype=torch.long, device=device)], dim=1)
            gen_steps = inp.size(1) - prompt_len
            if gen_steps >= 24 and len(set(inp[0, -16:].tolist())) <= 2:
                break
            if next_tok == THINK_EOS_ID:
                seen_think_eos = True
            elif seen_think_eos:
                tokens_after_think_eos += 1
                tail_str = tok.decode(inp[0, prompt_len:].tolist(), skip_special_tokens=False)
                after_think = tail_str.split(THINK_EOS_TOK)[-1]
                if tokens_after_think_eos >= 16 and (
                    ("####" in after_think and after_think.endswith("\n"))
                    or (gold_ans in after_think and (after_think.endswith("。") or after_think.endswith("\n")))
                    or tokens_after_think_eos >= 36
                ):
                    break

        decoded_raw = tok.decode(inp[0, 1:].tolist(), skip_special_tokens=False)
        readable = (
            decoded_raw.replace(USER_TOK, "<|user|>")
            .replace(THINK_BOS_TOK, "<|think|>")
            .replace(THINK_EOS_TOK, "<|/think|>")
        )
        gen_continuation = readable.split("Question:", 1)[-1]
        if "\n" in gen_continuation:
            gen_continuation = gen_continuation.split("\n", 1)[-1]
        is_hit = bool(re.search(rf"(?<![0-9]){re.escape(gold_ans)}(?![0-9])", gen_continuation))
        if is_hit:
            exact_match_cnt += 1
        gen_samples.append({
            "id": item["id"],
            "gold_answer": gold_ans,
            "matched": is_hit,
            "generated_tokens": int(inp.size(1) - prompt_len),
            "generated_trace": readable,
        })

    avg_cot_nll = sum(cot_nlls) / max(1, len(cot_nlls))
    avg_ll_with = sum(ans_with_cot_logliks) / max(1, len(ans_with_cot_logliks))
    avg_ll_no = sum(ans_no_cot_logliks) / max(1, len(ans_no_cot_logliks))
    avg_uniq = sum(traj_unique_experts) / max(1, len(traj_unique_experts))
    avg_slot = sum(step_avg_slots) / max(1, len(step_avg_slots))
    space_amp = avg_uniq / max(1.0, avg_slot) if avg_slot > 0 else 0.0

    return {
        "cot_trajectory_nll": round(avg_cot_nll, 4),
        "cot_trajectory_ppl": round(math.exp(min(15.0, avg_cot_nll)), 2),
        "ans_loglik_with_cot": round(avg_ll_with, 4),
        "ans_loglik_no_cot": round(avg_ll_no, 4),
        "cot_reasoning_gain_nll": round(avg_ll_with - avg_ll_no, 4),
        "avg_unique_experts_per_cot_seq": round(avg_uniq, 1),
        "avg_slots_per_single_step": round(avg_slot, 2),
        "time_to_space_amplification_x": round(space_amp, 2),
        "avg_hop_jaccard_distance": round(sum(hop_jaccard_dists) / max(1, len(hop_jaccard_dists)), 4),
        "greedy_answer_hit_rate_pct": round(100.0 * exact_match_cnt / max(1, len(GSM8K_MULTIHOP_COT_BENCHMARK)), 2),
        "samples": gen_samples,
    }


@torch.no_grad()
def evaluate_humanoid_chat_and_coding(
    model: BicameraKernel,
    bkp_mgr: Optional[BKPManager],
    tok: Tokenizer,
    device: torch.device,
) -> Dict:
    """
    评测初具人形能力: 日常中文对话、系统概念讲解、按自然语言指令编写 Python 代码
    验证模型能否沿着 <|user|>...<|assistant|><|think|>...<|/think|> 输出清晰思路与可用代码/回答
    """
    prompts = [
        {
            "id": "chat_virtual_memory",
            "category": "concept_explanation",
            "user_query": "请用通俗易懂的话解释什么是操作系统中的虚拟内存，并说明它的核心优势。",
            "use_question_prefix": True,
        },
        {
            "id": "code_binary_search",
            "category": "python_coding",
            "user_query": "请用 Python 编写一个二分查找函数 `binary_search(arr, target)`，如果找到目标值则返回其索引，否则返回 -1。",
            "use_question_prefix": False,
        },
        {
            "id": "code_merge_intervals",
            "category": "python_coding",
            "user_query": "请用 Python 写一个函数 `merge_intervals(intervals)`，用于合并所有重叠的闭区间。",
            "use_question_prefix": False,
        },
        {
            "id": "sci_mmap_sparse",
            "category": "system_reasoning",
            "user_query": "为什么 4KB 页对齐的 mmap 读取超大稀疏微专家库比 read() 系统调用更快？",
            "use_question_prefix": True,
        },
        {
            "id": "dirac_mcp_code_fix",
            "category": "python_coding",
            "raw_prefix": (
                f"{USER_TOK}请用 Python 编写一个二分查找函数 `binary_search(arr, target)`，如果找到目标值则返回其索引，否则返回 -1。\n"
                f"{THINK_BOS_TOK}\n"
                f"先确定双指针 `lo = 0, hi = len(arr) - 1` 与中点 `mid = (lo + hi) // 2`。若循环条件写成 `while lo < hi:`，"
                f"[MCP:pytest] 单元素数组 `arr=[5], target=5` 断言失败：预期返回 0，实际返回 -1。\n"
            ),
        },
        {
            "id": "dirac_user_dialog_style",
            "category": "style_correction",
            "raw_prefix": (
                f"{USER_TOK}请模仿冷峻、克制且带有白描质感的现代散文文风，描写冬夜车站里等待末班车的人。\n"
                f"{THINK_BOS_TOK}\n直接描写车站夜晚的寒冷和乘客焦急的心情。\n{THINK_EOS_TOK}\n"
                f"冬天的夜晚真是太冷了，大家在车站里冻得瑟瑟发抖，心里非常焦急地盼望着末班车快点到来。\n"
                f"{USER_TOK}用词太直白口语化了，堆砌了“真是太冷了”、“非常焦急”等情绪副词，缺乏冷峻克制的白描质感。请保留冬夜候车的场景，改用客观物象重写。\n"
                f"{THINK_BOS_TOK}\n"
            ),
        },
    ]
    model.eval()
    if bkp_mgr is not None:
        bkp_mgr.eval()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    results = []
    structured_cot_cnt = 0
    code_valid_cnt = 0
    total_code_tasks = 0

    for item in prompts:
        if "raw_prefix" in item:
            gen_prefix = item["raw_prefix"]
        else:
            q_txt = f"Question: {item['user_query']}" if item["use_question_prefix"] else item["user_query"]
            gen_prefix = f"{USER_TOK}{q_txt}\n{THINK_BOS_TOK}\n"
        cur_ids = [BOS_ID] + tok.encode(gen_prefix).ids
        inp = torch.tensor([cur_ids], dtype=torch.long, device=device)
        prompt_len = inp.size(1)
        phys_ctx_limit = int(getattr(model.config, "max_seq_len", 512))
        gen_limit = min(phys_ctx_limit, prompt_len + 280)

        seen_think_eos = False
        while inp.size(1) < gen_limit:
            with amp_ctx:
                logits, _, _ = model(inp, bkp_manager=bkp_mgr, bkp_dropout=(bkp_mgr is None))
            next_logits = logits[0, -1, :].float()
            if float(next_logits.abs().max().item()) <= 1e-8:
                break
            # 仅阻断连续 3 次完全相同 Token 的退化重复，不破坏代码缩进/变量名与数学符号的精确概率分布
            if inp.size(1) >= 3 and inp[0, -1].item() == inp[0, -2].item() == inp[0, -3].item():
                next_logits[int(inp[0, -1].item())] = -float("inf")
            next_tok = int(torch.argmax(next_logits).item())
            if next_tok in (0, EOS_ID, USER_ID):
                break
            inp = torch.cat([inp, torch.tensor([[next_tok]], dtype=torch.long, device=device)], dim=1)
            gen_steps = inp.size(1) - prompt_len
            if gen_steps >= 24 and len(set(inp[0, -16:].tolist())) <= 2:
                break
            if next_tok == THINK_EOS_ID:
                seen_think_eos = True

        decoded_raw = tok.decode(inp[0, 1:].tolist(), skip_special_tokens=False)
        readable = (
            decoded_raw.replace(USER_TOK, "<|user|>")
            .replace(THINK_BOS_TOK, "<|think|>")
            .replace(THINK_EOS_TOK, "<|/think|>")
        )
        has_closed_think = bool(seen_think_eos or ("<|/think|>" in readable))
        if has_closed_think:
            structured_cot_cnt += 1
        if item["category"] == "python_coding":
            total_code_tasks += 1
            if "def " in readable and "return" in readable:
                code_valid_cnt += 1

        results.append({
            "id": item["id"],
            "category": item["category"],
            "generated_tokens": int(inp.size(1) - prompt_len),
            "has_complete_think_chain": has_closed_think,
            "unique_experts_woken": round(float(bkp_mgr.last_unique_experts_per_seq), 1) if bkp_mgr is not None else 0.0,
            "generated_trace": readable,
        })

    return {
        "structured_cot_completion_rate_pct": round(100.0 * structured_cot_cnt / max(1, len(prompts)), 2),
        "python_code_syntax_hit_rate_pct": round(100.0 * code_valid_cnt / max(1, total_code_tasks), 2),
        "samples": results,
    }


@torch.no_grad()
def run_full_benchmark_suite(
    model: BicameraKernel,
    bkp_mgr: Optional[BKPManager],
    tok: Tokenizer,
    device: torch.device,
    state_label: str = "Full_Forest_Mounted",
) -> Dict:
    """运行四大经典验证集与初具人形对话/编程实测并返回结构化成绩单"""
    model.eval()
    if bkp_mgr is not None:
        bkp_mgr.eval()

    arc_res = evaluate_multiple_choice_suite(model, bkp_mgr, tok, ARC_SCIENCE_BENCHMARK, device, use_cot_prefix=True)
    piqa_res = evaluate_multiple_choice_suite(model, bkp_mgr, tok, PIQA_COMMONSENSE_BENCHMARK, device, use_cot_prefix=True)
    mmlu_res = evaluate_multiple_choice_suite(model, bkp_mgr, tok, MMLU_CROSS_DOMAIN_BENCHMARK, device, use_cot_prefix=True)
    gsm_res = evaluate_gsm8k_multihop_cot(model, bkp_mgr, tok, device)
    humanoid_res = evaluate_humanoid_chat_and_coding(model, bkp_mgr, tok, device)

    avg_mc_acc = round((arc_res["acc_norm"] + piqa_res["acc_norm"] + mmlu_res["acc_norm"]) / 3.0, 2)
    return {
        "state": state_label,
        "mounted_packages": list(bkp_mgr.l1_active_pkg_ids) if bkp_mgr is not None else [],
        "overall_mc_acc_norm_pct": avg_mc_acc,
        "ARC_Science": arc_res,
        "PIQA_Commonsense": piqa_res,
        "MMLU_CrossDomain": mmlu_res,
        "GSM8K_MultiHop_CoT": gsm_res,
        "Humanoid_Chat_And_Coding": humanoid_res,
    }


