import os
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_WORKSPACE = Path(__file__).resolve().parent / "workspace"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(DEFAULT_WORKSPACE / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import gc
import re
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

_LONG_NONSPACE_RE = re.compile(r"\S{512,}")

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|sep|>",
    "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>", "<|reserved_3|>",
]
BOS_ID, EOS_ID, SEP_ID = 1, 2, 3
# 对话与连续思维流控制符 (复用原子 Special Token ID 4, 6, 7，零重训 Tokenizer):
# - USER_TOK        (ID=4, <|reserved_0|>): 用户提问或任务起点 <|user|>
# - THINK_BOS_TOK   (ID=6, <|reserved_2|>): 连续自然思维流起点 <|think|>
# - THINK_EOS_TOK   (ID=7, <|reserved_3|>): 连续自然思维流终点 <|/think|>
USER_TOK, THINK_BOS_TOK, THINK_EOS_TOK = (
    "<|reserved_0|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
)
ASST_TOK = ""
USER_ID, THINK_BOS_ID, THINK_EOS_ID = 4, 6, 7
ASST_ID = 5

from eval_bicamera_benchmarks import (
    get_all_benchmark_forbidden_substrings,
    is_text_contaminated_by_benchmark,
)
_BENCHMARK_FORBIDDEN_PATTERNS = get_all_benchmark_forbidden_substrings()

# 多路免授权高纯度开源语料配比 (涵盖高分中英教育语料、合成教科书、百科与代码):
CORPUS_SOURCES = [
    ("zh_general",    0.35, "opencsg/Fineweb-Edu-Chinese-V2.1", None, "train", 0),
    ("en_general",    0.30, "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0),
    ("cosmopedia_v2", 0.15, "HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", "train", 0),
    ("wiki_zh",       0.06, "wikimedia/wikipedia", "20231101.zh", "train", 0),
    ("wiki_en",       0.06, "wikimedia/wikipedia", "20231101.en", "train", 0),
    ("code_py",       0.08, "codeparrot/codeparrot-clean", None, "train", 0),
]

# 垂直领域 (Domain BKP) 专属语料配方:
# 每个元组: (分片名, 占比, HF仓库, 子集名, split, 跳过前N条避免与底座重复)
DOMAIN_PRESETS: Dict[str, List[Tuple[str, float, str, Optional[str], str, int]]] = {
    "general": CORPUS_SOURCES,
    "math": [
        ("math_web_en", 0.60, "open-web-math/open-web-math", None, "train", 0),
        ("math_edu_zh", 0.40, "opencsg/Fineweb-Edu-Chinese-V2.1", None, "train", 250000),
    ],
    "code": [
        ("code_py_deep",      0.45, "codeparrot/codeparrot-clean", None, "train", 50000),
        ("code_magicoder",    0.30, "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", 0),
        ("code_feedback_ins", 0.25, "m-a-p/CodeFeedback-Filtered-Instruction", None, "train", 0),
    ],
    "wiki_sci": [
        ("wiki_zh_deep",      0.35, "wikimedia/wikipedia", "20231101.zh", "train", 40000),
        ("wiki_en_deep",      0.35, "wikimedia/wikipedia", "20231101.en", "train", 40000),
        ("cosmopedia_sci_v2", 0.30, "HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", "train", 100000),
    ],
    "cot_distill": [
        ("cot_multi_hop", 1.00, "multi_source_real_cot_and_dialogue", None, "train", 0),
    ],
}

# 旧版合成算术/应用题黑名单指纹 (用于彻底清洗残留的合成模板，永绝"吊兰/曲奇"过拟合)
_LEGACY_SYNTH_BLACKLIST = (
    "苗圃里培育了",
    "吊兰的盆数",
    "多肉植物的盆数",
    "烘焙坊烤制了一批曲奇",
    "自动化包装线上午以",
    "ceramic bowls each shift",
    "计算算式：",
    "请直接给出 (",
    "解一元一次方程：",
    "的等差数列前 ",
    "已知系统初始基准量为",
    "递增等差数组 A[0]=",
    "已知二阶方阵 M =",
)

# 针对垂直领域的关键词轻量过滤器与本地源加速映射
DOMAIN_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "math": {
        "math_edu_zh": ["数学", "定理", "公式", "方程", "函数", "证明", "矩阵", "概率", "微积分", "几何", "代数", "求解", "="],
    },
    "code": {
        "code_edu_en": ["def ", "class ", "import ", "function", "algorithm", "return ", "```", "compiler", "database"],
    },
}

LOCAL_FALLBACK_RAW: Dict[str, str] = {
    "math_edu_zh": "zh_general.jsonl",
    "code_edu_en": "en_general.jsonl",
}


def init_workspace(root: Path, domain: str = "general", out_dataset: Optional[str] = None) -> dict:
    raw_dir = root / "raw_datasets" if domain == "general" else (root / "raw_datasets" / domain)
    if out_dataset:
        bin_dir = root / "datasets" / out_dataset
    elif domain == "general":
        bin_dir = root / "tokenized_bins"
    else:
        bin_dir = root / "datasets" / domain

    paths = {
        "root": root,
        "raw": raw_dir,
        "tok": root / "tokenizer",
        "bin": bin_dir,
        "bkp": root / "bkp_store",
        "ckpt": root / "checkpoints",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _count_existing_jsonl(file_path: Path) -> Tuple[int, int]:
    """统计已有 jsonl 文件的有效行数与字节数，支持无损断点续传 / 扩容追加"""
    if not file_path.exists():
        return 0, 0
    lines = 0
    size = file_path.stat().st_size
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            lines += 1
    return lines, size


def _passes_edu_score_filter(row: dict, repo_id: str, min_score: float = 4.0) -> bool:
    """
    针对 HuggingFaceFW/fineweb-edu 与 opencsg/Fineweb-Edu-Chinese-V2.1 执行教育质量高分档过滤:
    只保留 score >= 4.0 (或 int_score >= 4) 的最高分档样本。
    """
    if "fineweb-edu" not in repo_id.lower():
        return True
    raw_score = row.get("score")
    if raw_score is None:
        raw_score = row.get("int_score")
    if raw_score is None:
        return True
    try:
        return float(raw_score) >= min_score
    except Exception:
        return True


def _extract_row_text(row: dict, repo_id: str = "") -> str:
    """从不同开源仓库记录中提取高质量自然文本或代码问答文档"""
    if not _passes_edu_score_filter(row, repo_id, min_score=4.0):
        return ""
    txt = str(row.get("text") or row.get("content") or "").strip()
    if txt:
        return txt
    q = str(row.get("problem") or row.get("query") or row.get("instruction") or row.get("prompt") or "").strip()
    a = str(row.get("solution") or row.get("answer") or row.get("output") or row.get("response") or "").strip()
    if q and a:
        return f"{USER_TOK}{q}\n{a}"
    return ""


def _parse_hf_r1_row(row: dict, allow_direct_non_cot: bool = True, rng: Optional[random.Random] = None) -> Optional[str]:
    """
    将 HuggingFace 开源推理、多轮对话与代码指令数据集解析为训练样本:
    - 支持 Congliu/Chinese-DeepSeek-R1-Distill-data-110k, HuggingFaceTB/smoltalk,
      m-a-p/COIG-CQIA, BelleGroup/ShareGPT-Chinese, m-a-p/CodeFeedback-Filtered-Instruction,
      ise-uiuc/Magicoder-OSS-Instruct-75K；
    - 严禁截断思维链；严禁泄漏评测基准题；绝不生成任何合成模板。
    """
    user_q, think_txt, ans_txt = "", "", ""
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, list) and len(msgs) >= 2:
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or m.get("from") or "").lower()
            val = str(m.get("content") or m.get("value") or "").strip()
            if role in ("user", "human") and not user_q:
                user_q = val
            elif role in ("assistant", "gpt", "model") and val:
                if "<think>" in val and "</think>" in val:
                    p1, p2 = val.split("<think>", 1)[1].split("</think>", 1)
                    think_txt, ans_txt = p1.strip(), p2.strip()
                else:
                    think_txt = str(m.get("reasoning_content") or m.get("thought") or "").strip()
                    ans_txt = val
                break
    if not user_q:
        inst = str(
            row.get("instruction")
            or row.get("prompt")
            or row.get("question")
            or row.get("problem")
            or row.get("query")
            or ""
        ).strip()
        inp = str(row.get("input") or "").strip()
        user_q = f"{inst}\n{inp}".strip() if inp else inst
    if not ans_txt:
        raw_out = str(
            row.get("output")
            or row.get("response")
            or row.get("content")
            or row.get("solution")
            or row.get("answer")
            or ""
        ).strip()
        think_field = str(row.get("reasoning_content") or row.get("reasoning") or row.get("thought") or "").strip()
        if "<think>" in raw_out and "</think>" in raw_out:
            p1, p2 = raw_out.split("<think>", 1)[1].split("</think>", 1)
            think_txt, ans_txt = p1.strip(), p2.strip()
        else:
            think_txt, ans_txt = think_field, raw_out
    if len(user_q) < 4 or len(ans_txt) < 8:
        return None
    if any(b in user_q or b in ans_txt for b in _LEGACY_SYNTH_BLACKLIST):
        return None
    if is_text_contaminated_by_benchmark(user_q, _BENCHMARK_FORBIDDEN_PATTERNS) or is_text_contaminated_by_benchmark(ans_txt, _BENCHMARK_FORBIDDEN_PATTERNS):
        return None
    think_txt = re.sub(r"(?m)^Step\s*\d+\s*(\[[^\]]*\])?\s*[:：]?\s*", "", think_txt).strip()
    if not think_txt or len(think_txt) < 12:
        if allow_direct_non_cot:
            return f"{USER_TOK}{user_q}\n{ans_txt}"
        return None
    if allow_direct_non_cot and rng is not None and rng.random() < 0.15:
        return f"{USER_TOK}{user_q}\n{ans_txt}"
    return f"{USER_TOK}{user_q}\n{THINK_BOS_TOK}\n{think_txt}\n{THINK_EOS_TOK}\n{ans_txt}"


def _convert_raw_doc_to_multihop_cot(text: str, domain_tag: str, rng: random.Random) -> Optional[str]:
    """从真实语料中提取原生自然文本训练样本 (过滤评测基准与旧版合成模板)"""
    clean = text.strip()
    if len(clean) < 120:
        return None
    if any(b in clean for b in _LEGACY_SYNTH_BLACKLIST):
        return None
    if is_text_contaminated_by_benchmark(clean, _BENCHMARK_FORBIDDEN_PATTERNS):
        return None
    return clean


def _purge_legacy_synth_from_file(file_path: Path) -> Tuple[int, int]:
    """若已有 jsonl 文件中含有旧版合成模板 ('吊兰/曲奇/算式' 等)，原地彻底清除这些污染行"""
    if not file_path.is_file():
        return 0, 0
    clean_lines, removed, clean_bytes = [], 0, 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if any(b in line for b in _LEGACY_SYNTH_BLACKLIST):
                removed += 1
                continue
            ln = line if line.endswith("\n") else line + "\n"
            clean_lines.append(ln)
            clean_bytes += len(ln.encode("utf-8"))
    if removed > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(clean_lines)
        print(f"  [彻底净空] 从 {file_path.name} 铲除 {removed:,} 条合成模板，保留纯净样本 {len(clean_lines):,} 条 ({clean_bytes / 1e6:.1f} MB)")
    return len(clean_lines), clean_bytes


def _build_cot_distill_jsonl(paths: dict, out_file: Path, target_bytes: int, seed: int = 42) -> None:
    """构建 100% 真实开源 R1 思维链 + 多轮对话 (smoltalk/COIG-CQIA/Belle) + 代码指令 (CodeFeedback/Magicoder) 语料"""
    rng = random.Random(seed)
    existing_lines, existing_bytes = _purge_legacy_synth_from_file(out_file)
    if existing_bytes >= target_bytes * 0.98:
        print(f"  [跳过] {out_file.name} 已达目标体积 ({existing_bytes / 1e6:.1f} MB / {existing_lines:,} 条)")
        return

    print(f"  [真实多源 CoT 与指令拉取] 构建 R1 + SmolTalk + COIG-CQIA + Belle + CodeFeedback + Magicoder -> {out_file} ({target_bytes / 1e6:.1f} MB)")
    written = existing_bytes
    need_total = max(1, target_bytes - existing_bytes)
    hf_cot_sources = [
        ("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", None, "train", 0.25),
        ("Congliu/Chinese-DeepSeek-R1-Distill-data-110k", None, "train", 0.10),
        ("HuggingFaceTB/smoltalk", "all", "train", 0.20),
        ("m-a-p/COIG-CQIA", "zhihu", "train", 0.10),
        ("BelleGroup/train_1M_CN", None, "train", 0.10),
        ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train", 0.15),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", 0.10),
    ]
    if os.environ.get("PENROSE_SKIP_HF_R1") != "1":
        try:
            from datasets import load_dataset
            for repo_id, sub_name, sp_name, share_ratio in hf_cot_sources:
                if written >= target_bytes:
                    break
                repo_target = min(target_bytes, written + int(need_total * share_ratio * 1.25))
                try:
                    print(f"  [HF-Mirror 真实指令/CoT 流] 连接: {repo_id} (子集={sub_name}) ...")
                    ds_r1 = (
                        load_dataset(repo_id, name=sub_name, split=sp_name, streaming=True)
                        if sub_name else load_dataset(repo_id, split=sp_name, streaming=True)
                    )
                    r1_cnt = 0
                    with open(out_file, "a", encoding="utf-8") as dst_f, tqdm(
                        initial=written, total=repo_target, unit="B", unit_scale=True, desc=f"  拉取 {repo_id.split('/')[-1]}"
                    ) as pbar_r1:
                        for row in ds_r1:
                            formatted = _parse_hf_r1_row(row, allow_direct_non_cot=True, rng=rng)
                            if not formatted:
                                continue
                            out_line = json.dumps({"text": formatted, "source": repo_id}, ensure_ascii=False) + "\n"
                            b_len = len(out_line.encode("utf-8"))
                            dst_f.write(out_line)
                            written += b_len
                            r1_cnt += 1
                            pbar_r1.update(b_len)
                            if written >= repo_target:
                                dst_f.flush()
                                break
                    print(f"  [HF-Mirror 真实指令/CoT 流] 从 {repo_id} 吸收 {r1_cnt:,} 条真实样本！", flush=True)
                except Exception as e_repo:
                    print(f"  [INFO] 跳过 {repo_id} ({type(e_repo).__name__}: {e_repo})", flush=True)
        except Exception:
            pass

    if written < target_bytes:
        raw_root = paths["root"] / "raw_datasets"
        candidate_files = [
            ("code_magicoder", raw_root / "code" / "code_magicoder.jsonl"),
            ("code_feedback", raw_root / "code" / "code_feedback_ins.jsonl"),
            ("math", raw_root / "math" / "math_web_en.jsonl"),
            ("math_zh", raw_root / "math" / "math_edu_zh.jsonl"),
            ("code", raw_root / "code" / "code_py_deep.jsonl"),
            ("wiki_sci_zh", raw_root / "wiki_sci" / "wiki_zh_deep.jsonl"),
            ("general_zh", raw_root / "zh_general.jsonl"),
        ]
        open_streams = [(t, open(fp, "r", encoding="utf-8", errors="ignore")) for t, fp in candidate_files if fp.is_file()]
        if open_streams:
            with open(out_file, "a", encoding="utf-8") as dst_f, tqdm(
                initial=written, total=target_bytes, unit="B", unit_scale=True, desc="  补充真实多学科文档"
            ) as pbar:
                while written < target_bytes and open_streams:
                    batch_samples, active_streams = [], []
                    for tag, fh in open_streams:
                        line_raw = fh.readline()
                        if not line_raw:
                            fh.close()
                            continue
                        active_streams.append((tag, fh))
                        try:
                            cot_doc = _convert_raw_doc_to_multihop_cot(json.loads(line_raw).get("text", ""), tag, rng)
                            if cot_doc:
                                batch_samples.append((tag, cot_doc))
                        except Exception:
                            pass
                    open_streams = active_streams
                    if not batch_samples:
                        break
                    rng.shuffle(batch_samples)
                    for src_tag, s_txt in batch_samples:
                        out_line = json.dumps({"text": s_txt, "source": src_tag}, ensure_ascii=False) + "\n"
                        b_len = len(out_line.encode("utf-8"))
                        dst_f.write(out_line)
                        written += b_len
                        pbar.update(b_len)
                        if written >= target_bytes:
                            break
            for _, fh in open_streams:
                fh.close()


def step_download(paths: dict, total_mb: int, domain: str = "general"):
    if domain == "cot_distill":
        out_file = paths["raw"] / "cot_multi_hop.jsonl"
        _build_cot_distill_jsonl(paths, out_file, int(total_mb * 1024 * 1024))
        return
    sources = DOMAIN_PRESETS[domain]
    kw_map = DOMAIN_KEYWORDS.get(domain, {})
    print(f"\n>>> [Step 1/3] 流式拉取 [{domain}] 语料 (目标总计 ~{total_mb} MB) -> {paths['raw']}")

    for name, ratio, repo, subset, split, base_skip in sources:
        out_file = paths["raw"] / f"{name}.jsonl"
        target_bytes = int(total_mb * 1024 * 1024 * ratio)
        existing_lines, existing_bytes = _count_existing_jsonl(out_file)

        if existing_bytes >= target_bytes * 0.98:
            print(f"  [跳过] {out_file.name} 已达目标体积 ({existing_bytes / 1e6:.1f} MB / {existing_lines:,} 条)")
            continue

        need_bytes = target_bytes - existing_bytes
        skip_rows = base_skip + existing_lines
        keywords = kw_map.get(name)

        # 若存在本地通用大语料缓存且当前为关键词提纯分片，优先从本地高速流式提纯 (零网络/零显存开销)
        local_src_name = LOCAL_FALLBACK_RAW.get(name)
        local_src_path = paths["root"] / "raw_datasets" / local_src_name if local_src_name else None
        if local_src_path and local_src_path.is_file() and existing_bytes < target_bytes * 0.98:
            print(f"  [本地高速提纯] {name} <- 从本地 {local_src_path.name} 按领域关键词筛选 ({target_bytes / 1e6:.1f} MB)")
            written = existing_bytes
            with open(local_src_path, "r", encoding="utf-8", errors="ignore") as src_f, open(
                out_file, "a", encoding="utf-8"
            ) as dst_f, tqdm(
                initial=existing_bytes, total=target_bytes, unit="B", unit_scale=True, desc=f"  提纯 {name}"
            ) as pbar:
                for line in src_f:
                    try:
                        obj = json.loads(line)
                        text = (obj.get("text") or "").strip()
                    except Exception:
                        continue
                    if len(text) < 64:
                        continue
                    if keywords and not any(k in text for k in keywords):
                        continue
                    out_line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    b_len = len(out_line.encode("utf-8"))
                    dst_f.write(out_line)
                    written += b_len
                    pbar.update(b_len)
                    if written >= target_bytes:
                        break
            existing_lines, existing_bytes = _count_existing_jsonl(out_file)
            if existing_bytes >= target_bytes * 0.95:
                continue

        mode_str = f"断点续传追加 (+{need_bytes / 1e6:.1f} MB, 跳过前 {skip_rows:,} 条)" if existing_bytes > 0 else f"全新拉取 ({target_bytes / 1e6:.1f} MB)"
        print(f"  [启动] {name} ({repo}) -> {mode_str}")

        from datasets import load_dataset
        max_retries = 8
        for attempt in range(1, max_retries + 1):
            existing_lines, existing_bytes = _count_existing_jsonl(out_file)
            if existing_bytes >= target_bytes * 0.98:
                break
            cur_skip = base_skip + existing_lines
            keywords = kw_map.get(name)
            try:
                ds = load_dataset(repo, name=subset, split=split, streaming=True)
                if cur_skip > 0 and not keywords:
                    ds = ds.skip(cur_skip)
                    cur_skip = 0

                written = existing_bytes
                raw_seen = 0
                since_flush = 0

                with open(out_file, "a", encoding="utf-8") as f, tqdm(
                    initial=existing_bytes, total=target_bytes, unit="B", unit_scale=True, desc=f"  下载 {name}"
                ) as pbar:
                    for row in ds:
                        text = _extract_row_text(row, repo)
                        if len(text) < 64:
                            continue
                        if keywords and not any(k in text for k in keywords):
                            continue
                        if cur_skip > 0:
                            raw_seen += 1
                            if raw_seen <= cur_skip:
                                continue

                        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                        b_len = len(line.encode("utf-8"))
                        f.write(line)
                        written += b_len
                        since_flush += b_len
                        if since_flush >= 2 * 1024 * 1024:
                            f.flush()
                            since_flush = 0
                        pbar.update(b_len)
                        if written >= target_bytes:
                            f.flush()
                            break
                break
            except Exception as e:
                wait_s = min(3 * attempt, 15)
                print(
                    f"\n  [WARN] {name} 网络流闪断 (第 {attempt}/{max_retries} 次): {type(e).__name__}: {e} -> {wait_s}s 后基于本地断点自动续传...",
                    flush=True,
                )
                time.sleep(wait_s)
                if attempt == max_retries:
                    raise


def step_train_tokenizer(paths: dict, vocab_size: int, sample_mb: int):
    tok_path = paths["tok"] / "tokenizer.json"
    print(f"\n>>> [Step 2/3] 训练 {vocab_size} 词表双语+代码+数学+对话 Byte-Level BPE -> {tok_path}")

    raw_root = paths["root"] / "raw_datasets"

    def balanced_iterator():
        for name, ratio, *_ in CORPUS_SOURCES:
            fp = raw_root / f"{name}.jsonl"
            if not fp.is_file():
                continue
            budget = int(sample_mb * 1024 * 1024 * ratio)
            used = 0
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        text = json.loads(line).get("text", "")
                    except Exception:
                        continue
                    if not text:
                        continue
                    used += len(text.encode("utf-8"))
                    yield text
                    if used >= budget:
                        break
        # 额外纳入垂直领域 (math / code / wiki_sci / cot_distill) 语料，确保 64K 词表高密度覆盖推理与专业符号
        extra_files = [
            raw_root / "math" / "math_web_en.jsonl",
            raw_root / "math" / "math_edu_zh.jsonl",
            raw_root / "code" / "code_py_deep.jsonl",
            raw_root / "wiki_sci" / "wiki_zh_deep.jsonl",
            raw_root / "wiki_sci" / "wiki_en_deep.jsonl",
            raw_root / "cot_distill" / "cot_multi_hop.jsonl",
        ]
        extra_budget = int(sample_mb * 1024 * 1024 * 0.15)
        for efp in extra_files:
            if not efp.is_file():
                continue
            used = 0
            with open(efp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        text = json.loads(line).get("text", "")
                    except Exception:
                        continue
                    if not text:
                        continue
                    used += len(text.encode("utf-8"))
                    yield text
                    if used >= extra_budget:
                        break

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(balanced_iterator(), trainer=trainer)
    tokenizer.save(str(tok_path))

    # 快速自检可逆性与压缩率
    probe = "Bicamera 认知解耦验证: Base.bkp 页大小 4096 Bytes, x_id = 9527!\n"
    enc = tokenizer.encode(probe)
    assert tokenizer.decode(enc.ids) == probe, "Tokenizer 编解码非无损!"
    print(f"  [自检通过] 实际词表={tokenizer.get_vocab_size()} | 探针 {len(probe)} 字符 -> {len(enc.ids)} tokens")


def step_compile_bins(paths: dict, domain: str = "general", val_ratio: float = 0.01, seed: int = 42):
    tok_file = paths["tok"] / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(f"未找到统一分词器文件: {tok_file}，请先运行 --step tokenizer")
    tok = Tokenizer.from_file(str(tok_file))
    assert tok.get_vocab_size() <= 65536, f"词表 ({tok.get_vocab_size()}) 超出 uint16 容量上限 (65536)"

    sources = DOMAIN_PRESETS[domain]
    train_path = paths["bin"] / "train.bin"
    val_path = paths["bin"] / "val.bin"
    print(f"\n>>> [Step 3/3] 离线混洗编译 [{domain}] uint16 连续内存映射文件 -> {paths['bin']}")

    rng = random.Random(seed)
    handles = [
        (open(paths["raw"] / f"{name}.jsonl", "r", encoding="utf-8"), max(16, int(round(ratio * 256))))
        for name, ratio, *_ in sources
        if (paths["raw"] / f"{name}.jsonl").is_file()
    ]

    train_tokens, val_tokens = 0, 0
    doc_count = 0
    contaminated_skipped = 0
    sample_previews: List[str] = []
    max_doc_chars = 16384
    step_cnt = 0
    with open(train_path, "wb") as f_tr, open(val_path, "wb") as f_va:
        active = True
        pbar = tqdm(unit="tok", unit_scale=True, desc=f"  编译 {domain} tokens")
        while active:
            block = []
            active = False
            for fh, quota in handles:
                for _ in range(quota):
                    line = fh.readline()
                    if not line:
                        break
                    active = True
                    try:
                        obj = json.loads(line)
                        txt = obj["text"]
                        meta_parts = []
                        title_str = str(obj.get("title") or "").strip()
                        author_str = str(obj.get("author") or "").strip()
                        if title_str and title_str not in txt[:200]:
                            meta_parts.append(f"Title: {title_str}")
                        if author_str and author_str not in txt[:200]:
                            meta_parts.append(f"Author: {author_str}")
                        if meta_parts:
                            txt = " | ".join(meta_parts) + "\n" + txt
                    except Exception:
                        continue
                    # 强制编译期防泄漏与旧版合成模板守卫
                    if any(b in txt for b in _LEGACY_SYNTH_BLACKLIST) or is_text_contaminated_by_benchmark(txt, _BENCHMARK_FORBIDDEN_PATTERNS):
                        contaminated_skipped += 1
                        continue
                    if len(sample_previews) < 5 and len(txt) >= 64:
                        sample_previews.append(txt[:180].replace("\n", " "))
                    if _LONG_NONSPACE_RE.search(txt):
                        txt = _LONG_NONSPACE_RE.sub(" ", txt)
                    if len(txt) > max_doc_chars:
                        for pos in range(0, len(txt), max_doc_chars):
                            sub = txt[pos : pos + max_doc_chars]
                            if len(sub) >= 64:
                                block.append(sub)
                    elif len(txt) >= 64:
                        block.append(txt)
            if not block:
                break

            rng.shuffle(block)
            doc_count += len(block)
            tr_buf, va_buf = [], []
            for e in tok.encode_batch(block):
                doc_ids = [BOS_ID] + e.ids + [EOS_ID]
                if rng.random() < val_ratio:
                    va_buf.extend(doc_ids)
                else:
                    tr_buf.extend(doc_ids)

            if tr_buf:
                arr = np.asarray(tr_buf, dtype=np.uint16)
                f_tr.write(arr.tobytes())
                train_tokens += arr.size
            if va_buf:
                arr_v = np.asarray(va_buf, dtype=np.uint16)
                f_va.write(arr_v.tobytes())
                val_tokens += arr_v.size
            pbar.update(len(tr_buf) + len(va_buf))
            step_cnt += 1
            if step_cnt % 200 == 0:
                gc.collect()
        pbar.close()

    for fh, _ in handles:
        fh.close()

    meta = {
        "domain": domain,
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "sep_id": SEP_ID,
        "user_id": USER_ID,
        "assistant_id": ASST_ID,
        "think_bos_id": THINK_BOS_ID,
        "think_eos_id": THINK_EOS_ID,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "doc_count": doc_count,
        "bin_size_mb": round((train_tokens + val_tokens) * 2 / (1024 * 1024), 2),
        "contaminated_filtered_docs": contaminated_skipped,
        "sources": [f"{s[0]} ({s[2]})" for s in sources],
        "sample_previews": sample_previews,
        "tokenizer_path": str(tok_file),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(paths["bin"] / "meta.json", "w", encoding="utf-8") as fm:
        json.dump(meta, fm, indent=2, ensure_ascii=False)

    # 记录本地语料构建与投喂账本 (workspace/corpus_manifest.jsonl)，支持后续删除原始大文件后完整溯源
    manifest_file = paths["root"] / "corpus_manifest.jsonl"
    manifest_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": domain,
        "bin_dir": str(paths["bin"]),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "bin_size_mb": round((train_tokens + val_tokens) * 2 / (1024 * 1024), 2),
        "sources": [s[0] for s in sources],
    }
    with open(manifest_file, "a", encoding="utf-8") as f_man:
        f_man.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    print(f"  [完成] [{domain}] -> {paths['bin']} | train.bin: {train_tokens:,} tokens | val.bin: {val_tokens:,} tokens")
    print(f"  [账本] 已登记语料元信息至: {manifest_file}")


def _iter_existing_bin_docs(bin_dir: Path, old_tok: Tokenizer, max_docs: int = 500000):
    """当本地或远端的 raw .jsonl 已被清理时，直接从既有 uint16 train.bin / val.bin 经 Byte-Level BPE 100% 无损还原原始文档流"""
    count = 0
    for split_name in ["train.bin", "val.bin"]:
        bp = bin_dir / split_name
        if not bp.is_file():
            continue
        arr = np.memmap(str(bp), dtype=np.uint16, mode="r")
        n_total = len(arr)
        chunk_step = 1024 * 1024
        rem_ids: List[int] = []
        for start in range(0, n_total, chunk_step):
            chunk = arr[start : min(n_total, start + chunk_step)].tolist()
            if rem_ids:
                chunk = rem_ids + chunk
                rem_ids = []
            doc_spans = []
            cur_start = 0
            for idx, tid in enumerate(chunk):
                if tid == EOS_ID:
                    doc_ids = [t for t in chunk[cur_start:idx] if t != BOS_ID and t != 0]
                    if len(doc_ids) >= 16:
                        doc_spans.append(doc_ids)
                    cur_start = idx + 1
            if cur_start < len(chunk):
                rem_ids = chunk[cur_start:]
            if doc_spans:
                for txt in old_tok.decode_batch(doc_spans, skip_special_tokens=False):
                    clean_t = txt.strip()
                    if len(clean_t) >= 48:
                        yield clean_t
                        count += 1
                        if count >= max_docs:
                            return


def rebuild_all_domains_with_new_vocab(root: Path, vocab_size: int = 65536, sample_mb: int = 250, seed: int = 42) -> None:
    import shutil
    tok_path = root / "tokenizer" / "tokenizer.json"
    old_tok: Optional[Tokenizer] = Tokenizer.from_file(str(tok_path)) if tok_path.is_file() else None
    if old_tok is not None and old_tok.get_vocab_size() < vocab_size:
        legacy_bak = root / "tokenizer" / f"tokenizer_{old_tok.get_vocab_size()}_legacy.json"
        if not legacy_bak.is_file():
            shutil.copy2(tok_path, legacy_bak)
    domains = ["general", "math", "code", "wiki_sci", "cot_distill"]

    def _domain_doc_iterator(dom: str, byte_budget: Optional[int] = None):
        ws_d = init_workspace(root, domain=dom)
        raw_files = [ws_d["raw"] / f"{s[0]}.jsonl" for s in DOMAIN_PRESETS[dom] if (ws_d["raw"] / f"{s[0]}.jsonl").is_file()]
        used_bytes = 0
        if len(raw_files) > 0:
            for rf in raw_files:
                with open(rf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            txt = json.loads(line).get("text", "")
                        except Exception:
                            continue
                        if not txt or len(txt) < 48:
                            continue
                        yield txt
                        used_bytes += len(txt.encode("utf-8"))
                        if byte_budget is not None and used_bytes >= byte_budget:
                            return
        elif old_tok is not None and (ws_d["bin"] / "train.bin").is_file():
            for txt in _iter_existing_bin_docs(ws_d["bin"], old_tok):
                yield txt
                used_bytes += len(txt.encode("utf-8"))
                if byte_budget is not None and used_bytes >= byte_budget:
                    return

    print(f"\n>>> [Vocab Upgrade 1/2] 联合 5 大领域语料训练 {vocab_size} (64K) Byte-Level BPE -> {tok_path}", flush=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    per_dom_budget = int(sample_mb * 1024 * 1024 / len(domains))
    tokenizer.train_from_iterator(
        (doc for d in domains for doc in _domain_doc_iterator(d, byte_budget=per_dom_budget)),
        trainer=trainer,
    )
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tok_path))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  [Tokenizer 升级完成] 实际词表容量 = {actual_vocab:,}", flush=True)

    rng = random.Random(seed)
    for dom in domains:
        ws_d = init_workspace(root, domain=dom)
        if dom == "cot_distill" and (ws_d["raw"] / "cot_multi_hop.jsonl").is_file():
            refresh_cot_synth_and_compile(ws_d, total_mb=25, seed=seed)
            continue
        raw_files = [ws_d["raw"] / f"{s[0]}.jsonl" for s in DOMAIN_PRESETS[dom]]
        if all(rf.is_file() for rf in raw_files):
            step_compile_bins(ws_d, domain=dom, seed=seed)
        elif old_tok is not None and (ws_d["bin"] / "train.bin").is_file():
            print(f"\n>>> [Vocab Upgrade 2/2] 从既有 .bin 无损解码并用 {actual_vocab} 新词表重编译 [{dom}]...", flush=True)
            docs, contaminated_skipped = [], 0
            for doc in _iter_existing_bin_docs(ws_d["bin"], old_tok):
                if any(b in doc for b in _LEGACY_SYNTH_BLACKLIST) or is_text_contaminated_by_benchmark(doc, _BENCHMARK_FORBIDDEN_PATTERNS):
                    contaminated_skipped += 1
                    continue
                docs.append(doc)
            rng.shuffle(docs)
            train_tokens, val_tokens = 0, 0
            with open(ws_d["bin"] / "train.bin", "wb") as f_tr, open(ws_d["bin"] / "val.bin", "wb") as f_va:
                for b_start in range(0, len(docs), 512):
                    tr_buf, va_buf = [], []
                    for enc in tokenizer.encode_batch(docs[b_start : b_start + 512]):
                        doc_ids = [BOS_ID] + enc.ids + [EOS_ID]
                        (va_buf if rng.random() < 0.02 else tr_buf).extend(doc_ids)
                    if tr_buf:
                        arr_tr = np.asarray(tr_buf, dtype=np.uint16)
                        f_tr.write(arr_tr.tobytes())
                        train_tokens += arr_tr.size
                    if va_buf:
                        arr_va = np.asarray(va_buf, dtype=np.uint16)
                        f_va.write(arr_va.tobytes())
                        val_tokens += arr_va.size
            meta = {
                "domain": dom, "vocab_size": actual_vocab, "dtype": "uint16",
                "bos_id": BOS_ID, "eos_id": EOS_ID, "sep_id": SEP_ID,
                "user_id": USER_ID, "assistant_id": ASST_ID,
                "think_bos_id": THINK_BOS_ID, "think_eos_id": THINK_EOS_ID,
                "train_tokens": train_tokens, "val_tokens": val_tokens,
                "doc_count": len(docs),
                "bin_size_mb": round((train_tokens + val_tokens) * 2 / (1024 * 1024), 2),
                "contaminated_filtered_docs": contaminated_skipped,
                "sample_previews": [d[:180].replace("\n", " ") for d in docs[:5]],
                "tokenizer_path": str(tok_path),
            }
            with open(ws_d["bin"] / "meta.json", "w", encoding="utf-8") as fm:
                json.dump(meta, fm, indent=2, ensure_ascii=False)
            print(f"  [重编译完成] [{dom}] vocab={actual_vocab:,} | train={train_tokens:,} | val={val_tokens:,}", flush=True)


def refresh_cot_synth_and_compile(paths: dict, total_mb: int = 120, seed: int = 42) -> None:
    """彻底清除旧版合成算术模板，重新拉取 100% 真实多源 R1 思维链、SmolTalk、COIG-CQIA、Belle 与代码指令并重编译 .bin"""
    out_file = paths["raw"] / "cot_multi_hop.jsonl"
    target_bytes = int(total_mb * 1024 * 1024)
    _purge_legacy_synth_from_file(out_file)
    _build_cot_distill_jsonl(paths, out_file, target_bytes, seed=seed)
    step_compile_bins(paths, domain="cot_distill", seed=seed)


def expand_and_clean_remote_corpora(root: Path, cot_mb: int = 60, extra_mb: int = 30, seed: int = 42) -> None:
    """在远端彻底清洗重建 cot_distill 并为全领域增量合入 7 大精品源，生成带真实样本概括的 meta.json"""
    from datasets import load_dataset
    rng = random.Random(seed)
    tok_path = root / "tokenizer" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    actual_vocab = tok.get_vocab_size()

    ws_cot = init_workspace(root, domain="cot_distill")
    cot_raw = ws_cot["raw"] / "cot_multi_hop.jsonl"
    if cot_raw.is_file():
        cot_raw.unlink()
    for bf in ["train.bin", "val.bin", "meta.json"]:
        p_b = ws_cot["bin"] / bf
        if p_b.is_file():
            p_b.unlink()
    print(f"\n>>> [1/2] 已彻底清空旧版 cot_distill，开始拉取 100% 真实多源对话/思维链/代码指令 ({cot_mb} MB)...", flush=True)
    _build_cot_distill_jsonl(ws_cot, cot_raw, int(cot_mb * 1024 * 1024), seed=seed)
    step_compile_bins(ws_cot, domain="cot_distill", seed=seed)
    if cot_raw.is_file():
        cot_raw.unlink()

    inc_specs = {
        "code": [
            ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", 0.55),
            ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train", 0.45),
        ],
        "wiki_sci": [
            ("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", "train", 1.00),
        ],
        "general": [
            ("opencsg/Fineweb-Edu-Chinese-V2.1", None, "train", 0.40),
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.35),
            ("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", "train", 0.25),
        ],
        "math": [
            ("opencsg/Fineweb-Edu-Chinese-V2.1", None, "train", 1.00),
        ],
    }
    for dom in ["code", "wiki_sci", "general", "math"]:
        ws_d = init_workspace(root, domain=dom)
        tr_p, va_p = ws_d["bin"] / "train.bin", ws_d["bin"] / "val.bin"
        new_docs: List[str] = []
        target_b = int((extra_mb if dom != "math" else max(10, extra_mb // 3)) * 1024 * 1024)
        for repo_id, sub_name, sp_name, ratio in inc_specs.get(dom, []):
            quota_b, got_b = int(target_b * ratio), 0
            try:
                print(f"  [增量扩容 {dom}] 流式拉取高分源 {repo_id} (+{quota_b / 1e6:.1f} MB)...", flush=True)
                ds = load_dataset(repo_id, name=sub_name, split=sp_name, streaming=True) if sub_name else load_dataset(repo_id, split=sp_name, streaming=True)
                if dom in ("general", "math") and "fineweb" in repo_id.lower():
                    ds = ds.skip(300000)
                for row in ds:
                    txt = _extract_row_text(row, repo_id)
                    if len(txt) < 80:
                        continue
                    if dom == "math" and not any(k in txt for k in DOMAIN_KEYWORDS["math"]["math_edu_zh"]):
                        continue
                    if any(b in txt for b in _LEGACY_SYNTH_BLACKLIST) or is_text_contaminated_by_benchmark(txt, _BENCHMARK_FORBIDDEN_PATTERNS):
                        continue
                    new_docs.append(txt[:16384])
                    got_b += len(txt.encode("utf-8"))
                    if got_b >= quota_b:
                        break
            except Exception as e:
                print(f"  [INFO] 跳过 {repo_id}: {e}", flush=True)

        if new_docs:
            rng.shuffle(new_docs)
            with open(tr_p, "ab") as f_tr, open(va_p, "ab") as f_va:
                for b_start in range(0, len(new_docs), 512):
                    tr_buf, va_buf = [], []
                    for enc in tok.encode_batch(new_docs[b_start : b_start + 512]):
                        doc_ids = [BOS_ID] + enc.ids + [EOS_ID]
                        (va_buf if rng.random() < 0.015 else tr_buf).extend(doc_ids)
                    if tr_buf:
                        f_tr.write(np.asarray(tr_buf, dtype=np.uint16).tobytes())
                    if va_buf:
                        f_va.write(np.asarray(va_buf, dtype=np.uint16).tobytes())

        tr_tok = tr_p.stat().st_size // 2 if tr_p.is_file() else 0
        va_tok = va_p.stat().st_size // 2 if va_p.is_file() else 0
        previews = [d[:180].replace("\n", " ") for d in new_docs[:3]]
        for old_d in _iter_existing_bin_docs(ws_d["bin"], tok, max_docs=5):
            if len(previews) >= 5:
                break
            previews.append(old_d[:180].replace("\n", " "))
        meta = {
            "domain": dom, "vocab_size": actual_vocab, "dtype": "uint16",
            "bos_id": BOS_ID, "eos_id": EOS_ID, "sep_id": SEP_ID,
            "user_id": USER_ID, "assistant_id": ASST_ID,
            "think_bos_id": THINK_BOS_ID, "think_eos_id": THINK_EOS_ID,
            "train_tokens": tr_tok, "val_tokens": va_tok,
            "bin_size_mb": round((tr_tok + va_tok) * 2 / (1024 * 1024), 2),
            "sources": [f"{s[0]} ({s[2]})" for s in DOMAIN_PRESETS[dom]],
            "sample_previews": previews,
            "tokenizer_path": str(tok_path),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(ws_d["bin"] / "meta.json", "w", encoding="utf-8") as fm:
            json.dump(meta, fm, indent=2, ensure_ascii=False)
        print(f"  [扩容完成] [{dom}] train={tr_tok:,} | val={va_tok:,} | size={meta['bin_size_mb']} MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Penrose 离线双语底座与多领域 (Domain BKP) 语料构建管线")
    ap.add_argument("--workspace", type=str, default=str(DEFAULT_WORKSPACE))
    ap.add_argument(
        "--domain",
        choices=list(DOMAIN_PRESETS.keys()),
        default="general",
        help="语料领域配方: general(通用底座), math(数学推理), code(代码工程), wiki_sci(高密度百科科学), cot_distill(R1对话与多跳思维链)",
    )
    ap.add_argument(
        "--out-dataset",
        type=str,
        default=None,
        help="自定义输出数据集子目录名 (位于 workspace/datasets/<name>，不指定则 general 输出到 tokenized_bins，垂直领域输出到 datasets/<domain>)",
    )
    ap.add_argument("--step", choices=["all", "download", "tokenizer", "compile"], default="all")
    ap.add_argument("--total-mb", type=int, default=1200, help="拉取原始语料目标总体积(MB)，支持断点续传追加扩容")
    ap.add_argument("--tok-sample-mb", type=int, default=250, help="用于训练 BPE 的均衡采样体积(MB)")
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--clean-raw", action="store_true", help="编译为 .bin 后立即删除本地 raw_datasets 下的 .jsonl 原始大文件(仅留账本 log)")
    ap.add_argument("--refresh-cot-synth", action="store_true", help="将 cot_distill 中的合成算术题刷新为 100% 真实多源语料并重编译 .bin")
    ap.add_argument("--rebuild-all-domains", action="store_true", help="升级训练新词表 (默认 65536) 并一键重编译全部已有领域的 .bin 文件")
    ap.add_argument("--expand-remote", action="store_true", help="在远端彻底清洗重建 cot_distill 并为全领域增量合入 7 大精品源")
    ap.add_argument("--cot-mb", type=int, default=80, help="--expand-remote 模式下 cot_distill 目标拉取体积(MB)")
    ap.add_argument("--extra-mb", type=int, default=40, help="--expand-remote 模式下每个其余领域增量扩容体积(MB)")
    args = ap.parse_args()

    ws = init_workspace(Path(args.workspace), domain=args.domain, out_dataset=args.out_dataset)
    if args.expand_remote:
        expand_and_clean_remote_corpora(Path(args.workspace), cot_mb=args.cot_mb, extra_mb=args.extra_mb)
        raise SystemExit(0)
    if args.rebuild_all_domains:
        rebuild_all_domains_with_new_vocab(Path(args.workspace), vocab_size=args.vocab_size, sample_mb=args.tok_sample_mb)
        raise SystemExit(0)
    if args.refresh_cot_synth:
        refresh_cot_synth_and_compile(ws, total_mb=args.total_mb)
        raise SystemExit(0)
    if args.step in ("all", "download"):
        step_download(ws, args.total_mb, domain=args.domain)
    if args.step == "tokenizer" or (args.step == "all" and args.domain == "general" and not (ws["tok"] / "tokenizer.json").exists()):
        step_train_tokenizer(ws, args.vocab_size, args.tok_sample_mb)
    if args.step in ("all", "compile"):
        step_compile_bins(ws, domain=args.domain)
        if args.clean_raw and args.domain != "general":
            for name, *_ in DOMAIN_PRESETS[args.domain]:
                raw_f = ws["raw"] / f"{name}.jsonl"
                if raw_f.is_file():
                    raw_f.unlink()
                    print(f"  [清理] 已删除本地原始语料缓存: {raw_f} (元信息已留存于 corpus_manifest.jsonl)")