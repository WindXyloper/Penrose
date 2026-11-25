#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Penrose Studio — 独立 Streamlit 交互式前端 (penrose_ui.py)
支持:
1. 实时切换 BicameraKernel 内核版本；
2. 勾选/热插拔 .bkp 知识森林包，或一键拔除验证纯内核退化；
3. 跨核 BKPWaistBus 锚点仿射折叠校准 (Zero-Overhead Affine Folding)；
4. 自由切换 [常规直接对话 (Non-CoT)] 与 [深度思维链推理 (<|think|> CoT)]；
5. 支持无 Token 硬上限流式生成 (Unbounded RoPE Streaming)，点击 Stop 即可随时中断后端 GPU 推理。
"""

import json
import os
import urllib.request
import streamlit as st

API_BASE = os.environ.get("PENROSE_API_BASE", "http://127.0.0.1:8018")

st.set_page_config(
    page_title="Penrose Bicamera & .bkp Studio",
    page_icon="🧬",
    layout="wide",
)


def api_get(path: str) -> dict:
    req = urllib.request.Request(f"{API_BASE}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post(path: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


try:
    status = api_get("/v1/penrose/status")
except Exception as e:
    st.error(f"无法连接 Penrose 推理服务 ({API_BASE})，请先启动 `python serve_penrose_api.py`: {e}")
    st.stop()

with st.sidebar:
    st.title("🧬 Penrose Studio")
    st.caption(f"后端模式: `{status.get('mount_mode', 'RO_SYS')}` | 显存: `{status.get('gpu_vram_allocated_mb', 0)} MB`")

    st.subheader("1. 内核切换 (Kernel Host)")
    runs = status.get("available_runs", [])
    cur_run = status.get("active_run", "")
    sel_run = st.selectbox("选择活跃内核", options=runs, index=runs.index(cur_run) if cur_run in runs else 0)
    if sel_run != cur_run:
        if st.button("切换至该内核", use_container_width=True, type="primary"):
            api_post("/v1/penrose/switch", {"run_name": sel_run})
            st.rerun()

    pcie = status.get("pcie_vram_cache", {})
    col1, col2 = st.columns(2)
    col1.metric("唤醒微专家", f"{status.get('woken_experts', 0)} / {status.get('total_experts', 0)}")
    col2.metric("PCIe 缓存命中率", f"{pcie.get('hit_rate_pct', 0.0)}%")

    st.divider()
    st.subheader("2. .bkp 知识森林热插拔")
    avail_pkgs = status.get("available_packages", [])
    mounted_pkgs = set(status.get("mounted_packages", []))
    pkg_details = status.get("package_details", {})

    c_all, c_none = st.columns(2)
    if c_all.button("全挂森林", use_container_width=True):
        api_post("/v1/penrose/switch", {"run_name": cur_run, "active_bkps": avail_pkgs})
        st.rerun()
    if c_none.button("一键拔除 (纯核)", use_container_width=True):
        api_post("/v1/penrose/switch", {"run_name": cur_run, "active_bkps": []})
        st.rerun()

    selected_pkgs = []
    for p in avail_pkgs:
        woken_n = pkg_details.get(p, {}).get("woken_experts", 0)
        if st.checkbox(f"{p} ({woken_n} exp)", value=(p in mounted_pkgs), key=f"pkg_{p}"):
            selected_pkgs.append(p)

    if set(selected_pkgs) != mounted_pkgs:
        if st.button("应用 .bkp 挂载变更", use_container_width=True, type="primary"):
            api_post("/v1/penrose/switch", {"run_name": cur_run, "active_bkps": selected_pkgs})
            st.rerun()

    st.divider()
    st.subheader("3. 跨核 BKPWaistBus 仿射折叠")
    ref_run = st.selectbox("选择异源参考核 (.bkp 来源)", options=runs, index=0, key="ref_run_sel")
    gamma_val = st.slider("仿射混合比 γ (1.0=纯仿射)", 0.0, 1.0, 1.0, 0.1)
    fold_out = st.checkbox("同时折叠 P_out", value=False)
    if st.button("执行锚点校准并原地折叠", use_container_width=True):
        api_post(
            "/v1/penrose/calibrate_waist",
            {
                "reference_run": ref_run,
                "affine_gamma": gamma_val,
                "fold_p_out": fold_out,
                "mount_reference_bkps": True,
            },
        )
        st.rerun()

    cal_info = status.get("waist_bus_calibration", {})
    if cal_info.get("calibrated") and cal_info.get("metrics"):
        m = cal_info["metrics"]
        st.success(f"已折叠至 `{cal_info['calibrated_from_run']}` | R²={m.get('r2_folded_affine_pct')}% | Cos={m.get('calibrated_cosine_sim')}")

    st.divider()
    st.subheader("4. 推理与采样参数")
    enable_cot = st.toggle("启用 <|think|> 思维链模式", value=True, help="关闭后模型将作为常规非 CoT 语言模型直接回答")
    unbounded_gen = st.toggle("无上限生成 (直至自然结束)", value=True)
    max_new_tokens = 0 if unbounded_gen else st.slider("最大生成 Tokens", 32, 8192, 1024, 32)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
    rep_pen = st.slider("Repetition Penalty", 1.0, 1.3, 1.05, 0.01)

    if st.button("清空对话历史", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

st.subheader("💬 Penrose 双栖推理工作台 (支持流式中断与实时路由遥测)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("reasoning"):
            with st.expander("💭 展开查看 <|think|> 思维链推导", expanded=False):
                st.markdown(msg["reasoning"])
        st.markdown(msg["content"])
        if msg.get("telemetry"):
            t = msg["telemetry"]
            st.caption(
                f"⚡ 挂载包: `{t.get('mounted_packages', 0)}` | 唤醒微专家: `{t.get('unique_experts_woken', 0)}` | "
                f"时空放大: `{t.get('time_to_space_amplification_x', 0)}x` | PCIe 缓存命中率: `{t.get('pcie_vram_cache_hit_rate_pct', 0)}%`"
            )

prompt = st.chat_input("输入任何问题或代码需求 (支持非 CoT 直接回答与 <|think|> 深度推演)...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        think_expander = st.expander("💭 <|think|> 实时思维链流...", expanded=True) if enable_cot else None
        think_box = think_expander.empty() if think_expander is not None else None
        ans_box = st.empty()
        tel_box = st.empty()

        payload = {
            "model": f"penrose-{cur_run}",
            "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": rep_pen,
            "use_bkp": len(selected_pkgs) > 0,
            "enable_cot": enable_cot,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{API_BASE}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        acc_think = ""
        acc_ans = ""
        last_tel = {}
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                d_r = delta.get("reasoning_content", "")
                d_c = delta.get("content", "")
                if d_r:
                    acc_think += d_r
                    if think_box is not None:
                        think_box.markdown(acc_think + " ▌")
                if d_c:
                    acc_ans += d_c
                    ans_box.markdown(acc_ans + " ▌")
                last_tel = chunk.get("bkp_telemetry", last_tel)

        if think_box is not None:
            think_box.markdown(acc_think if acc_think else "*（无思考链或直接输出）*")
        final_ans = acc_ans if acc_ans else acc_think
        ans_box.markdown(final_ans)
        if last_tel:
            tel_box.caption(
                f"⚡ 挂载包: `{last_tel.get('mounted_packages', 0)}` | 唤醒微专家: `{last_tel.get('unique_experts_woken', 0)}` | "
                f"时空放大: `{last_tel.get('time_to_space_amplification_x', 0)}x` | PCIe 缓存命中率: `{last_tel.get('pcie_vram_cache_hit_rate_pct', 0)}%`"
            )
        st.session_state["messages"].append({
            "role": "assistant",
            "content": final_ans,
            "reasoning": acc_think if acc_ans else "",
            "telemetry": last_tel,
        })

