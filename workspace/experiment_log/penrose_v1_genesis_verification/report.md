# Penrose v1.0 创世架构契约与八大流形定理第一性原理去伪存真审计报告 (`penrose_v1_genesis_verification`)

- **测试脚本**: `F:\Penrose\test_remote.py` & `F:\Penrose\train_penrose_v1.py`
- **核心模块**: `penrose_pmc.py` (`PMC`), `penrose_mpu.py` (`MPU`), `penrose_harness.py` (`Penrose Harness`)
- **审计原则**: **100% 剔除 `torch.randn` 合成向量、100% 剔除外部 Python 词频字典 (`_cjk_bigram_counts`) 与手工 `nucleate_composite_word_sink` 造词、100% 剔除手工修改 `u_codebook / v_codebook / activation_counts / stream_next_ptr / _pkg_coactivation_matrix`**。所有测试仅输入真实 UTF-8 自然语言语料（加载自 `workspace/real_raw_corpus_sample.jsonl`）并通过标准 `geodesic_kaczmarz_snap_update` / `train_manifold_trajectory` 演化，如实记录成立定律与暴露的模型冷启动 Gap。
- **原始 JSON 报告**: `workspace/experiment_log/penrose_v1_genesis_verification/report.json`

---

## 一、 八大核心契约与 15 步端到端训练去伪存真审计汇总

### 1. $2048 = 8 \times 256$ 沃尔什-阿达玛全息波包换能器 (`exp1_holographic_wavepacket_8x256`) — **[VERIFIED]**
- **可学习参数量 (`learnable_params`)**: **`0`**（彻底废除 `Tokenizer`、`nn.Embedding` 与 `LM Head`）
- **单字节正交内积 (`inter_byte_dot`)**: **`0.00000000`**
- **跨槽位正交内积 (`inter_slot_dot`)**: **`0.00000000`**
- **8 注意力头能量均匀度误差 (`head_energy_uniformity_err`)**: **`1.0e-8`**（每头能量恒为 $1/8 = 0.125$）
- **公理 2 Gumbel 孔径自适应长度无损还原率 (`roundtrip_accuracy`)**: **`9/9 (100.0%)`**
- **非阿贝尔顺序区分度 (`ab_vs_ba_cosine`)**: **`0.000000`**

### 2. 双层物理相变 + `PMC` 负散度漏斗流式切分 (`exp2_two_tier_negative_divergence_funnel`) — **[MODEL_GAP_EXPOSED]**
- **输入测试脏语料**: `'{text: "这是一段训练语料。"}'`
- **第一层物理相变门隔离前导格式壳 (`shell_prefix_isolated`)**: `['{', 'text', ': "']`（第一层 UTF-8 脚本范畴 / ASCII 符号边界相变门 **100% 真实有效**）
- **第 0 步冷启动空白流形切分 (`cold_start_chunks`, 14 波包)**:
  `['{', 'text', ': "', '这', '是', '一', '段', '训', '练', '语', '料', '。', '"', '}']`
- **废除外部字典与手工 `nucleate_composite_word_sink` 后，仅靠 `train_manifold_trajectory` 训练 3 轮后的切分 (`warm_funnel_chunks`, 14 波包)**:
  `['{', 'text', ': "', '这', '是', '一', '段', '训', '练', '语', '料', '。', '"', '}']`（`funnel_collapsed_without_external_injection = false`）
- **第一性原理根因诊断（暴露的模型 Gap）**:
  - 冷启动时第一层相变门将中文字符切分为单字波包 $q(\text{"训"}), q(\text{"练"})$，`geodesic_kaczmarz_snap_update` 在单字锚点 $q(\text{"训"})$ 处记录转移并累加 `activation_counts`，但从未在复合坐标 $q(\text{"训练"})$ 处写入微专家；
  - 因此第二层漏斗直接用复合波包 $q(\text{"训练"})$ 查询 `PMC` 时无法命中单字锚点。后续改进应让第二层漏斗沿热流线表 `stream_next_ptr` 检测高转移频次的相邻单字流线边 $e(c_1) \to e(c_2)$ 并自发坍缩合成复合波包锚点。

### 3. `MPU` 五段式纯球面流水线与三态总线分支 (`exp3_mpu_five_module_and_tri_state_bus`) — **[VERIFIED]**
- **静态词表大小 (`vocab_size`)**: **`0`**
- **流水线拓扑**: Module 1 (1-Jet Ingress) $\to$ Module 2 (前 6 层 8 头割线积分栈) $\to$ Module 3 (腰部 `PMC` 单位残差门) $\to$ Module 4 (后 6 层 8 头曲率精炼栈) $\to$ Module 5 (跨核球面 ResNet 切空间正交闭环)
- **Module 1 $\psi^{(0)}(\tau)$ 单位球面范数误差**: **`6.0e-8`**（并在第 0 微秒自动触发异步预取 **`24`** 个微专家页）
- **Module 5 切空间正交化误差 $\langle \Delta q_\perp(\tau), q(\tau)\rangle$**: **`1.8e-7`**
- **Module 5 输出 $q(\tau+1)$ 单位球面范数误差**: **`1.2e-7`**
- **分支 C 真实语料波包球面弦长损失 (`branch_c_chord_loss`)**: **`0.987449`**
- **三态总线分支轨迹 (`tri_state_branch_trace`)**: `["branch_a_curl_loop", "branch_a_curl_loop", "branch_b_sink_emit"]`

### 4. 无 `AdamW` 的 Kaczmarz + 罗德里格斯大圆旋转一步锁死与逆向拖拽精准擦除 (`exp4_adamw_free_kaczmarz_and_dirty_shell`) — **[VERIFIED]**
- **`PMC` 优化器动量显存占用 (`pmc_optimizer_memory_bytes`)**: **`0 Bytes`**
- **脏文档 1 单步 Kaczmarz 弦长误差下降 (`doc1_loss_before_to_after`)**: **`0.9905 -> 0.0840`（单步直降 `-91.5%`）**
- **脏文档 2 单步 Kaczmarz 弦长误差下降 (`doc2_loss_before_to_after`)**: **`0.7538 -> 0.1593`（单步直降 `-78.9%`）**
- **真实语料触发 Kaczmarz 测地吸附与有丝分裂**: `total_kaczmarz_snaps = 31`, `total_mitosis_spawns = 15`
- **干净提问时 JSON 格式符垃圾泄露 (`clean_query_json_leakage`)**: **`false`（0% 泄露）**
- **逆向拖拽擦除 `drag_and_drop_erase(doc_2)` 后 `doc_2` 弦长误差回退 (`doc2_erase_loss_before_to_after`)**: **`0.1630 -> 0.6990`（显著回退至未学习状态）**
- **擦除 `doc_2` 触发公理 9 独占微专家页细胞凋亡清零数 (`erase_apoptosis_pages`)**: **`12` 页**
- **擦除 `doc_2` 触发共享外壳微专家页秩-1 正交剥离数 (`erase_peeled_shared_pages`)**: **`4` 页**
- **擦除 `doc_2` 后共享流形中 `doc_1` 记忆保留弦长误差 (`doc1_loss_retained_after_doc2_erase`)**: **`0.0876`（对比擦除前 `0.0840` 误差仅波动 `0.0036`）**

### 5. 流形穿孔邻域三次厄米特 $C^1$ 单纯形插值自愈 (`exp5_manifold_puncture_c1_simplex_healing`) — **[VERIFIED]**
- **测试方式（零手工放置邻域码本）**: 直接将 `workspace/real_raw_corpus_sample.jsonl` 中 12 条包含 `"地球"` 的真实科学语料喂入 `geodesic_kaczmarz_snap_update(enable_auto_mitosis=True)`，让 `"地"` 在同一子键行自然分裂出伴随微专家簇。
- **硬性拔除中心微专家 (`punctured_center_eid`)**: `1092`（对应真实语料波包 `"地"`）
- **单纯形邻域自然伴随微专家 (`healed_top_neighbors`)**: `[1117, 1107, 1113]`
- **$C^1$ 单纯形邻域插值自愈余弦 (`c1_simplex_healing_cosine`)**: **`0.998858`**

### 6. 森林自治算子、非对称谱双聚类与多坐标卡图册 (`exp6_six_autonomous_forest_ops_and_wormhole`) — **[PARTIAL / HONESTLY RECORDED]**
- **测试方式（零手工注入 `activation_counts` / `stream_next_ptr` / `_pkg_coactivation_matrix`）**: 仅向 `grid_k=64` 种子包连续喂入 4 大领域（`wiki_sci`, `math`, `code`, `cot_distill`）各 8 条真实自然语言长句并调用 `step_autonomous_forest_evolution`。
- **所有衍生块类型均为 `PMCPackage` (`all_packages_are_pmc`)**: **`true`**
- **四叉树扩容零步误差 (`expand_zero_step_err`)**: **`5.4e-7`**，流线指针重映射 `5684 == 5684`
- **真实语料单轮扫描自发触发的算子 (`operator_totals`)**:
  - `Auto-Expand`: **`true`** (`expand_count = 1`, `needs_expand_triggered = true`)
  - `Auto-Split` + `BridgeAB`: **`true`** (`split_count = 2`, 非对称谱双聚类自然切出 `asymmetric_split_grid_ks = [64, 128]`, `has_bridge_pkg = true`)
  - `Auto-Closure`: **`true`** (`closure_count = 352`)
  - `Auto-Mitosis`: **`true`** (`mitosis_spawns = 146`, `kaczmarz_snaps = 689`)
  - `Auto-Merge`: 在 `exp6` 的 3 步静态探测中为 `false`（但在下方 15 步端到端循环训练的第 15 步中自发触发 `TriadicMerge_1_s15.pmc`！）
  - `Auto-Spawn`: **`false`**（暴露冷启动特性：当种子包挂载为可写基底包时，新语料直接写入基底包并通过 `Auto-Expand` / `Auto-Split` 扩容分化，因此 residual anomaly 不会越过 `Auto-Spawn` 孤儿孵化阈值）
  - **母体超球面共振归属路由 (`routed_chart_id`)**: `ForestSeed_SubB_s2.pmc`

### 7. 脏数据输入无时序分槽与 `Auto-Split` 独立分块 + 测地圆自然低唤醒 (`exp7_dirty_data_auto_split_and_low_wake`) — **[VERIFIED]**
- **微观真假输入无时序效应（Order Invariance，真实词元 `"牛顿第二定律"` $\to$ `"加速度正比于合力"` vs `"永动机无中生有"`）**:
  - **「先喂 5 次真事实、后喂 1 次伪造篡改」分槽计数 (`order_invariance_counts_truth_first`)**: **`[5, 1]`**
  - **「先喂 1 次伪造篡改、后喂 5 次真事实」分槽计数 (`order_invariance_counts_fake_first`)**: **`[5, 1]`**
  - **两种相反输入时序下主干真微专家的恢复余弦差 (`order_invariance_truth_cosine_diff`)**: **`3.28e-6`（严格时序无关）**
- **宏观伪造词条簇 `Auto-Split` 自动划归独立 `.pmc` 块与测地圆自然静默（零手动设置流线图，纯靠真实物理段落 vs JSON/HTML 广告垃圾段落自然训练）**:
  - **是否自发触发 `Auto-Split` (`has_auto_split`)**: **`true`**（切分为 2 个非桥接子包）
  - **主干真实知识子包 (`split_clean_pmc_id`)**: `MixedCorpus_SubB_s1.pmc (grid_k=128)`
  - **自动剥离出的伪造脏数据孤岛子包 (`split_dirty_pmc_id`)**: `MixedCorpus_SubA_s1.pmc (grid_k=64)`
  - **正常主干查询下主干块唤醒命中槽位数 (`clean_query_clean_pmc_slot_hits`)**: **`144`**
  - **正常主干查询下脏数据块唤醒命中槽位数 (`clean_query_dirty_pmc_slot_hits`)**: **`0`（脏块误唤醒率 `0.0%`）**

### 8. `Penrose v1.0` 五维时间与空间复杂度标度律实测 (`exp8_five_dimensional_complexity_scaling`) — **[4/5 VERIFIED, DIM 4 COLD-START GAP EXPOSED]**
- **维度 1 — 零词表 I/O 复杂度 $\mathcal{O}(d)$ (`dim1_vocab_size_and_transducer_params`)**: **`[0, 0]`**
- **维度 2 — 序列长度 $N$ 扩大 $8\times$（$64 \to 512$ 真实语料波包）的 $\mathcal{O}(N)$ 时间与 $\mathcal{O}(1)$ 状态显存**:
  - **跨窗口活跃状态显存比值 (`dim2_seq_len_8x_state_mem_ratio`)**: **`1.0000`（严格 $\mathcal{O}(1)$ 常数显存）**
  - **端到端流式耗时比值 (`dim2_seq_len_8x_time_ratio`)**: **`6.96x`（严格符合 $\mathcal{O}(N)$ 线性时间标度）**
- **维度 3 — 知识库微专家总数 $M_{\text{chunk}}$ 扩大 $16\times$（$4,096 \to 65,536$）的寻址复杂度**:
  - **微专家总数扩容倍数 (`dim3_experts_16x_scale_4096_to_65536`)**: **`16.0x`**
  - **冷启动乘积子键内积宽度增长倍数 (`dim3_cold_subkey_ops_scale_sqrt_m`)**: **`4.0x`（严格遵循 $\mathcal{O}(\sqrt{M_{\text{chunk}}})$）**
  - **热流线 `stream_next_ptr` 直接预取跳数 (`dim3_hot_streamline_pointer_hops`)**: **`1`（严格 $\mathcal{O}(1)$）**
- **维度 4 — 推理深度 $K$ 的流线虫洞坍缩 (`dim4_reasoning_hops_before_and_after_wormhole`) — [COLD-START COLLISION EXPOSED]**:
  - 当废除手动指定 `eids = [2, 9, 18, ...]` 链表，直接将英文短串 `"Premise_0 Step_1 Step_2 ... Conclusion_7"` 在冷启动未中心化的 `WormholeDepth.pmc` 上做单次 `geodesic_kaczmarz_snap_update` 时，实测 `dim4_reasoning_hops_before_and_after_wormhole = [0, -1]`。
  - **原因诊断**: 在未预热的空白码本上，`update_center=False` 导致首尾 ASCII 短词落入同一子键格点 (`chain_eids[0] == chain_eids[-1]`)，揭示了冷启动码本在短序列单次写入前必须先经过球面中心化预热 (`update_center=True`) 或正交码本重锚定。
- **维度 5 — 局部拖拽训练/擦除 $\mathcal{O}(1)$ 更新与零优化器显存 (`M = 16,384` 微专家)**:
  - **单步更新实际触碰微专家页数 (`dim5_16384_experts_touched_pages`)**: **`2` 页**
  - **完全零触碰微专家占比 (`dim5_untouched_experts_pct`)**: **`99.9878%`**
  - **优化器动量显存占用 (`dim5_optimizer_memory_bytes`)**: **`0 Bytes`**

### 9. 15 步端到端免清洗拖拽流形训练 (`end_to_end_training_15steps`) — **[VERIFIED: EXPAND + SPLIT + BRIDGE + MERGE + MITOSIS + CLOSURE ALL EMERGE NATURALLY]**
- `Step 01`: `ChordLoss = 0.1206` | `ActivePMC = 31` | `Mitosis = 19` | `Closure = 0`
- `Step 05`: `ChordLoss = 0.2939` | `ActivePMC = 290` | `Mitosis = 57` | `Closure = 200` | 自发触发 `Auto-Expand (Base_v1_SubA_s3.pmc: 64 -> 128, woken_ratio=0.0515)` + `Auto-Closure (64 pages)`
- `Step 10`: `ChordLoss = 0.2663` | `ActivePMC = 402` | `Mitosis = 75` | `Closure = 328` | 自发触发 `Auto-Split (Base_v1_SubA_s3.pmc -> SubA + SubB + BridgeAB, 128 -> 64, 64)` + `Auto-Closure (128 pages)`
- `Step 15`: `ChordLoss = 0.3057` | `ActivePMC = 475` | `Mitosis = 77` | `Closure = 392` | 自发触发 `Auto-Merge (SubA + SubB -> TriadicMerge_1_s15.pmc, coactivation_ratio=1.0, new_grid_k=128)` + `Auto-Closure (64 pages)`

