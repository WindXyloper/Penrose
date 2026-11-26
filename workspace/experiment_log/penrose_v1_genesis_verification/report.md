# Penrose v1.0 创世架构契约与六大流形定理实测报告 (`penrose_v1_genesis_verification`)

- **测试脚本**: `F:\Penrose\test_remote.py` & `F:\Penrose\train_penrose_v1.py`
- **核心模块**: `penrose_pmc.py` (`PMC`), `penrose_mpu.py` (`MPU`), `penrose_harness.py` (`Penrose Harness`)
- **测试设备**: `cuda`
- **原始 JSON 报告**: `workspace/experiment_log/penrose_v1_genesis_verification/report.json`

---

## 一、 六大核心契约实测结果汇总

### 1. $2048 = 8 \times 256$ 沃尔什-阿达玛全息波包换能器 (`exp1_holographic_wavepacket_8x256`)
- **可学习参数量 (`learnable_params`)**: **`0`**（彻底废除 `Tokenizer`、`nn.Embedding` 与 `LM Head`）
- **单字节正交内积 (`inter_byte_dot`)**: **`0.00000000`**
- **跨槽位正交内积 (`inter_slot_dot`)**: **`0.00000000`**
- **8 注意力头能量均匀度误差 (`head_energy_uniformity_err`)**: **`1.0e-8`**（每头能量恒为 $1/8 = 0.125$）
- **公理 2 Gumbel 孔径自适应长度无损还原率 (`roundtrip_accuracy`)**: **`9/9 (100.0%)`**
- **非阿贝尔顺序区分度 (`ab_vs_ba_cosine`)**: **`0.000000`**

### 2. 双层物理相变 + `PMC` 负散度漏斗流式切分 (`exp2_two_tier_negative_divergence_funnel`)
- **输入测试脏语料**: `'{text: "这是一段训练语料。"}'`
- **第一层物理相变门隔离前导格式壳 (`shell_prefix_isolated`)**: `['{', 'text', ': "']`（100% 零正文混入）
- **第 0 步冷启动空白流形切分 (`cold_start_chunks`, 14 波包)**:
  `['{', 'text', ': "', '这', '是', '一', '段', '训', '练', '语', '料', '。', '"', '}']`
- **凝聚负散度数据微专家后漏斗捕获切分 (`warm_funnel_chunks`, 12 波包)**:
  `['{', 'text', ': "', '这', '是', '一', '段', '训练', '语料', '。', '"', '}']`（自动将 `"训练"` 与 `"语料"` 吸入复合词波包漏斗）

### 3. `MPU` 五段式纯球面流水线与三态总线分支 (`exp3_mpu_five_module_and_tri_state_bus`)
- **静态词表大小 (`vocab_size`)**: **`0`**
- **流水线拓扑**: Module 1 (1-Jet Ingress) $\to$ Module 2 (前 6 层 8 头割线积分栈) $\to$ Module 3 (腰部 `PMC` 单位残差门) $\to$ Module 4 (后 6 层 8 头曲率精炼栈) $\to$ Module 5 (跨核球面 ResNet 切空间正交闭环)
- **Module 1 $\psi^{(0)}(\tau)$ 单位球面范数误差**: **`6.0e-8`**（并在第 0 微秒自动触发异步预取 **`23`** 个微专家页）
- **Module 5 切空间正交化误差 $\langle \Delta q_\perp(\tau), q(\tau)\rangle$**: **`2.3e-7`**
- **Module 5 输出 $q(\tau+1)$ 单位球面范数误差**: **`6.0e-8`**
- **三态总线分支轨迹 (`tri_state_branch_trace`)**: `["branch_a_curl_loop", "branch_a_curl_loop", "branch_b_sink_emit"]`

### 4. 无 `AdamW` 的 Kaczmarz + 罗德里格斯大圆旋转一步锁死 (`exp4_adamw_free_kaczmarz_and_dirty_shell`)
- **`PMC` 优化器动量显存占用 (`pmc_optimizer_memory_bytes`)**: **`0 Bytes`**
- **脏文档 1 单步 Kaczmarz 弦长误差下降**: **`1.0022 -> 0.5072`（单步直降 `-49.4%`）**
- **脏文档 2 单步 Kaczmarz 弦长误差下降**: **`0.7922 -> 0.4035`（单步直降 `-49.1%`）**
- **`{text: "` 道岔自动触发 `Auto-Mitosis` 切向有丝分裂数**: **`9` 页**
- **干净提问时 JSON 格式符垃圾泄露 (`clean_query_json_leakage`)**: **`false`（0% 泄露）**

### 5. 流形穿孔邻域三次厄米特 $C^1$ 单纯形插值自愈 (`exp5_manifold_puncture_c1_simplex_healing`)
- **硬性拔除中心微专家 (`punctured_center_eid`)**: `0`
- **单纯形邻域伴随微专家 (`healed_top_neighbors`)**: `[1, 2, 3]`
- **$C^1$ 单纯形邻域插值自愈余弦 (`c1_simplex_healing_cosine`)**: **`0.999977`**
- **自愈场与真实目标场余弦 (`healed_to_true_field_cosine`)**: **`0.999679`**

### 6. 六大森林自治算子、非对称谱双聚类、多坐标卡图册与霍奇三分解「拓扑虫洞道岔」 (`exp6_six_autonomous_forest_ops_and_wormhole`)
- **所有衍生块类型均为 `PMCPackage` (`all_packages_are_pmc`)**: **`true`**
- **四叉树扩容零步误差 (`expand_zero_step_err`)**: **`0.00000000`**
- **流线边自动重映射索引 (`stream_ptr_remapped_eid`)**: **`129`** (`(0,0) -> (1,1)` 在 `grid_k=128` 下精确映射为 $1 \times 128 + 1 = 129$)
- **非对称谱双聚类切分规格 (`asymmetric_split_grid_ks`)**: **`[64, 128]`**（在剥离未激活空白死区后的活跃子图上执行最小归一化电导割扫描，将 $20 \times 20$ 与 $88 \times 88$ 双簇精确向上补齐为 $64 \times 64$ 与 $128 \times 128$，摒弃硬编码 `k // 2` 中点对半切分）
- **多坐标卡图册 (`SubA.pmc` + `SubB.pmc` + `BridgeAB.pmc`) 霍奇调和流占比 (`eta_harmonic_max`)**: **`1.0`（实测验证 $\mathcal{H}^1(\mathcal{M})$ 非平凡调和虫洞分量激活，且 100% 复用 `BKPManager.query_and_inject` 三级测地线球路由实现零重复扫描）**
- **母体超球面共振归属路由 (`select_resonant_writable_package`)**: 自动路由至共振度最高的可写局部坐标卡 `ForestSeed_SubB_s2.pmc`
- **森林算子触发计数 (`operator_totals`)**: `split_count=1, expand_count=1, merge_count=1, spawn_count=1, closure_count=1`
- **验证思考回流触发 `Auto-Mitosis` 虫洞道岔分裂数 (`wormhole_mitosis_switches`)**: **`1`**（在同一子键行开辟伴随列构建直达虫洞，100% 保留原多步思考盘山公路）

### 7. 15 步端到端免清洗拖拽流形训练 (`end_to_end_training_15steps`)
- `Step 01`: `ChordLoss = 0.1615` | `ActivePMC = 35` | `Mitosis = 5` | `Closure = 0`
- `Step 05`: `ChordLoss = 0.1871` | `ActivePMC = 4269` | `Mitosis = 27` | `Closure = 64` | 触发 `Auto-Spawn (Spec_general_1_s5.pmc)` + `Auto-Closure`
- `Step 10`: `ChordLoss = 0.3170` | `ActivePMC = 3916` | `Mitosis = 78` | `Closure = 192` | 触发 `Auto-Split (128 -> 64, 64)` + `Auto-Closure`
- `Step 15`: `ChordLoss = 0.3510` | `ActivePMC = 3981` | `Mitosis = 126` | `Closure = 256` | 触发 `Auto-Merge (TriadicMerge_1_s15.pmc, coact=1.0)` + `Auto-Closure`

