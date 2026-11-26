# Penrose v1.0 — 连续黎曼流形计算架构白皮书 (`MPU` + `PMC` + `Penrose Harness`)

> **从离散静态词表（Tokenizer + Embedding + LM Head）向连续单位超球面状态推进器（$\mathbb{S}^{2047}$ Manifold Processing Unit）的代际跃迁**
>
> - **代码许可协议（Code License）**：`MPU`（`penrose_mpu.py`）、`PMC` 引擎（`penrose_pmc.py`）、`Penrose Harness`（`penrose_harness.py`）等核心机制代码遵循 **[GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)** 协议。
> - **流形记忆块许可协议（Manifold Chunk License）**：所有训练、蒸馏与结晶生成的 **`.pmc`（Penrose Manifold Chunk）文件**遵循 **[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)** 协议。
> - *历史初代验证架构 `Bicamera (v1 ~ v42)` 核心算法机制与学术账本已归档至 `workspace/old_bicamera/`。*


---

## 一、 架构总览与七步第一性原理（为什么 Penrose 能够存在）

**`Penrose v1.0`** 彻底终结了传统大语言模型将计算内核死绑在几万行静态词表（`nn.Embedding` 与 `LM Head Softmax`）以及海量优化器状态（`AdamW` $m_t, v_t$）上的旧范式。其全部物理与数学合法性建立在以下**七步第一性原理推导链**之上：

1. **第一性原理 1（内禀流形公理）**：任何自洽的信息系统（自然语言、形式数学、计算机代码、物理定律）都不是杂乱无章的离散词表，而是受局部语法与全局逻辑共同约束的**低维内禀微分流形 $\mathcal{M}^k$**（内禀维数 $k \ll 2048$）。
2. **第一性原理 2（母体超球面等距嵌入公理）**：通过零参数的 **$2048 = 8 \times 256$ 沃尔什-阿达玛（Walsh-Hadamard）正交全息换能器 $\mathbf{H}_8 \otimes \mathbf{H}_{256}$** 与纳什高余维等距嵌入定理（Nash Isometric Embedding），任意领域的内禀流形 $\mathcal{M}^k$ 皆被无损嵌入到**全宇宙唯一共享的 2048 维母体单位超球面 $\mathbb{S}^{2047} \subset \mathbb{R}^{2048}$** 之中：
   $$\mathcal{M}^k \xhookrightarrow{\quad \iota \quad} \mathbb{S}^{2047}, \qquad T_q\mathbb{S}^{2047} = \underbrace{T_q\mathcal{M}^k}_{\text{内禀切空间 }(k\text{ 维})} \;\oplus\; \underbrace{N_q\mathcal{M}^k}_{\text{法向余维体空间 }(2047-k\text{ 维})}$$
3. **第一性原理 3（嘉当活动标架与空间曲率自适应粒度）**：每一个 4KB 微专家页 $(u_m, v_m) \in \mathbb{R}^{2 \times 2048}$ 是钉在母体超球面 $\mathbb{S}^{2047}$ 上的**嘉当活动标架（Cartan Moving Frame）**——$u_m$ 指向流形特征点并编码 1-Jet 入射朝向，尽头的 $v_m$ 同时提供法向负散度汇点引力（将偏离轨迹拉回 $\mathcal{M}^k$）与切平面推力（驱动状态沿流形前进）。
   - **为什么微专家无需存储额外的“粒度标量”**：所有微专家保持 **100% 物理同质**；流形的局部粒度完全由微专家在 $\mathbb{S}^{2047}$ 上的**空间排布密度与拓扑分裂**自然涌现——平滑低曲率区微专家稀疏分布，靠公理 6 的球面 $\ell_2$ 单位分解（$\sum_{m \in \text{Simplex}(q)} w_m^2(q) \equiv 1$）与三次厄米特 $C^1$ 插值即可精确还原流形表面；陡峭高曲率或逻辑分岔区则自动高频触发 **`Auto-Mitosis`（切向叶状分支）** 与 **`Auto-Expand`（四叉树加密）**，用高密度标架群精细刻画复杂拓扑。
4. **第一性原理 4（`.pmc` 作为局部坐标卡 Local Coordinate Chart）**：每一个 **`.pmc`（Penrose Manifold Chunk，流形记忆块）** 并非各自孤立的“小宇宙”，而是贴合在唯一母体超球面 $\mathbb{S}^{2047}$ 特定区域 $U_\alpha \subset \mathbb{S}^{2047}$ 上的**自适应分辨率局部坐标卡 $(U_\alpha, \varphi_\alpha)$**（内部维护 $K_\alpha \times K_\alpha$ 双正交乘积子键网格）。
5. **第一性原理 5（曲率奇点裂变与测地线联络桥 `BridgeAB.pmc`）**：当单个 `.pmc` 坐标卡覆盖的区域跨越两个差异显著的知识板块、导致曲率张量饱和时，**`Auto-Split`** 自动将其裂变为两张紧贴各自分流形的子坐标卡 **`SubA.pmc`** 与 **`SubB.pmc`**，并将跨板块跃迁的非对角微专家单独抽取为**测地线联络桥（Transition Bundle `BridgeAB.pmc`）**——其锚点 $u_{\text{bridge}} \in \partial U_A$ 钉在板块 A 出口，切向推力 $v_{\text{bridge},\perp}$ 精确指向板块 B 入口 $U_B$。
6. **第一性原理 6（米尔诺 1-柄拓扑手术与霍奇正交三分解定理）**：当多步思考链（CoT）经真值校验通过并回流起点结晶出直达虫洞微专家（或生成跨板块 `BridgeAB.pmc`）时，其推力借道母体超球面的法向余维体空间 $N_q\mathcal{M}^k$，对知识子流形执行**米尔诺 1-柄粘合拓扑手术（Milnor 1-Handle Attachment）**，使子流形的第一贝蒂数（一维上同调维数）严格加 1：$b_1(\mathcal{M}_{\text{after}}) = b_1(\mathcal{M}_{\text{before}}) + 1$。由此，知识子流形上的向量场严格满足完备的**霍奇分解定理（Hodge Decomposition Theorem）**：
   $$\mathbf{V}\big|_{\mathcal{M}} = \underbrace{\mathbf{V}_{\text{div}}}_{\substack{\text{数据微专家 }(\operatorname{div}<0) \\ \text{吸引盆汇点 / 事实锚点}}} \;\oplus\; \underbrace{\mathbf{V}_{\text{curl}}}_{\substack{\text{指令微专家 }(\operatorname{curl}\neq 0) \\ \text{局部盘山公路 / 步进推演流}}} \;\oplus\; \underbrace{\mathcal{H}^1(\mathcal{M})}_{\substack{\text{拓扑调和虫洞场 / Bridge.pmc} \\ \text{跨区域直达捷径 }\big(\dim\mathcal{H}^1(\mathcal{M}) = b_1(\mathcal{M})\big)}}$$
7. **第一性原理 7（`MPU` 作为流形上的测地线微分方程积分器）**：既然知识与指令皆已编码为 `.pmc` 图册在 $\mathbb{S}^{2047}$ 上张成的三维正交向量场（$\mathbf{V}_{\text{div}} \oplus \mathbf{V}_{\text{curl}} \oplus \mathcal{H}^1(\mathcal{M})$），计算内核 **`MPU`（Manifold Processing Unit）** 就不再需要记忆静态事实，而纯粹退化为母体超球面上的**五段式高阶常微分方程（ODE）数值积分器**，驱动状态点 $q(\tau)$ 沿流形测地线精准滑行至 $q(\tau+1) \in \mathbb{S}^{2047}$。

---

### 三大核心部件总览

| 核心组件 | 全称与代码模块 | 现代计算机对标 | 核心数学与物理职责 |
| :--- | :--- | :--- | :--- |
| **`MPU`** | **Manifold Processing Unit**<br>(`penrose_mpu.py`) | **CPU / 测地线微分方程积分器** | **0 词表参数**；严格实现「2048 维单位球面向量进 $\to$ 2048 维单位球面向量出」的五段式 12 层（8 注意力头）状态推进器：$q(\tau) \to q(\tau+1) \in \mathbb{S}^{2047}$。 |
| **`PMC`** | **Penrose Manifold Chunk**<br>(`penrose_pmc.py`, `.pmc`) | **4KB 局部坐标卡图册 / 固态流形盘** | **100% 物理同质嘉当活动标架** $(u_m, v_m) \in \mathbb{R}^{2 \times 2048}$；在简单流形区稀疏分布并由球面 $\ell_2$ 单位分解平滑插值，在复杂高曲率区通过 `Auto-Mitosis`（叶状分支）与 `Auto-Split`（分流形 `SubA/SubB` + 联络桥 `BridgeAB`）自适应加密；采用 **0 字节优化器显存**的 Kaczmarz 正交投影与罗德里格斯大圆旋转一步锁死流形。 |
| **`Penrose Harness`** | **Manifold Bus & OS Scheduler**<br>(`penrose_harness.py`) | **MMU + DMA 预取器 + 外围 I/O 换能总线** | 外置 **$2048 = 8 \times 256$ 沃尔什-阿达玛（Walsh-Hadamard）正交谐波全息波包换能器**（0 参数）与**双层负散度漏斗流式切分器**；管控 Module 5 出口的**硬件级三态总线分支（分支 A/B/C）**、早一轮白盒流线预取与自治森林六算子演化（`Auto-Mitosis / Closure / Expand / Split / Merge / Spawn`）。 |

---

## 二、 `MPU` 五段式纯球面流水线（Module 1 $\sim$ Module 5）

```latex
\text{MPU 单拍流水线}: \quad 
\underbrace{\big(q(\tau), \, q(\tau-1)\big)}_{\text{输入：当前与上一位置}} 
\xrightarrow{\text{Module 1}} \psi^{(0)}(\tau) 
\xrightarrow{\text{Module 2 (下 6 层)}} \psi^{(6)}(\tau) 
\xrightarrow{\text{Module 3 (PMC 腰部)}} \tilde{\psi}^{(6)}(\tau) 
\xrightarrow{\text{Module 4 (上 6 层)}} \Delta q(\tau) 
\xrightarrow{\text{Module 5}} \underbrace{q(\tau+1) \in \mathbb{S}^{2047}}_{\text{输出：下一状态位置}}
```

1. **模块 1：入口——1-Jet 相空间合成器（`MPUModule1JetIngress`，0 可学习参数）**
   - 彻底取代传统 `nn.Embedding` 查表层；调用**公理 5（1-Jet 正交分解）**将当前球面位置 $q(\tau)$ 与切向入射速度 $\widehat{\Delta q}_\perp(\tau)$ 合成为第 0 层 2048 维单位向量：
     $$\widehat{\Delta q}_{\perp}(\tau) = \text{Normalize}\Big( q(\tau) - \langle q(\tau), q(\tau-1) \rangle q(\tau) \Big), \qquad \psi^{(0)}(\tau) = \sqrt{\frac{2}{3}} \, q(\tau) + \sqrt{\frac{1}{3}} \, \widehat{\Delta q}_{\perp}(\tau) \in \mathbb{S}^{2047}$$
   - **早一轮白盒流线预取触发点**：在 $\psi^{(0)}(\tau)$ 合成的第 0 微秒，旁路立即扫描 `PMC` 乘积键网格与静态流线指针表（BTB）发起异步 DMA 预取，彻底掩蔽外存延迟。
2. **模块 2：上半身——前 6 层“上下文割线积分栈”（Lower 6 Blocks，8 头混合注意力 + 全激活正交基 `OrthogonalMetaMoE`）**
   - 扫描历史球面轨迹 $[\psi^{(0)}(1), \dots, \psi^{(0)}(\tau)]$，提取多轮上下文向量割线 $\mathbf{P}(\Gamma)$，输出到达腰部的查询向量 $\psi^{(6)}(\tau)$。
3. **模块 3：腰部——`PMC` 流形场耦合门（`MPUModule3WaistCoupler`）**
   - 唤醒 `PMC` 四叉树中的局部三维单纯形微专家，合成流形场切向推力与旋度偏转向量 $\Delta_{\text{PMC}}(\tau) = \mathbf{V}_{\text{PMC}}(\psi^{(6)}(\tau))$，经单位残差门控注入主干：
     $$\tilde{\psi}^{(6)}(\tau) = \psi^{(6)}(\tau) \odot \Big( 1.0 + \tanh\!\big( W_{\delta} \, \Delta_{\text{PMC}}(\tau) \big) \Big)$$
4. **模块 4：下半身——后 6 层“轨迹曲率精炼栈与风格塑形器”（Upper 6 Blocks，8 头注意力 + 双线性正交元算子）**
   - 融合高阶非阿贝尔李括号耦合 $[\Omega_1, \Omega_2]$，输出切向位移推力 $\Delta q(\tau) \in \mathbb{R}^{2048}$。
5. **模块 5：出口——跨核球面 ResNet 闭环器（`MPUModule5GeodesicEgress`，0 可学习参数）**
   - 彻底砍掉传统 `LM Head` 与 `Softmax`；执行切空间正交化与跨核球面 ResNet 残差闭环：
     $$\Delta q_{\perp}(\tau) = \Delta q(\tau) - \big\langle \Delta q(\tau), \, q(\tau) \big\rangle \, q(\tau), \qquad q(\tau + 1) = \mathcal{P}_{\mathbb{S}^{2047}}\Big( q(\tau) + \Delta q_{\perp}(\tau) \Big) \in \mathbb{S}^{2047}$$

---

## 三、 外围 I/O 换能总线：$2048 = 8 \times 256$ 全息波包与双层负散度漏斗

编码（字节 $\to$ 波包）与解码（波包 $\to$ 字节）完全剥离在 `MPU` 外部的 `Penrose Harness` 总线换能器（`HolographicWavepacketTransducer`）上：

1. **微观波包合成与 Gumbel 极值自适应解码（0 可学习参数）**：
   - 利用 $d = 2048 = 8 \times 256$，单字节 $b \in \{0..255\}$ 映射为 $256 \times 256$ 阶沃尔什-阿达玛正交矩阵 $\mathbf{H}_{256}$ 的第 $b$ 行 $\mathbf{h}_{256}(b)$（任意两不同字节内积恒为 $\delta_{b_1, b_2} = 0$）；
   - 8 个正交谐波槽位配合非阿贝尔黄金角 Givens 相位旋转 $\mathbf{R}_{\text{pos}}^{(i)}$ 与全局 $\mathbf{H}_8 \otimes \mathbf{I}_{256}$ 混频，使每个字节的能量均匀铺满 8 个 256 维注意力头：
     $$\mathbf{u}^{(i)}\big(b^{(i)}\big) = (\mathbf{H}_8 \otimes \mathbf{I}_{256})\,\mathbf{R}_{\text{pos}}^{(i)}\Big(\mathbf{e}_{\text{slot}(i \bmod 8)} \otimes \mathbf{h}_{256}\big(b^{(i)}\big)\Big), \qquad q(\tau) = \frac{1}{\sqrt{k}}\sum_{i=1}^k \mathbf{u}^{(i)}\big(b^{(i)}\big) \in \mathbb{S}^{2047}$$
   - 解码时作用正交逆变换，利用**公理 2（冈贝尔极值孔径阈值 $\tau_z^*(256)$）**自动判定波包有效字节数 $k$，无需结束符即可 $100\%$ 无损还原 UTF-8 字节流。
2. **宏观双层负散度漏斗流式切分（免分词器、免正则清洗）**：
   - **第一层（零参数物理相变边界）**：跨 Unicode 物理相态（拉丁字母数字、CJK 汉字、语法标点、空白控制相）自动封口，天然将 `{text: "` 等 JSON 外壳隔离为独立前导波包；
   - **第二层（`PMC` 负散度漏斗动态捕获）**：实时检测连续字符合成波包是否掉入 `PMC` 的负散度吸引盆（$\operatorname{div}\mathbf{V} < 0$），实现从冷启动单字波包到复合词波包的自发凝聚演进。
3. **Module 5 出口的硬件级三态总线分支**：
   - **分支 A（内部旋度闭环 / 隐式思考）**：当 $\operatorname{div}\mathbf{V} \approx 0, \eta_{\text{curl}} \to 1$ 时，$q(\tau+1)$ 不经过解码器，直接经内部高速旁路跳回 Module 1 连续推进；
   - **分支 B（外部负散度坍缩 / 符号吐出）**：当 $\operatorname{div}\mathbf{V} < 0$（落入概念汇点）时，由外围总线执行逆阿达玛变换吐出 UTF-8 字节流；
   - **分支 C（球面 ResNet 并行训练态）**：训练期完全不执行解码，直接在 Module 5 出口计算球面弦长误差 $\mathcal{L}_{\text{Penrose}}(\tau) = 1 - \langle q(\tau+1), q^*(\tau+1)\rangle$。
4. **公理 9 & 10 闭式逆向流形擦除算子（`drag_and_drop_erase` / `geodesic_kaczmarz_erase_downdate`）**：
   - 正向拖拽学习（`drag_and_drop_ingest`）与逆向拖拽遗忘（`drag_and_drop_erase`）构成流形记忆的正逆对偶群算子：
     - **独占微专家页（公理 9 细胞凋亡）**：当目标轨迹对应的微专家页残余有效样本计数 $h_{\text{rem}} = h_{\text{old}} - \eta_{\text{erase}}\Delta h \le 0$ 时，执行 `apoptosis_expert_pages` 将其 INT8 载荷、FP16 缩放与流线边 100% 物理归零，并将同行/同列子键码本重锚定至剩余存活微专家的页内键重心；
     - **共享微专家页（公理 10 秩-1 正交剥离）**：当该微专家页同时承载其他文档的公共知识（$h_{\text{rem}} > 0$，如共享的语法或前缀道岔）时，按逆向样本权重 $\beta = \min(\eta_{\text{erase}}\Delta h / h_{\text{old}}, 1 - \frac{1}{\sqrt{d_h}})$ 从载荷 $v_{\text{old}}$ 中剥离平行于待擦除轨迹的分量，并按反向角 $-\beta\theta$ 执行逆向 Rodrigues 测地旋转与流线熔断（`sever_streamline_transition`），在彻底抹除目标轨迹的同时对共存文档记忆实现零连带损伤。

---

## 四、 快速验证与训练指南

```bash
# 1. 运行 Penrose v1.0 六大核心架构契约与流形定理验证套件
python test_remote.py

# 2. 运行免清洗全语料流形拖拽训练与 Kaczmarz 测地线吸附主程序
python train_penrose_v1.py --steps 20
```

---

## 五、 开源许可协议（License & Dual-Licensing Policy）

本项目采用**计算机制代码**与**流形知识构件（`.pmc`）**分离的双轨开源许可协议：

1. **核心机制代码（`GNU AGPL-3.0`）**：
   - 本仓库中的 **`MPU`（`penrose_mpu.py`）**、**`PMC` 流形管理引擎（`penrose_pmc.py`）**、**`Penrose Harness` 外围换能与调度总线（`penrose_harness.py`）**、**`bkp_manager.py`** 及相关训练/验证脚本均遵循 **[GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html)** 协议开源。
   - 任何基于本架构代码的修改、衍生开发或通过网络对外提供推理/计算服务的部署，均须遵循 AGPL-3.0 协议要求完整公开相应源代码。
2. **训练生成的 `.pmc` 流形记忆块文件（`CC-BY-SA 4.0`）**：
   - 所有通过本架构训练、拖拽蒸馏或自治演化结晶生成的 **`.pmc`（Penrose Manifold Chunk）流形文件**均遵循 **[Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0，知识共享 署名-相同方式共享 4.0 国际许可协议)](https://creativecommons.org/licenses/by-sa/4.0/)**。
   - 允许在保留署名（Attribution）的前提下自由共享、挂载、合并（`Auto-Merge`）、分裂（`Auto-Split`）或再演绎 `.pmc` 流形块，但任何基于该 `.pmc` 衍生、微调或混合重构出的新 `.pmc` 文件必须以相同的 **CC-BY-SA 4.0** 协议进行分发。

