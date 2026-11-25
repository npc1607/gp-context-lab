# 第 3 课：去冗余（MMR / k-DPP）、记忆衰减（ACT-R）与“裸 Surprisal 陷阱”

> **前置背景**：面向后台开发的 `gp-context-lab` 学习笔记（第 3 课）。  
> **关联项目文档**：
> - [01 - 问题诊断与总体目标](../docs/01-问题诊断与总体目标.md)
> - [02 - 数学工具清单与选型论证](../docs/02-数学工具清单与选型论证.md)

---

## 1. 为什么拿到 $W(d_i)$ 后不能直接 Top-K？

在上下文预算有限（例如只给召回留 $B_{\text{recall}} = 4000$ tokens）时，直接 `ORDER BY score DESC LIMIT K` 有两个经典 Bug：

1. **复读机挤占预算**：  
   连续 6 轮都在 `grep` 同一个报错或修改同一个 typo，这 6 条得分全排在前 6，Top-K 会把它们全塞进去，把第 7 名（分数略低但包含“并发上限为 2”的关键环境约束）挤出预算。
2. **大块头吞噬名额**：  
   一条 3,000 token 的中等高分日志，直接吃掉 75% 的预算，把本来能放 10 条高密度决策的额度全占光了。

### 解法一：Budgeted MMR（成本感知最大边界相关性）

MMR 就像**组建全栈团队**：既要个人能力强，又要和已招进来的人技能不重复，还得看性价比（Token 长度）。每挑下一条时，选择满足 $c_i \le B_{\text{rem}}$ 且下式最大的条目：

$$
d^{\ast} = \arg\max_{d_i \in C \setminus S,\enspace c_i \le B_{\text{rem}}} \frac{\lambda \cdot \tilde W(d_i) - (1-\lambda)\thinspace\max_{d_j \in S} \text{Sim}(d_i, d_j)}{c_i^{\eta}}
$$

- **$\lambda \cdot \tilde W(d_i)$**：这条消息自身有多重要；
- **$-(1-\lambda)\max_{d_j \in S}\text{Sim}(d_i, d_j)$**：它和**已选集合 $S$ 中最像的那一条**有多重复（可用余弦相似度 + 实体 Jaccard 相似度）；
- **分母 $c_i^\eta$**（$\eta \in [0.5, 1.0]$）：$c_i$ 是 Token 长度。相当于算“**每花 1 个 Token 买到的净信息增益**”，天然偏好短小精悍的高密度决策。

### 解法二：Budgeted k-DPP（行列式点过程，作为理论上限对照）

MMR 只看“和最像的那一条”是否重复；如果多条信息两两都不算极像，但在特征空间里**共线冗余**（张成同一子空间），MMR 可能漏掉组合冗余。  
**k-DPP** 构造核矩阵 $L_{ij} = \tilde W(d_i)\langle \boldsymbol\phi_i, \boldsymbol\phi_j\rangle \tilde W(d_j)$，通过最大化 $\det(\mathbf L_S)$（几何上等于这批向量张成的**超平行多面体体积**）从全局子空间角度彻底去重，利用增量 Cholesky 分解可在 $< 2\text{ ms}$ 内求解。

---

## 2. 记忆衰减：为什么用 ACT-R 幂律，而不用指数衰减？

长对话里有两类截然不同的历史：
- **一次性试错噪音**（第 12 轮敲错了一个命令，之后再没提过）$\rightarrow$ 快速遗忘；
- **核心架构约定**（第 5 轮定下的协议，第 30、65、88 轮又被反复引用或关联修改）$\rightarrow$ 即使过百轮也保持高激活。

项目采用认知科学中经过验证的 **ACT-R 幂律激活方程**（取代易出现参数歧义的 Hawkes 全局 $\gamma$）：

$$
B_i(t_{\text{now}}) = \ln \left( \sum_{k=1}^{m} \left( \Delta t_k + \delta \right)^{-d} \right), \qquad W_{\text{time}}(i, t_{\text{now}}) = \sigma\left(\frac{B_i(t_{\text{now}}) - \tau_B}{s}\right)
$$

- $\Delta t_k = t_{\text{now}} - t_k$：距离第 $k$ 次创建/引用/调回过去的**逻辑轮次（Turn）**，避免受挂机/休息等物理墙上时间干扰；
- **重尾特性 + 间隔效应（Spacing Effect）**：单次出现会平滑衰减让出空间；只要在不同时间跨度被反复强化（$m$ 次累加），激活值就会长期稳居高位。

---

## 3. 避坑实录：为什么严禁使用“裸 Surprisal（$-\log P$）”？

早期直觉容易误以为“语言模型觉得越意外（信息熵 / 惊讶度 $-\log P$ 越高）的内容越重要”。本项目明确禁用裸 Surprisal（仅在 `E5` 作为反证负对照组）：

1. **高熵 $\neq$ 高价值**：UUID、十六进制内存地址、Go panic 堆栈、混淆 JS、长 URL 全是超高 $-\log P$，但绝大多数属于噪音，按熵加权会把垃圾顶到最前面。
2. **自恋偏差与额外延迟**：用下游同源模型打分会引入自恋偏差，且逐条算 logprob 代价高昂。
3. **正确替代**：用轻量级的 **压缩比（如 `zlib` 压缩率）或独特词占比** 构造信息密度惩罚 $\rho(d_i)$，专门用来**降权**重复堆栈和客套废话。

---

## 4. 课后思考题

1. 在 **Suite C（高熵噪音风暴）** 里，连续 8 轮贴了长 Go panic 堆栈和 UUID，为什么负对照组 `E5`（用了裸 Surprisal）会惨败，而 `E1`（Budgeted MMR）能保住第 22 轮那句极短的“测试环境并发上限固定为 2”？
2. 为什么 ACT-R 公式里的 $\Delta t_k$ 要用**逻辑轮次（Turn）**而不是服务器时间戳（`time.Now()`）？
