# gp-context-lab

> 面向 [`gp`（Go Coding Agent Harness）](https://github.com/npc1607/gp) 的**上下文权重计算、隐性关联发现与缓存友好型工作集（Working Set）实验项目**。

---

## 为什么开这个独立实验项目？

在长周期 Coding / 运维排查会话中，每次 LLM 请求把全量历史原封不动发给模型，会同时带来 **Token 成本爆炸、首字延迟升高、上下文腐烂（Context Rot）** 三个问题。

[`gp`](https://github.com/npc1607/gp) 此前已经通过 [`Observation-Masking与DTOC上下文防污染方案.md`](https://github.com/npc1607/gp/blob/main/docs/Observation-Masking与DTOC上下文防污染方案.md) 解决了第一层“**工具长输出膨胀**”问题（最近 5 轮原文保留 + 早期大输出折叠 + `restore_observation` 按需还原）。但在进一步演化到**百轮级、超长程自主编码任务**时，仍然存在三个未被解决的核心矛盾：

1. **隐性多跳关联发现缺失**：只靠时间滑窗或纯向量相似度，无法解决“**A 和 B 表面毫无字面交集，但通过文件依赖、符号调用链或中间决策实际强相关**”的问题。
2. **自然语言摘要漂移与无溯源**：[`compactor.go`](https://github.com/npc1607/gp/blob/main/internal/core/compaction/compactor.go) 在到达 `0.80 * contextWindow` 后将早期对话压成 5 段自然语言文本，存在约束漂移、冲突无法仲裁、无法回查原始 `entry_id` 的隐患。
3. **Prompt Caching 前缀击穿陷阱**：[`applyObservationMasking`](https://github.com/npc1607/gp/blob/main/internal/agent/observation.go) 每新增一轮工具调用都会将 `cutoffIdx` 向前推进一轮，相当于**每轮都在改写前缀中的历史消息**；同时如果动态召回片段插在稳定前缀前面，会反复击穿 KV / Prefix Cache，导致“省了 token 反而账单更贵”。

本项目旨在**不直接改动 `gp` 主干稳定性**的前提下，先独立完成：
- **数学工具选型与公式收敛**（纠偏常见直觉误区，如裸 Surprisal 陷阱、Hawkes 全局衰减率歧义、卡尔曼滤波错误类比）；
- **可量化对比实验设计与照妖镜基准集构建**；
- **消融实验（Ablation Study）验证达标后**，再将验证通过的算法模块合入 `gp`。

---

## 项目目录结构

```text
gp-context-lab/
├── README.md                              # 项目定位、实验结论与文档导航
├── go.mod                                 # 零外部依赖纯 Go 实验模块
├── cmd/
│   ├── eval/                              # Phase 2 离线评测 Runner（Layer 1 检索 + LCP 缓存仿真 + 延迟）
│   └── e2e/                               # Phase 3 真实 LLM 端到端评测（Layer 2 硬断言 + Layer 3 盲评 + 统计检验）
├── internal/
│   ├── algo/                              # PPR、热核扩散、ACT-R 幂律、Budgeted MMR / k-DPP、RRF、密度惩罚
│   ├── cache/                             # Prompt Caching Token 级 LCP 仿真与双厂商计费模型
│   ├── dataset/                           # 基准数据集加载与 Schema 校验
│   ├── index/                             # 代码分词 BM25、本地 Hash N-Gram 向量、实体抽取、异构图
│   ├── judge/                             # Layer 2 正则门禁、Layer 3 引号接地盲评裁判、四象限、McNemar/Bootstrap
│   ├── ledger/                            # ContextLedger 结构化约束状态机与启发式/真值抽取器
│   ├── llm/                               # OpenAI / Anthropic / DeepSeek / Qcoder 客户端与本地磁盘缓存
│   ├── metrics/                           # Layer 1 召回/桥接/污染/冗余与耗时分位数统计
│   ├── report/                            # 汇总报表与 Go/No-Go 准出判定
│   ├── strategy/                          # B0~B2 基线与 E1~E5 消融策略实现
│   └── tokenizer/                         # 确定性子词切分与 Token 计数
├── docs/
│   ├── 01-问题诊断与总体目标.md             # 基于 gp 源码的现状诊断、争议修正与 4 项量化硬目标
│   ├── 02-数学工具清单与选型论证.md         # 7 类核心数学工具的公式、复杂度、陷阱与在 gp 中的映射
│   ├── 03-结构化工作集与防丢信息架构.md     # 不可变仓库 + 结构化约束账本 + 溯源回查协议设计
│   ├── 04-实验设计与评测基准.md             # 4 类测试套件、8 组对照/消融实验、计费模型与验收门槛
│   ├── 05-打分与评判机制设计.md             # 三层评分架构（检索层 / 硬断言 / Atomic Rubric 裁判 + 归因四象限）
│   ├── 06-Go离线评测Runner架构设计.md        # Phase 2 纯 Go 离线评测 Runner 架构、接口与缓存仿真设计
│   └── 07-实验结果与准出评审报告.md         # Phase 2/3 全量实测数据、显著性检验与 Go/No-Go 准出结论
├── learn/                                 # 面向后台开发的前置知识学习笔记（01~04 课）
└── datasets/                              # 23 个正式基准用例（Suite A/B/C/D）+ 示例 + Schema
```

---

## 核心实验结论（Go / No-Go：**PASS**）

在全部 **23 个正式基准用例**（Suite A/B/C 各 6 个、Suite D 长会话回放 5 个，共 24 个探针）上，严格使用 `--extractor heuristic`（零标注泄露）完成了 Phase 2 离线仿真与 Phase 3 真实大模型（`deepseek-chat`）端到端盲评，详表见 [**07 - 实验结果与准出评审报告**](docs/07-实验结果与准出评审报告.md)。

### 1. 关键指标对比（`E4-A` vs 基线）

| 策略 | 离线多跳召回 (Suite A) | 过时约束污染 (Suite B) | Suite D 缓存命中 | Suite D 账单降幅 (vs B0) | 端到端通过率 (`deepseek-chat`) | 输入总 Token | 构建耗时 p95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0`（全量发送） | 100.0% | 85.7% | 96.8% | 0.0% | 91.7% (22/24) | 345,118 | 0.90 ms |
| `B1`（`gp` 现状） | 33.3% | 85.7% | 36.4% | 10.9% | 29.2% (7/24) | 18,930 | 0.22 ms |
| `B2`（朴素 RAG） | 22.2% | 71.4% | — | 6.8% | 66.7% (16/24) | 32,435 | 0.14 ms |
| **`E4-A`（推荐合入）** | **100.0%** | **0.0%** | **78.9%** | **52.3%** | **100.0% (24/24)** | **55,795** | **0.41 ms** |
| `E4-B` | 100.0% | 0.0% | 71.9% | 41.2% | 95.8% (23/24) | 55,794 | 0.39 ms |

### 2. 三条关键决策结论

1. **所有 4 项 Go / No-Go 门槛全部 `[PASS]`，建议合入 [`gp`](https://github.com/npc1607/gp)**：
   - **隐性多跳召回**：`E4-A` 在 Suite A 达到 **100.0%**（高于朴素 RAG `B2` **+77.8pp**）；
   - **约束零污染**：借助结构化 `ContextLedger` + `active`/`superseded` 状态机，Suite B 过时约束误触发率降至 **0.0%**；
   - **含缓存账单与速度**：在 `Epoch = 16` 冻结下，Suite D 缓存命中率达到 **78.9%**（`B1` 的 2.1 倍），含缓存总账单较 `B0` 降低 **52.3%**，纯本地构建延迟 **p95 = 0.41 ms**（远低于 `30 ms` 上限）。
2. **`E4-A` 在缓存效率与端到端准确率上双赢 `E4-B`**：
   - 当 `WorkingSet` 在同一个 Epoch 内冻结复用时，把高频近期窗口放在末尾追加（`E4-A`：`[System + Ledger] → [WorkingSet] → [EpochRecentWindow] → [Query]`）能避免每轮新消息推高后置 `WorkingSet` 偏移而击穿缓存；对称位置交换盲评中 `E4-A` 对 `E4-B` 为 `2 : 0`（平局 22）。
3. **上下文裁剪不仅省钱，还能反超全量发送（`100.0%` vs `91.7%`）**：
   - 在百轮级长会话上，全量发送 34.5 万 tokens 的 `B0` 会因上下文腐烂（Context Rot）和历史作废方案干扰在 `suite_d_03` / `suite_d_05` 上失分；`E4-A` 仅消耗 `B0` **16.2%** 的输入 Token 即实现 **24/24 (100.0%)** 全通过（对比 `B1` 的 McNemar $p = 0.0000$，Bootstrap $\Delta\text{Pass} = +71.1\%$；对比 `B2` 的 McNemar $p = 0.0078$）。

---

## 快速复现运行

```bash
# 1. 运行全部单元测试（算法、缓存 LCP 仿真、Layer 2/3 评判与统计检验）
go test -v ./...

# 2. 运行 Phase 2 纯 Go 离线评测（23 个基准用例，毫秒级完成）
go run ./cmd/eval --datasets ./datasets --strategies all --extractor heuristic --epoch 16

# 3. 运行 Phase 3 端到端评测（支持 mock 零成本演练，或接入 deepseek/openai/anthropic/qcoder）
go run ./cmd/e2e --provider mock --strategies B0,B1,B2,E4-A,E4-B
```

---

## 文档阅读顺序

1. [**01 - 问题诊断与总体目标**](docs/01-问题诊断与总体目标.md)：对齐“我们要解决什么问题”“吸收了哪些理论纠偏”“实验成功的 4 条量化标准是什么”。
2. [**02 - 数学工具清单与选型论证**](docs/02-数学工具清单与选型论证.md)：逐项列出 PPR 图扩散、热核扩散、MMR / k-DPP 去冗余、ACT-R 幂律遗忘、任务条件信息增益、缓存感知背包优化等工具的精确公式与目的。
3. [**03 - 结构化工作集与防丢信息架构**](docs/03-结构化工作集与防丢信息架构.md)：说明如何用 `结构化 JSON 约束账本 + entry_id 溯源指针 + 外置回查工具` 替代脆弱的自然语言摘要，并与 `gp` 的 `SessionTree` / `SQLite` 对接。
4. [**04 - 实验设计与评测基准**](docs/04-实验设计与评测基准.md)：给出可直接落地的对照实验矩阵（含 Prompt Caching 拼接顺序 A/B 实验与负对照验证）。
5. [**05 - 打分与评判机制设计**](docs/05-打分与评判机制设计.md)：定义“确定性检索评分 + 回答硬断言一票否决 + Atomic Rubric 盲评裁判 + 归因四象限”的完整评判标准，解决 LLM-as-a-Judge 漂移与阶段混淆问题。
6. [**06 - Phase 2 Go 离线评测 Runner 架构设计**](docs/06-Go离线评测Runner架构设计.md)：定义纯 Go 离线评测器的包结构、`ContextStrategy` 接口、逐轮 LCP 缓存仿真、双模式抽取防泄露及性能优化设计。
7. [**07 - 实验结果与准出评审报告**](docs/07-实验结果与准出评审报告.md)：汇总 23 个基准用例在离线与 DeepSeek-Chat 端到端下的完整实测结果、McNemar/Bootstrap 检验与 `E4-A` 准出建议。

---

## 落地节奏

- **Step 1（已完成）**：锁定数学模型、测试集 Schema 与评测指标文档（`docs/01`–`05`）。
- **Step 2（已完成）**：构建 23 个高信噪比基准用例（Suite A/B/C/D）与 `case.schema.json`。
- **Step 3（已完成）**：实现纯 Go 离线评测 Runner（`cmd/eval`）与端到端 LLM 盲评流水线（`cmd/e2e`），完成全部消融实验与准出评审（`docs/07`）。
- **Step 4（下一阶段）**：将已通过全部准出门槛的 **`E4-A` 核心组件**（`ContextLedger` + `PPR` + `ACT-R` + `Budgeted MMR` + `Epoch=16` 冻结掩码）迁移合入 [`gp`](https://github.com/npc1607/gp) 的 `internal/core/compaction` 与 `internal/agent`。
