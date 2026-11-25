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
│   ├── 07-实验结果与准出评审报告.md         # Phase 2/3 实测数据、敏感性/校准验证与审慎准出结论
├── learn/                                 # 面向后台开发的前置知识学习笔记（01~04 课）
└── datasets/                              # 28 个正式基准用例（Suite A/B/C/D，含隐晦对抗集）+ 示例 + Schema
```

---

## 核心实验结论（方向可信、幅度与抽取边界需审慎限定）

在 **28 个合成基准用例**（Suite A 6 个、Suite B 11 个[含 5 个无显式关键词对抗用例]、Suite C 6 个、Suite D 长会话回放 5 个）上完成了 Phase 2 离线仿真、敏感性/校准校验与 Phase 3 `deepseek-chat` 端到端盲评，详表与限制说明见 [**07 - 实验结果与准出评审报告**](docs/07-实验结果与准出评审报告.md)。

### 1. 关键指标对比（诚实 `B1` 基线口径，28 用例 `--extractor heuristic --epoch 16`）

| 策略 | Suite A 多跳召回 | Suite B 过时污染 (含隐晦集) | Suite D 缓存命中 | Suite D 账单降幅 (vs B0) | p95 延迟 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `B0`（全量发送） | 100.0% | 91.7% | 96.8% | 0.0% | 0.92 ms |
| `B1`（诚实版 `gp` 现状） | 83.3%（Bridge 50.0%） | 91.7% | 40.1% | -5.4% | 0.37 ms |
| `B2`（朴素 RAG） | 22.2% | 83.3% | — | 6.8% | 0.10 ms |
| **`E4-A`（推荐结构）** | **100.0%** | **41.7%**（`oracle` 事件下 **0.0%**） | **78.9%** | **52.3%** | **0.41 ms** |
| `E4-B` | 100.0% | 41.7% | 71.9% | 41.2% | 0.38 ms |

### 2. 四条审慎结论与边界声明

1. **`E4-A` 的缓存布局与 Epoch 机制真实有效**：
   - 当 `WorkingSet` 在 Epoch 内冻结复用时，`E4-A`（`[System + Ledger] → [WorkingSet] → [EpochRecentWindow] → [Query]`）将高频窗口置于尾部追加，在 `Epoch = 8/12/16/24/32` 敏感性曲线上**全档位击败 `E4-B`**，并在 `Epoch = 12–16` 形成帕累托平台（Suite D 缓存命中 `75.0%–78.9%` vs 诚实 `B1` `40.1%`，成本降幅 `52.2%–52.3%`）。
2. **启发式关键词抽取不可用于生产（`0%` 污染率需降级）**：
   - 加入 5 个无显式触发词的口语化变更用例（`suite_b_07`–`11`）后，`HeuristicExtractor` 的过时污染率从 `0.0%` 跌至 **`41.7%`**（而输入准确事件时状态机仍为 `0.0%`）。合入 `gp` 时，`ContextLedger` 必须由 **结构化 LLM 提取或工具调用** 维护，不能靠正则关键词。
3. **诚实基线与 Tokenizer 偏差校准**：
   - 将 `B1` 从早期的 48 字符截断改为保留完整句段的诚实基线后，`B1` 离线 Suite A 召回从 `33.3%` 升至 `83.3%`，但仍因滑窗改写前缀与无状态仲裁在缓存成本（`-5.4%`）和过时污染（`91.7%`）上显著落后；此前用旧版 `B1` 跑出的 E2E `29.2%` 通过率存在基线高估，报告已明确标注限制。
   - 自制 Tokenizer 对照 120 次真实 `deepseek-chat` 输入的总体比值为 `0.973x`（中位误差 `9.2%`），但短中文句段 p95 误差达 `111.3%`，离线成本降幅属于近似参考值。
4. **合入 `gp` 的稳健路径**：
   - 先提最小高确定性 PR（`ContextLedger` 状态机 + `Epoch=12~16` 冻结掩码 + `E4-A` 拼接顺序 + 只读回查工具），在真实 `gp.db` 长会话回放集上用真实 LLM 摘要基线验收后，再决定是否合入 `PPR + ACT-R + MMR`。

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
7. [**07 - 实验结果与准出评审报告**](docs/07-实验结果与准出评审报告.md)：汇总 28 个基准用例在诚实 `B1`、`Epoch` 敏感性曲线、Tokenizer 校准、隐晦对抗集及 DeepSeek-Chat 端到端下的完整实测与审慎准出建议。
