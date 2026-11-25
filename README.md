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
├── README.md                              # 项目定位、背景与文档导航
├── docs/
│   ├── 01-问题诊断与总体目标.md             # 基于 gp 源码的现状诊断、争议修正与 4 项量化硬目标
│   ├── 02-数学工具清单与选型论证.md         # 7 类核心数学工具的公式、复杂度、陷阱与在 gp 中的映射
│   ├── 03-结构化工作集与防丢信息架构.md     # 不可变仓库 + 结构化约束账本 + 溯源回查协议设计
│   └── 04-实验设计与评测基准.md             # 4 类测试套件、8 组对照/消融实验、计费模型与验收门槛
└── datasets/
    ├── case.schema.json                   # 评测用例 JSON Schema 规范
    └── sample-multihop-case.json          # 多跳隐性关联 + 约束演进示例用例
```

---

## 文档阅读顺序

1. [**01 - 问题诊断与总体目标**](docs/01-问题诊断与总体目标.md)：对齐“我们要解决什么问题”“吸收了哪些理论纠偏”“实验成功的 4 条量化标准是什么”。
2. [**02 - 数学工具清单与选型论证**](docs/02-数学工具清单与选型论证.md)：逐项列出 PPR 图扩散、热核扩散、MMR / k-DPP 去冗余、ACT-R 幂律遗忘、任务条件信息增益、缓存感知背包优化等工具的精确公式与目的。
3. [**03 - 结构化工作集与防丢信息架构**](docs/03-结构化工作集与防丢信息架构.md)：说明如何用 `结构化 JSON 约束账本 + entry_id 溯源指针 + 外置回查工具` 替代脆弱的自然语言摘要，并与 `gp` 的 `SessionTree` / `SQLite` 对接。
4. [**04 - 实验设计与评测基准**](docs/04-实验设计与评测基准.md)：给出可直接落地的对照实验矩阵（含 Prompt Caching 拼接顺序 A/B 实验与负对照验证）。

---

## 下一步落地节奏

- **Step 1（当前阶段）**：锁定数学模型、测试集 Schema 与评测指标文档。
- **Step 2**：基于 `datasets/case.schema.json` 构建 20–30 个高信噪比“照妖镜”用例 + 脱敏真实 `gp` 长会话回放集（导出自 `gp.db`）。
- **Step 3**：用 Go 编写轻量离线评测 Runner（零侵入复现 `B0~B2` 与 `E1~E5` 策略并核算含缓存账单）。
- **Step 4**：仅将实验指标全面跑赢基线的策略组件迁移回 [`gp`](https://github.com/npc1607/gp) 的 `internal/core/compaction` 与 `internal/agent`。
