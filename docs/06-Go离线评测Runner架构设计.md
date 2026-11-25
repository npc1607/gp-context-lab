# 06 · Phase 2 Go 离线评测 Runner 架构设计

> 本文档定义 Phase 2 **纯 Go 离线评测 Runner** 的工程架构、核心接口、双模式抽取防泄露机制、逐轮缓存仿真流水线及性能优化设计。  
> **上游依据**：
> - [02 - 数学工具清单与选型论证](02-数学工具清单与选型论证.md)
> - [04 - 实验设计与评测基准](04-实验设计与评测基准.md)
> - [05 - 打分与评判机制设计](05-打分与评判机制设计.md)
> - [`datasets/case.schema.json`](../datasets/case.schema.json)

---

## 1. 设计目标与工程边界

Phase 2 的核心任务是：**在不调用外部大模型生成回答的前提下**，用纯 Go 在本地毫秒级跑通全量 8 组策略（`B0~B2`、`E1~E5`）在 4 类基准数据集上的表现，输出两类量化数据：

1. **Layer 1 检索质量指标**：`Evidence Recall`（$R_{\text{ev}}$）、`Bridge Completeness`（$B_{\text{comp}}$）、`Contamination Rate`（$P_{\text{contam}}$）、`Token Efficiency`（$U_{\text{tok}}$）与 `Redundancy`；
2. **缓存/成本仿真与计算耗时**：逐轮回放的 **LCP 缓存命中率**、**Anthropic / OpenAI 双口径归一化账单**，以及本地组装上下文的 **p50 / p95 / p99 耗时（准出门槛 p95 $\le 30\text{ ms}$）**。

**核心约束**：
- **零外部重型依赖**：单机纯 Go 内存运行，不引入 Milvus / Qdrant / Neo4j 或本地推理服务；
- **严防数据泄露**：默认关闭数据集中的黄金标注字段，真实还原线上抽取难度；
- **可向 Phase 3 / Phase 4 平滑演进**：`algo/` 与 `ledger/`、`index/` 保持与 `gp` 主循环解耦，后续可直接迁移至 [`gp`](https://github.com/npc1607/gp) 的 `internal/core/compaction` 与 `internal/agent`。

---

## 2. Go 模块与目录结构

```text
gp-context-lab/
├── go.mod
├── cmd/
│   └── eval/
│       └── main.go              # CLI：执行实验矩阵、参数网格扫描、导出报表
├── internal/
│   ├── dataset/                 # 加载与校验 datasets/*.json（对应 case.schema.json）
│   ├── tokenizer/               # 统一 Token 估算与切片（供预算控制与 LCP 缓存仿真）
│   ├── index/
│   │   ├── code_tokenize.go     # 代码分词器（CamelCase / snake_case / 路径斜杠）
│   │   ├── bm25.go              # 增量 BM25 倒排索引
│   │   ├── entity.go            # 实体抽取（Heuristic 正则/AST vs Oracle 标注对照）
│   │   ├── graph.go             # 异构图（V_msg, V_ent, V_rule）+ IDF 与度惩罚
│   │   └── embedder.go          # 离线向量接口（本地投影 / 预计算磁盘缓存）
│   ├── algo/                    # 纯函数数学库（无状态，便于单测 & go test -bench）
│   │   ├── ppr.go               # 个性化 PageRank（Power Iteration）
│   │   ├── heatkernel.go        # 图热核扩散（Chebyshev / Taylor 近似）
│   │   ├── actr.go              # ACT-R 幂律激活 + 逻辑轮次访问表
│   │   ├── mmr.go               # Budgeted MMR
│   │   ├── kdpp.go              # Budgeted k-DPP（增量 Cholesky 分解）
│   │   └── rrf.go               # RRF / 加权融合 + zlib 压缩比信息密度惩罚 ρ
│   ├── ledger/
│   │   ├── ledger.go            # ContextLedger 状态机（add / supersede / resolve）
│   │   └── extractor.go         # 约束变更抽取（规则/启发式 vs Oracle 对照）
│   ├── strategy/                # Strategy 接口及 B0~B2、E1~E5 实现
│   ├── cache/
│   │   └── simulator.go         # Prompt Caching 仿真（LCP、1024 门槛、双厂商计费）
│   ├── metrics/
│   │   └── layer1.go            # R_ev, B_comp, P_contam, U_tok, Redundancy, Latency
│   └── report/
│       └── report.go            # 控制台汇总表 + Markdown / JSON 导出 + Go/No-Go 门禁判定
└── datasets/
```

---

## 3. 核心接口与数据结构

### 3.1 `ContextStrategy` 统一策略接口

`B0~B2` 基线与 `E1~E5` 实验组实现同一个接口，保证单变量消融对比严格公平：

```go
package strategy

type ContextStrategy interface {
	Name() string
	Reset(systemPrompt string, tokenBudget int)
	// OnTurn 在每轮对话入库时触发：增量更新图、BM25、ACT-R 访问记录、Ledger / Epoch 状态
	OnTurn(turn dataset.Turn) (UpdateStats, error)
	// BuildPrompt 在模拟发请求时组装当前 Prompt，返回分段结构供 LCP 缓存仿真与 Layer 1 评分
	BuildPrompt(query string, atTurn int) (PromptAssembly, BuildMetrics, error)
}
```

### 3.2 `PromptAssembly`（同时服务缓存仿真与防污染判定）

```go
type SegmentKind string

const (
	SegSystem       SegmentKind = "system"
	SegLedger       SegmentKind = "ledger"
	SegWorkingSet   SegmentKind = "working_set"
	SegRecentWindow SegmentKind = "recent_window"
	SegQuery        SegmentKind = "query"
)

type PromptSegment struct {
	Kind     SegmentKind
	Content  string
	TokenIDs []int // 预切分好的 token 序列，用于 O(minLen) 精确计算 LCP
}

type PromptAssembly struct {
	Segments []PromptSegment

	// 供 Layer 1 指标计算（严格对齐 docs/05 第 2 节）
	SelectedEntryIDs []string // 进入上下文的全部有效来源（Ledger active provenance + WorkingSet + RecentWindow）
	PollutedEntryIDs []string // 以“有效正文/有效约束”形式出现的条目（不含 Ledger 极简 supersedes 脚注）
	WorkingSetTokens int
}
```

> **设计要点**：在 [05 - 打分与评判机制设计](05-打分与评判机制设计.md#L43-L48) 中，`ContextLedger` 为防止模型回滚误判而保留的极简 `supersedes` 变更脚注**不算污染**；只有将已作废条目以有效上下文或有效约束形式注入工作集时才计入 $P_{\text{contam}}$。将 `SelectedEntryIDs` 与 `PollutedEntryIDs` 分开返回，可使 Layer 1 判定 100% 精确。

---

## 4. 逐轮回放流水线与离线工程细节

### 4.1 逐轮回放引擎（Turn-by-Turn Replay Loop）

如果只在最后一个 `at_turn` 调用一次 `BuildPrompt`，只能测出 `Evidence Recall`，无法测出 **Prompt Caching 命中率**。Runner 主循环需按真实对话节奏逐轮推进：

1. 按 `turn = 1..T` 顺序调用 `strategy.OnTurn(turn)`；
2. 每当到达模拟请求点（`role == "user"` 或工具结果返回后准备下一步决策时），调用 `strategy.BuildPrompt(...)`；
3. 将本次序列化后的 `TokenIDs` 与上一轮请求的 `TokenIDs` 送入 `cache.Simulator` 计算 **最长公共前缀（LCP）**：
   - 若 $\text{LCP}(P_t, P_{t-1}) \ge 1024$，命中部分记为 $T_{\text{read}}(t)$，其余部分记为 $T_{\text{uncached}}(t)$（Anthropic 口径同时计入 $T_{\text{write}}$）；
   - 借此精确对比 `B1`（每轮滑窗掩码改写 $T-5$ 轮导致前缀击穿）与 `E4-A` / `E4-B`（Epoch 冻结 + 拼接顺序优化）的真实账单差异。
4. 当 `turn_id == probe.at_turn` 时，调用 `BuildPrompt(probe.query, probe.at_turn)` 并核算该 Probe 的 Layer 1 指标。

### 4.2 双模式抽取器（严防数据集标注泄露）

[`sample-multihop-case.json`](../datasets/sample-multihop-case.json) 与正式套件中包含 `mentioned_entities` 和 `constraint_event` 两个标注字段。为防止离线算法“偷看答案”，抽取器提供两种模式：

- **`--extractor=heuristic`（默认 / 准出验收口径）**：完全屏蔽 `mentioned_entities` 和 `constraint_event`，仅用纯 Go 正则/符号扫描（文件路径、Go 包/函数名、错误码、命令行参数）与规则状态机从原始 `content` 和 `tool_calls` 中实时抽取。
- **`--extractor=oracle`（消融诊断口径）**：允许读取标注真值，用于隔离“抽取器误差”与“图扩散/去重/衰减算法误差”。

### 4.3 离线稠密向量与负对照 `E5` 实现

- **`S_dense`（`index.Embedder`）**：
  1. `HashNGramEmbedder`：纯 Go 字符 n-gram + 随机投影，0 外部依赖，用于单元测试与结构联调；
  2. `CachedFileEmbedder`：支持读取预计算并缓存到 `.cache/embeddings.jsonl` 的真实语义向量，离线调参阶段以 $< 1\text{ ms}$ 延迟获得真实语义相似度。
- **负对照 `E5`（裸 Surprisal）**：离线模式下基于本地字符/子词 Unigram 分布估算单条消息平均自信息量 $H(d_i) = -\frac{1}{L}\sum \log P(t)$，无需在线 LLM logprob 即可在 Suite C 上稳定复现 UUID / panic 十六进制堆栈挤占关键约束的劣化效应。

---

## 5. 性能优化保障（死守 $\text{p95} \le 30\text{ ms}$ 红线）

1. **增量索引更新**：`OnTurn` 阶段完成单条分词、BM25 倒排追加、异构图节点/边增量写入及 `compress/zlib` 信息密度 $\rho(d_i)$ 预计算与缓存；`BuildPrompt` 热路径绝不重建全图。
2. **扁平 CSR / 连续切片表示**：图节点映射为连续 `int32` ID，PPR 幂迭代复用预分配的 `[]float64` 双缓冲切片（`rCurr` / `rNext`），消除热路径 map 寻址与 GC 抖动。
3. **两阶段候选截断**：`BM25 + Dense + PPR` 粗排取 Top-$N$（$N \le 200$）后再送入 `Budgeted MMR` 或 `Budgeted k-DPP`，保证子集选择耗时稳定在 $1\text{–}2\text{ ms}$ 以内。

---

## 6. CLI 运行与 Go / No-Go 验收输出规划

```bash
# 1. 跑全量 8 组消融实验 + 全部 datasets/suite_*.json，输出汇总表与 Go/No-Go 校验
go run ./cmd/eval --datasets ./datasets --strategies all --extractor heuristic --out ./reports/baseline.json

# 2. 横向对比 MMR vs k-DPP、PPR vs Heat Kernel，并采集 p50/p95/p99 基准耗时
go run ./cmd/eval --strategies E1-mmr,E1-kdpp,E2-ppr,E2-heat --bench

# 3. 网格参数扫描（调优 α, λ, η, d, Epoch）
go run ./cmd/eval --sweep --strategy E4-B --alpha 0.70,0.75,0.85 --epoch 4,8,12
```

报表模块自动按 [04 - 实验设计与评测基准](04-实验设计与评测基准.md#L101-L106) 输出 `[PASS/FAIL]` 准出核验结果：
- Suite A Multi-Hop Recall $\ge 85\%$ 且高于 `B2` $\ge 25\%$；
- Suite B Contamination Rate $< 5\%$；
- Suite D 含缓存总账单较 `B0` 降幅 $\ge 45\%$ 且缓存命中率不低于 `B1`；
- 本地纯 Go 上下文组装延迟 p95 $\le 30\text{ ms}$。
