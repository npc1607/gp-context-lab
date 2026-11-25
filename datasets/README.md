# Benchmark Datasets（评测基准数据集）

本目录存放面向 [`gp`](https://github.com/npc1607/gp) 上下文权重与缓存优化实验的评测用例，所有用例均符合 [`case.schema.json`](case.schema.json) 规范。

> 注：除快速演示示例 [`sample-multihop-case.json`](sample-multihop-case.json)（8 轮精简版）外，18 个正式基准用例均构造了 **34–46 轮对话**、累计体积达 **3.7x–6.9x `token_budget`**，确保任何“全量塞入”或“仅看近期 5–8 轮窗口”的朴素策略都会失效，从而有效区分各组数学工具的真实能力。

---

## 套件分类与用例清单（共 19 个）

### 0. 入门示例
- [`sample-multihop-case.json`](sample-multihop-case.json)：8 轮复合示例（离线 VPC 构建 + MySQL 切 PostgreSQL）。

### 1. Suite A：隐性多跳关联（`suite_a_multihop`，6 个，A-B 词面重合率均为 `0.000`）
专测 **PPR / 热核扩散** 跨越无字面重合桥梁（文件依赖、符号引用、配置映射）捞回早期约束的能力：
1. [`suite_a_01_proto_tag_reservation.json`](suite_a_01_proto_tag_reservation.json)（42 轮 / 3509 tok vs 900 预算）：审计归档槽位 12 冻结 $\to$ `audit_meta.proto` $\to$ `order.proto` `OrderExtra` 新增字段排号。
2. [`suite_a_02_arm64_cgo_cross_compile.json`](suite_a_02_arm64_cgo_cross_compile.json)（42 轮 / 3434 tok vs 900 预算）：极简静态容器跨架构部署 $\to$ `Makefile` `CGO_ENABLED=0` $\to$ `OpenLocalCache` 能否切 `mattn/go-sqlite3`。
3. [`suite_a_03_redis_cluster_hash_tag.json`](suite_a_03_redis_cluster_hash_tag.json)（42 轮 / 3473 tok vs 900 预算）：六分片集群禁止跨节点多键 $\to$ `BuildKeys` / `ratelimit.lua` 线上报错根因。
4. [`suite_a_04_k8s_readonly_rootfs.json`](suite_a_04_k8s_readonly_rootfs.json)（42 轮 / 3453 tok vs 900 预算）：只读根文件系统安全规范 $\to$ Helm `values.yaml` $\to$ `RenderSummaryFile` `os.CreateTemp` 失败。
5. [`suite_a_05_clickhouse_nullable_sortkey.json`](suite_a_05_clickhouse_nullable_sortkey.json)（42 轮 / 3476 tok vs 900 预算）：ClickHouse 排序键禁空约定 $\to$ `001_events.sql` $\to$ `EventDTO.ActorID` 改 `*string` 风险。
6. [`suite_a_06_webhook_idempotency_retry_window.json`](suite_a_06_webhook_idempotency_retry_window.json)（42 轮 / 3524 tok vs 900 预算）：渠道 3s 应答 SLA 与指数重推 $\to$ `HandleChannelCallback` $\to$ `ProcessSync` 同步 4.2s PDF 导致重复入账。

### 2. Suite B：约束演进与状态冲突（`suite_b_conflict`，6 个）
专测 **结构化约束账本（`ContextLedger`）+ ACT-R 幂律遗忘 + Provenance** 压制已作废规则（`forbidden_superseded_entry_ids`）的能力：
1. [`suite_b_01_auth_three_stage_evolution.json`](suite_b_01_auth_three_stage_evolution.json)（46 轮 / 3494 tok vs 900 预算，2 个 Probe）：`X-API-Key` $\to$ `mTLS` $\to$ 5 分钟短效 EdDSA JWT（含废弃原因溯源 Probe）。
2. [`suite_b_02_api_version_and_envelope_rewrite.json`](suite_b_02_api_version_and_envelope_rewrite.json)（40 轮 / 3376 tok vs 900 预算）：`/v1/tasks` + `{code,msg,data}` $\to$ `/v2/jobs` + HTTP 202 + RFC 9457 ProblemDetails。
3. [`suite_b_03_cache_policy_aba_revert.json`](suite_b_03_cache_policy_aba_revert.json)（42 轮 / 3425 tok vs 900 预算）：本地 LRU (60s) $\to$ Redis Cluster $\to$ 回滚 16 分片本地 LRU + `singleflight` (15s) 的 **A $\to$ B $\to$ A'** 经典回滚测试。
4. [`suite_b_04_mq_and_codec_partial_supersession.json`](suite_b_04_mq_and_codec_partial_supersession.json)（42 轮 / 3386 tok vs 900 预算）：先换传输层（Kafka $\to$ NATS JetStream）、再换编码（JSON $\to$ Protobuf）的正交分段覆盖测试。
5. [`suite_b_05_deploy_env_and_branch_switch.json`](suite_b_05_deploy_env_and_branch_switch.json)（40 轮 / 3336 tok vs 900 预算）：`develop`/`staging-us` $\to$ `release/*`/`prod-sg` + 10% 灰度发布切换。
6. [`suite_b_06_error_model_upgrade.json`](suite_b_06_error_model_upgrade.json)（40 轮 / 3373 tok vs 900 预算）：哨兵错误/字符串匹配 $\to$ `*errs.BizError` + `errors.As` + 错误码常量联动。

### 3. Suite C：高熵噪音与复读机风暴（`suite_c_noise`，6 个）
专测 **Budgeted MMR / k-DPP 去冗余 + 密度过滤**，同时用于反证 **裸 Surprisal（$-\log P$）与朴素 Top-K** 的失效率：
1. [`suite_c_01_panic_stack_storm.json`](suite_c_01_panic_stack_storm.json)（34 轮 / 5491 tok vs 800 预算）：6 段 Go panic 内存地址堆栈 + 12 轮 typo 复读掩埋“并发上限=2”硬限制。
2. [`suite_c_02_uuid_webhook_noise.json`](suite_c_02_uuid_webhook_noise.json)（34 轮 / 5181 tok vs 800 预算）：海量 UUIDv4 / SHA512 载荷 + 状态码确认复读掩埋“`X-Sig` 剔除 `trace_id`”验签规则。
3. [`suite_c_03_minified_js_and_css_loop.json`](suite_c_03_minified_js_and_css_loop.json)（34 轮 / 3355 tok vs 800 预算）：混淆 JS chunk + CSS padding 微调复读掩埋微前端 `/console/v2/finance` 前缀约束。
4. [`suite_c_04_k8s_metrics_vs_gomemlimit.json`](suite_c_04_k8s_metrics_vs_gomemlimit.json)（34 轮 / 4168 tok vs 800 预算）：SHA256 digest / Prometheus 直方图 + 探针复读掩埋“512Mi 容器缺 `GOMEMLIMIT`”根因。
5. [`suite_c_05_sql_explain_and_lint_loop.json`](suite_c_05_sql_explain_and_lint_loop.json)（34 轮 / 3268 tok vs 800 预算）：SQL `EXPLAIN` 随机数值树 + unused-var lint 复读掩埋 `(tenant_id, status, created_at)` 最左前缀规则。
6. [`suite_c_06_ci_hashes_vs_license_redline.json`](suite_c_06_ci_hashes_vs_license_redline.json)（34 轮 / 3475 tok vs 800 预算）：CI SHA512 下载刷屏日志 + 依赖补丁小升级复读掩埋“禁引入 GPL/AGPL-3.0”法务红线。
