# Benchmark Datasets（评测基准数据集）

本目录存放面向 [`gp`](https://github.com/npc1607/gp) 上下文权重与缓存优化实验的评测用例，所有用例均符合 [`case.schema.json`](case.schema.json) 规范。

> 注：除快速演示示例 [`sample-multihop-case.json`](sample-multihop-case.json)（8 轮精简版）外，Suite A/B/C 共 23 个精标基准用例均构造了 **34–46 轮对话**、累计体积达 **3.7x–6.9x `token_budget`**（含 5 个无显式关键词的抗过拟合泛化用例 `suite_b_07`–`11`）；Suite D 共 5 个长会话回放用例构造了 **98–138 轮对话**、累计体积达 **40k+ tokens**。

---

## 套件分类与用例清单（共 29 个）

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
7. [`suite_b_07_implicit_db_revert.json`](suite_b_07_implicit_db_revert.json)（36 轮 / 900 预算）：无显式关键词的口语化数据库回退（“还是用回 MySQL 8.0 吧”），专测启发式关键词表泛化边界。
8. [`suite_b_08_implicit_cache_drop.json`](suite_b_08_implicit_cache_drop.json)（36 轮 / 900 预算）：无显式关键词搁置集群缓存（“先用 `sync.Map` 顶一下，之前说的集群方案先不上了”）。
9. [`suite_b_09_implicit_auth_override.json`](suite_b_09_implicit_auth_override.json)（36 轮 / 900 预算）：口语化同步覆盖鉴权方式（“统一走 OAuth2 Bearer Token，老的 header 留着也没人用了”）。
10. [`suite_b_10_implicit_log_format.json`](suite_b_10_implicit_log_format.json)（36 轮 / 900 预算）：隐晦切换日志输出格式（“顺手统一成结构化 JSON 吧，别打纯文本了”）。
11. [`suite_b_11_implicit_retry_policy.json`](suite_b_11_implicit_retry_policy.json)（36 轮 / 900 预算）：口语修改重试策略上限（“调成最多 2 次带指数退避，别再连发 5 次了”）。

### 3. Suite C：高熵噪音与复读机风暴（`suite_c_noise`，6 个）
专测 **Budgeted MMR / k-DPP 去冗余 + 密度过滤**，同时用于反证 **裸 Surprisal（$-\log P$）与朴素 Top-K** 的失效率：
1. [`suite_c_01_panic_stack_storm.json`](suite_c_01_panic_stack_storm.json)（34 轮 / 5491 tok vs 800 预算）：6 段 Go panic 内存地址堆栈 + 12 轮 typo 复读掩埋“并发上限=2”硬限制。
2. [`suite_c_02_uuid_webhook_noise.json`](suite_c_02_uuid_webhook_noise.json)（34 轮 / 5181 tok vs 800 预算）：海量 UUIDv4 / SHA512 载荷 + 状态码确认复读掩埋“`X-Sig` 剔除 `trace_id`”验签规则。
3. [`suite_c_03_minified_js_and_css_loop.json`](suite_c_03_minified_js_and_css_loop.json)（34 轮 / 3355 tok vs 800 预算）：混淆 JS chunk + CSS padding 微调复读掩埋微前端 `/console/v2/finance` 前缀约束。
4. [`suite_c_04_k8s_metrics_vs_gomemlimit.json`](suite_c_04_k8s_metrics_vs_gomemlimit.json)（34 轮 / 4168 tok vs 800 预算）：SHA256 digest / Prometheus 直方图 + 探针复读掩埋“512Mi 容器缺 `GOMEMLIMIT`”根因。
5. [`suite_c_05_sql_explain_and_lint_loop.json`](suite_c_05_sql_explain_and_lint_loop.json)（34 轮 / 3268 tok vs 800 预算）：SQL `EXPLAIN` 随机数值树 + unused-var lint 复读掩埋 `(tenant_id, status, created_at)` 最左前缀规则。
6. [`suite_c_06_ci_hashes_vs_license_redline.json`](suite_c_06_ci_hashes_vs_license_redline.json)（34 轮 / 3475 tok vs 800 预算）：CI SHA512 下载刷屏日志 + 依赖补丁小升级复读掩埋“禁引入 GPL/AGPL-3.0”法务红线。

### 4. Suite D：长会话回放与真实含缓存账单（`suite_d_replay`，5 个，98–138 轮）
专测 **Epoch 冻结掩码 + Prompt 拼接顺序（`E4-A` vs `E4-B`）** 在百轮级长会话中的真实缓存命中率、含缓存总账单降幅与长程多跳/抗污染稳定性：
1. [`suite_d_01_grpc_gateway_refactor.json`](suite_d_01_grpc_gateway_refactor.json)（98 轮 / 2400 预算）：gRPC Gateway 重构、REST JSON 切 Protobuf 及离线 VPC 多跳红线。
2. [`suite_d_02_k8s_oom_incident_replay.json`](suite_d_02_k8s_oom_incident_replay.json)（110 轮 / 2400 预算）：K8s OOM 排障、缓存切 Redis Cluster 及只读根文件系统多跳排障。
3. [`suite_d_03_pg_partition_migration.json`](suite_d_03_pg_partition_migration.json)（118 轮 / 2400 预算）：PostgreSQL 16 分区在线迁移与废弃 MySQL 8.0 抗污染验证。
4. [`suite_d_04_ci_runner_cache_debug.json`](suite_d_04_ci_runner_cache_debug.json)（130 轮 / 2400 预算）：ARM64 纯静态零 CGO 构建红线跨文件多跳关联与长 CI 日志回放。
5. [`suite_d_05_multi_tenant_rate_limiter.json`](suite_d_05_multi_tenant_rate_limiter.json)（138 轮 / 2400 预算）：Redis Cluster 同槽位 Hash Tag 约束跨文件关联与鉴权演进。
