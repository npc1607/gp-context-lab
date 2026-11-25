package ledger

import (
	"fmt"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
)

// Extractor inspects a turn and updates ContextLedger without leaking oracle annotations in heuristic mode.
type Extractor interface {
	ProcessTurn(turn dataset.Turn, l *ContextLedger)
}

// OracleExtractor applies dataset turn.ConstraintEvent directly (diagnostic upper bound only).
type OracleExtractor struct{}

func (OracleExtractor) ProcessTurn(turn dataset.Turn, l *ContextLedger) {
	if turn.ConstraintEvent == nil {
		return
	}
	ev := turn.ConstraintEvent
	l.Apply(ev.Action, ev.ConstraintID, "spec", ev.Statement, ev.Supersedes, turn.EntryID, turn.TurnID)
}

// HeuristicExtractor infers constraint additions and supersessions purely from user text
// using explicit rule/prohibition markers and domain-topic matching.
type HeuristicExtractor struct {
	seq      int
	byTopic  map[string]string // topic -> active constraint ID
}

func NewHeuristicExtractor() *HeuristicExtractor {
	return &HeuristicExtractor{
		byTopic: make(map[string]string),
	}
}

func (h *HeuristicExtractor) ProcessTurn(turn dataset.Turn, l *ContextLedger) {
	// Only extract constraints from user declarations (never blindly from noisy tool dumps).
	if turn.Role != "user" {
		return
	}
	text := strings.TrimSpace(turn.Content)
	if text == "" || strings.HasSuffix(text, "?") || strings.HasSuffix(text, "？") {
		return
	}

	isSupersede := containsAny(text,
		"改用", "废除", "废弃", "放弃", "切换", "迁移至", "切到", "回滚", "推翻", "不再兼容", "替换", "取代", "否定", "下线", "改为", "改走", "更新为",
	)
	isConstraint := isSupersede || containsAny(text,
		"禁止", "严禁", "不得", "不能", "必须", "红线", "硬性", "强制", "冻结", "固定为", "上限", "约定", "规范", "要求", "采用", "初期", "初始", "先按", "仅对接", "直接部署", "触发分支", "选型",
	)
	if !isConstraint {
		return
	}

	topics := inferTopics(text)
	for _, topic := range topics {
		h.seq++
		newID := fmt.Sprintf("cst_%s_%02d", topic, h.seq)
		prevID := ""
		if isSupersede {
			if oldID, ok := h.byTopic[topic]; ok {
				prevID = oldID
			} else if len(topics) == 1 {
				prevID = findBestOverlapConstraint(text, l)
			}
		}

		action := "add"
		if prevID != "" {
			action = "supersede"
		}

		l.Apply(action, newID, topic, text, prevID, turn.EntryID, turn.TurnID)
		h.byTopic[topic] = newID
	}
}

func findBestOverlapConstraint(text string, l *ContextLedger) string {
	terms := index.CodeTokenize(text)
	termSet := make(map[string]struct{}, len(terms))
	for _, t := range terms {
		termSet[t] = struct{}{}
	}
	bestID := ""
	bestScore := 0
	for id, c := range l.Constraints {
		if c.Status != StatusActive {
			continue
		}
		overlap := 0
		for _, ct := range index.CodeTokenize(c.Statement) {
			if _, ok := termSet[ct]; ok {
				overlap++
			}
		}
		if overlap > bestScore {
			bestScore = overlap
			bestID = id
		}
	}
	if bestScore >= 2 {
		return bestID
	}
	return ""
}

func inferTopics(text string) []string {
	lower := strings.ToLower(text)
	var topics []string
	add := func(t string) {
		topics = append(topics, t)
	}
	if containsAny(lower, "mysql", "postgres", "postgresql", "jsonb") {
		add("database")
	}
	if containsAny(lower, "x-api-key", "mtls", "jwt", "eddsa", "bearer", "鉴权", "认证") {
		add("auth")
	}
	if containsAny(lower, "lru", "redis", "ttl", "缓存") {
		add("cache")
	}
	if containsAny(lower, "kafka", "nats", "jetstream", "传输层", "消息总线", "mq") {
		add("mq")
	}
	if containsAny(lower, "json", "protobuf", "序列化", "二进制帧", "编码") && !containsAny(lower, "jsonb") {
		add("codec")
	}
	if containsAny(lower, "/v1/", "/v2/", "envelope", "problemdetails", "接口版本", "路由") {
		add("api")
	}
	if containsAny(lower, "staging", "prod", "develop", "release", "灰度", "分支", "发布", "部署") {
		add("deploy")
	}
	if containsAny(lower, "error", "bizerror", "errors.as", "错误模型", "哨兵错误") {
		add("error_model")
	}
	if containsAny(lower, "gpl", "agpl", "license", "开源协议", "许可证") {
		add("license")
	}
	if containsAny(lower, "并发", "worker", "qps") {
		add("concurrency")
	}
	if containsAny(lower, "cgo", "alpine", "arm64", "静态编译", "纯静态", "静态容器") {
		add("build_env")
	}
	if containsAny(lower, "内网", "离线", "公网", "vpc") {
		add("network")
	}
	if containsAny(lower, "槽位", "编号", "proto", "reserved") && !containsAny(lower, "protobuf") {
		add("proto_tag")
	}
	if containsAny(lower, "只读", "readonly", "rootfs") {
		add("fs_policy")
	}
	if containsAny(lower, "x-sig", "trace_id", "验签", "签名") {
		add("webhook_sig")
	}
	if len(topics) == 0 {
		add("general")
	}
	return topics
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
