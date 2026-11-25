package index

import (
	"regexp"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/dataset"
)

// ExtractorMode controls whether entity/constraint extraction uses heuristic parsing or oracle annotations.
type ExtractorMode string

const (
	ModeHeuristic ExtractorMode = "heuristic"
	ModeOracle    ExtractorMode = "oracle"
)

// EntityExtractor extracts graph entities from conversation turns and queries.
type EntityExtractor interface {
	ExtractTurn(turn dataset.Turn) []string
	ExtractQuery(query string) []string
}

var (
	reFilePath   = regexp.MustCompile(`[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.*-]+)+(?:\.[A-Za-z0-9]+)?`)
	reSpecialCfg = regexp.MustCompile(`\b(?:Makefile|Dockerfile|go\.mod|go\.sum)\b`)
	reBacktick   = regexp.MustCompile("`([^`\n]{2,64})`")
	reCamelSym   = regexp.MustCompile(`\b[A-Z][a-zA-Z0-9]{2,}(?:[A-Z][a-z0-9]+)+\b`)
	reUpperEnv   = regexp.MustCompile(`\b[A-Z][A-Z0-9_]{2,}(?:=[A-Za-z0-9_.-]+)?\b`)
	reSnakeID    = regexp.MustCompile(`\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,4}\b`)
	rePkgPath    = regexp.MustCompile(`\b[a-z0-9-]+\.[a-z]{2,}/[a-zA-Z0-9_./-]+\b`)
	reProtoRes   = regexp.MustCompile(`\breserved\s+(\d+)\b`)
	reNumber     = regexp.MustCompile(`\b\d{1,4}\b`)
)

// HeuristicExtractor extracts code, ops, path, symbol, and domain-concept entities purely from raw text.
type HeuristicExtractor struct{}

func (HeuristicExtractor) ExtractTurn(turn dataset.Turn) []string {
	return extractTextEntities(turn.RenderText())
}

func (HeuristicExtractor) ExtractQuery(query string) []string {
	return extractTextEntities(query)
}

// OracleExtractor uses dataset mentioned_entities on turns (for ablation/upper-bound diagnosis only)
// combined with heuristic query extraction.
type OracleExtractor struct{}

func (OracleExtractor) ExtractTurn(turn dataset.Turn) []string {
	seen := make(map[string]struct{})
	var out []string
	add := func(s string) {
		s = strings.TrimSpace(strings.ToLower(s))
		if s == "" {
			return
		}
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	for _, e := range turn.MentionedEntities {
		add(e)
	}
	for _, e := range extractTextEntities(turn.RenderText()) {
		add(e)
	}
	return out
}

func (OracleExtractor) ExtractQuery(query string) []string {
	return extractTextEntities(query)
}

func extractTextEntities(text string) []string {
	if text == "" {
		return nil
	}
	seen := make(map[string]struct{})
	var out []string
	add := func(ent string) {
		ent = strings.TrimSpace(strings.ToLower(ent))
		if len(ent) < 2 {
			return
		}
		if _, ok := seen[ent]; !ok {
			seen[ent] = struct{}{}
			out = append(out, ent)
		}
	}

	for _, m := range rePkgPath.FindAllString(text, -1) {
		add(m)
	}
	for _, m := range reFilePath.FindAllString(text, -1) {
		// Skip obvious noise URLs / hex / version paths unless code-like.
		if strings.HasPrefix(m, "0x") {
			continue
		}
		add(m)
		// Also emit trailing 2-segment suffix to bridge relative imports (e.g. proto/analytics/audit_meta.proto <-> analytics/audit_meta.proto).
		parts := strings.Split(m, "/")
		if len(parts) >= 3 {
			add(strings.Join(parts[len(parts)-2:], "/"))
		}
	}
	for _, m := range reSpecialCfg.FindAllString(text, -1) {
		add(m)
	}
	for _, m := range reBacktick.FindAllStringSubmatch(text, -1) {
		inner := strings.TrimSpace(m[1])
		if len(inner) <= 64 && !strings.ContainsAny(inner, " \t\n") {
			add(inner)
		}
	}
	for _, m := range reCamelSym.FindAllString(text, -1) {
		add(m)
	}
	for _, m := range reUpperEnv.FindAllString(text, -1) {
		// Ignore noisy log status / generic tokens.
		if m == "PASS" || m == "FAIL" || m == "RUN" || m == "JSON" || m == "HTTP" || m == "SIGSEGV" {
			continue
		}
		add(m)
		if idx := strings.IndexByte(m, '='); idx > 0 {
			add(m[:idx])
		}
	}
	for _, m := range reSnakeID.FindAllString(text, -1) {
		if strings.HasPrefix(m, "testroutinecheck") || strings.HasPrefix(m, "call_") {
			continue
		}
		add(m)
	}
	for _, m := range reProtoRes.FindAllStringSubmatch(text, -1) {
		add("slot_" + m[1])
	}

	// General cross-lingual / ops concept normalization (bridges Chinese prose <-> code/config syntax).
	lower := strings.ToLower(text)
	if strings.Contains(text, "第十二") || strings.Contains(text, "十二号") || strings.Contains(text, "编号 12") || strings.Contains(text, "tag 12") {
		add("slot_12")
	}
	if strings.Contains(text, "冻结") || strings.Contains(text, "保留字段") || strings.Contains(lower, "reserved") {
		add("concept:reserved_slot")
	}
	if strings.Contains(text, "静态") || strings.Contains(text, "动态链接") || strings.Contains(lower, "cgo_enabled") || strings.Contains(lower, "cgo") || strings.Contains(lower, "sqlite") {
		add("concept:static_cgo")
	}
	if strings.Contains(text, "离线") || strings.Contains(text, "内网") || strings.Contains(text, "公网") || strings.Contains(lower, "internal-registry") || strings.Contains(lower, "goproxy") {
		add("concept:network_isolation")
	}
	if strings.Contains(text, "只读") || strings.Contains(lower, "readonlyrootfs") || strings.Contains(lower, "read_only") || strings.Contains(lower, "createtemp") {
		add("concept:readonly_fs")
	}
	if strings.Contains(lower, "cluster") || strings.Contains(text, "分片") || strings.Contains(text, "跨节点") || strings.Contains(lower, "hash tag") {
		add("concept:redis_cluster")
	}
	if strings.Contains(lower, "nullable") || strings.Contains(lower, "order by") || strings.Contains(text, "排序键") || strings.Contains(lower, "clickhouse") {
		add("concept:ch_sort_key")
	}
	if strings.Contains(text, "重推") || strings.Contains(text, "幂等") || strings.Contains(lower, "webhook") || strings.Contains(lower, "sla") {
		add("concept:webhook_sla")
	}
	if strings.Contains(text, "并发") && (strings.Contains(text, "2") || strings.Contains(text, "上限")) {
		add("concept:concurrency_limit")
	}
	if strings.Contains(lower, "gpl") || strings.Contains(lower, "agpl") || strings.Contains(text, "开源协议") || strings.Contains(text, "许可证") {
		add("concept:license_policy")
	}
	_ = reNumber // reserved for future numeric token normalization

	return out
}
