package strategy

import (
	"fmt"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/algo"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/tokenizer"
)

// SegmentKind classifies prompt segments for Prompt Caching LCP analysis.
type SegmentKind string

const (
	SegSystem       SegmentKind = "system"
	SegLedger       SegmentKind = "ledger"
	SegWorkingSet   SegmentKind = "working_set"
	SegRecentWindow SegmentKind = "recent_window"
	SegQuery        SegmentKind = "query"
)

// PromptSegment represents one contiguous block in the assembled prompt.
type PromptSegment struct {
	Kind     SegmentKind
	Content  string
	TokenIDs []int
}

// PromptAssembly is the output of a ContextStrategy for a single LLM request.
type PromptAssembly struct {
	Segments         []PromptSegment
	SelectedEntryIDs []string
	PollutedEntryIDs []string
	WorkingSetTokens int
	WorkingSetCands  []algo.Candidate
}

// FlattenTokens concatenates all segment TokenIDs in order for LCP cache simulation.
func (p PromptAssembly) FlattenTokens() []int {
	total := 0
	for _, s := range p.Segments {
		total += len(s.TokenIDs)
	}
	out := make([]int, 0, total)
	for _, s := range p.Segments {
		out = append(out, s.TokenIDs...)
	}
	return out
}

// BuildMetrics captures fine-grained local compute breakdown (in ms).
type BuildMetrics struct {
	TotalMs float64
}

// Config parameterizes algorithms and extraction modes across strategies.
type Config struct {
	ExtractorMode   index.ExtractorMode
	Alpha           float64 // PPR teleport/damping (default 0.75)
	HeatT           float64 // Heat kernel diffusion time (default 3.0)
	Lambda          float64 // MMR relevance vs diversity tradeoff (default 0.6)
	Eta             float64 // Token cost exponent (default 0.5)
	ACTRDecay       float64 // ACT-R power-law exponent d (default 0.5)
	Epoch           int     // Epoch-frozen observation masking stride (default 8)
	RecentWindow    int     // Recent verbatim turns kept outside working set (default 4)
	BaseToolsTokens      int     // Synthetic base system/tools schema tokens (e.g., 0 or 1024)
	DegreeBeta           float64 // Hub degree penalty beta (default 0.4)
	WorkingSetFloorRatio float64 // Minimum fraction of tokenBudget guaranteed for WorkingSet (default 0.65)
}

// DefaultConfig returns calibrated default hyperparameters from docs/02 and docs/04.
func DefaultConfig() Config {
	return Config{
		ExtractorMode:        index.ModeHeuristic,
		Alpha:                0.75,
		HeatT:                3.0,
		Lambda:               0.6,
		Eta:                  0.5,
		ACTRDecay:            0.5,
		Epoch:                8,
		RecentWindow:         4,
		BaseToolsTokens:      0,
		DegreeBeta:           0.4,
		WorkingSetFloorRatio: 0.65,
	}
}

// ContextStrategy defines the unified interface implemented by B0~B2 and E1~E5.
type ContextStrategy interface {
	Name() string
	Reset(systemPrompt string, tokenBudget int)
	OnTurn(turn dataset.Turn) error
	BuildPrompt(query string, atTurn int) (PromptAssembly, BuildMetrics, error)
}

type storedTurn struct {
	turn     dataset.Turn
	text     string
	tokens   int
	tokIDs   []int
	vec      []float32
	entities []string
	entSet   map[string]struct{}
	density  float64
	entropy  float64
}

func makeSegment(kind SegmentKind, content string) PromptSegment {
	return PromptSegment{
		Kind:     kind,
		Content:  content,
		TokenIDs: tokenizer.Encode(content),
	}
}

func makeSystemSegment(systemPrompt string, baseToolsTokens int) PromptSegment {
	toks := tokenizer.Encode(systemPrompt)
	if baseToolsTokens > 0 {
		toks = append(toks, tokenizer.PadTokens("gp_tools_schema", baseToolsTokens)...)
	}
	return PromptSegment{
		Kind:     SegSystem,
		Content:  systemPrompt,
		TokenIDs: toks,
	}
}

func formatTurn(t dataset.Turn, contentOverride string) string {
	body := t.RenderText()
	if contentOverride != "" {
		body = contentOverride
	}
	return fmt.Sprintf("[%s#%s] %s\n", t.Role, t.EntryID, body)
}

func formatWorkingSet(cands []algo.Candidate, turns []storedTurn) string {
	if len(cands) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[RECALLED_WORKING_SET]\n")
	for _, c := range cands {
		b.WriteString(formatTurn(turns[c.TurnIdx].turn, ""))
	}
	return b.String()
}

func shouldMaskObservation(t dataset.Turn) bool {
	if t.Role != "tool" {
		return false
	}
	if len(t.Content) > 400 {
		return true
	}
	return strings.Count(t.Content, "\n") > 8
}

func maskedPlaceholder(t dataset.Turn) string {
	lines := strings.Count(t.Content, "\n") + 1
	return fmt.Sprintf("[已折叠历史工具输出: call_id=%s | 原文 %d 行 / %dB]", t.ToolCallID, lines, len(t.Content))
}

func uniqueAppend(dst []string, items ...string) []string {
	seen := make(map[string]struct{}, len(dst)+len(items))
	for _, x := range dst {
		seen[x] = struct{}{}
	}
	for _, x := range items {
		if x == "" {
			continue
		}
		if _, ok := seen[x]; !ok {
			seen[x] = struct{}{}
			dst = append(dst, x)
		}
	}
	return dst
}
