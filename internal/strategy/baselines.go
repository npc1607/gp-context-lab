package strategy

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/npc1607/gp-context-lab/internal/algo"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/tokenizer"
)

// B0Full sends complete history with zero truncation or masking (accuracy ceiling + full-cost baseline).
type B0Full struct {
	cfg          Config
	systemPrompt string
	turns        []storedTurn
}

func NewB0Full(cfg Config) *B0Full { return &B0Full{cfg: cfg} }
func (s *B0Full) Name() string     { return "B0" }

func (s *B0Full) Reset(systemPrompt string, _ int) {
	s.systemPrompt = systemPrompt
	s.turns = nil
}

func (s *B0Full) OnTurn(turn dataset.Turn) error {
	txt := turn.RenderText()
	s.turns = append(s.turns, storedTurn{
		turn:   turn,
		text:   txt,
		tokens: tokenizer.Count(txt),
	})
	return nil
}

func (s *B0Full) BuildPrompt(query string, _ int) (PromptAssembly, BuildMetrics, error) {
	start := time.Now()
	var hist strings.Builder
	ids := make([]string, 0, len(s.turns))
	for _, st := range s.turns {
		hist.WriteString(formatTurn(st.turn, ""))
		ids = append(ids, st.turn.EntryID)
	}
	asm := PromptAssembly{
		Segments: []PromptSegment{
			makeSystemSegment(s.systemPrompt, s.cfg.BaseToolsTokens),
			makeSegment(SegRecentWindow, hist.String()),
			makeSegment(SegQuery, query),
		},
		SelectedEntryIDs: ids,
		PollutedEntryIDs: ids,
	}
	return asm, BuildMetrics{TotalMs: float64(time.Since(start).Microseconds()) / 1000.0}, nil
}

// B1GPCurrent reproduces gp's current baseline:
// applyObservationMasking(window=5) shifting every turn + coarse natural-language compaction above tokenBudget.
type B1GPCurrent struct {
	cfg          Config
	systemPrompt string
	tokenBudget  int
	turns        []storedTurn
	compactedTo  int
	summaryText  string
	summaryIDs   []string
}

func NewB1GPCurrent(cfg Config) *B1GPCurrent { return &B1GPCurrent{cfg: cfg} }
func (s *B1GPCurrent) Name() string          { return "B1" }

func (s *B1GPCurrent) Reset(systemPrompt string, tokenBudget int) {
	s.systemPrompt = systemPrompt
	s.tokenBudget = tokenBudget
	s.turns = nil
	s.compactedTo = 0
	s.summaryText = ""
	s.summaryIDs = nil
}

func (s *B1GPCurrent) OnTurn(turn dataset.Turn) error {
	txt := turn.RenderText()
	s.turns = append(s.turns, storedTurn{
		turn:   turn,
		text:   txt,
		tokens: tokenizer.Count(txt),
	})
	return nil
}

func (s *B1GPCurrent) BuildPrompt(query string, _ int) (PromptAssembly, BuildMetrics, error) {
	start := time.Now()
	n := len(s.turns)
	maskCutoff := n - 5
	if maskCutoff < 0 {
		maskCutoff = 0
	}

	// Render history from compactedTo onward with per-turn sliding observation masking.
	renderSlice := func(from int) (string, int, []string) {
		var b strings.Builder
		var ids []string
		for i := from; i < n; i++ {
			st := s.turns[i]
			override := ""
			if i < maskCutoff && shouldMaskObservation(st.turn) {
				override = maskedPlaceholder(st.turn)
			}
			b.WriteString(formatTurn(st.turn, override))
			if override == "" {
				ids = append(ids, st.turn.EntryID)
			}
		}
		str := b.String()
		return str, tokenizer.Count(str), ids
	}

	histText, histToks, activeIDs := renderSlice(s.compactedTo)
	// Simulate 0.80 * budget auto-compaction into lossy natural-language summary (misses detail & keeps old statements).
	threshold := int(float64(s.tokenBudget) * 0.80)
	if threshold < 200 {
		threshold = 200
	}
	if histToks > threshold && n-5 > s.compactedTo {
		newCut := n - 5
		var sumBuf strings.Builder
		sumBuf.WriteString(s.summaryText)
		for i := s.compactedTo; i < newCut; i++ {
			st := s.turns[i]
			// Coarse summary keeps user goals/constraints indiscriminately (including old superseded ones!), drops tool internals.
			if st.turn.Role == "user" {
				snippet := st.text
				if len([]rune(snippet)) > 48 {
					snippet = string([]rune(snippet)[:48]) + "..."
				}
				sumBuf.WriteString(fmt.Sprintf("- [turn %d] %s\n", st.turn.TurnID, snippet))
				s.summaryIDs = uniqueAppend(s.summaryIDs, st.turn.EntryID)
			}
		}
		// Keep summary bounded so it mirrors lossy natural-language compaction drift.
		lines := strings.Split(strings.TrimSpace(sumBuf.String()), "\n")
		if len(lines) > 8 {
			// Natural language compaction often retains earliest/noisy lines or overwrites middle ones.
			lines = append(lines[:3], lines[len(lines)-4:]...)
		}
		s.summaryText = strings.Join(lines, "\n") + "\n"
		s.compactedTo = newCut
		histText, histToks, activeIDs = renderSlice(s.compactedTo)
	}

	var prefix strings.Builder
	if s.summaryText != "" {
		prefix.WriteString("[NL_COMPACTION_SUMMARY]\n")
		prefix.WriteString(s.summaryText)
	}
	prefix.WriteString(histText)

	allIDs := uniqueAppend(nil, s.summaryIDs...)
	allIDs = uniqueAppend(allIDs, activeIDs...)

	asm := PromptAssembly{
		Segments: []PromptSegment{
			makeSystemSegment(s.systemPrompt, s.cfg.BaseToolsTokens),
			makeSegment(SegRecentWindow, prefix.String()),
			makeSegment(SegQuery, query),
		},
		SelectedEntryIDs: allIDs,
		PollutedEntryIDs: allIDs,
	}
	return asm, BuildMetrics{TotalMs: float64(time.Since(start).Microseconds()) / 1000.0}, nil
}

// B2NaiveRAG uses recent window + pure dense cosine Top-K within token_budget (no graph, no MMR, no ledger).
type B2NaiveRAG struct {
	cfg          Config
	systemPrompt string
	tokenBudget  int
	embedder     index.Embedder
	turns        []storedTurn
}

func NewB2NaiveRAG(cfg Config) *B2NaiveRAG {
	return &B2NaiveRAG{
		cfg:      cfg,
		embedder: index.NewHashNGramEmbedder(128),
	}
}

func (s *B2NaiveRAG) Name() string { return "B2" }

func (s *B2NaiveRAG) Reset(systemPrompt string, tokenBudget int) {
	s.systemPrompt = systemPrompt
	s.tokenBudget = tokenBudget
	s.turns = nil
}

func (s *B2NaiveRAG) OnTurn(turn dataset.Turn) error {
	txt := turn.RenderText()
	s.turns = append(s.turns, storedTurn{
		turn:   turn,
		text:   txt,
		tokens: tokenizer.Count(txt),
		vec:    s.embedder.Embed(txt),
	})
	return nil
}

func (s *B2NaiveRAG) BuildPrompt(query string, _ int) (PromptAssembly, BuildMetrics, error) {
	start := time.Now()
	n := len(s.turns)
	recentStart := n - s.cfg.RecentWindow
	if recentStart < 0 {
		recentStart = 0
	}

	var recentBuf strings.Builder
	recentTokens := 0
	var selectedIDs []string
	for i := recentStart; i < n; i++ {
		line := formatTurn(s.turns[i].turn, "")
		recentBuf.WriteString(line)
		recentTokens += s.turns[i].tokens
		selectedIDs = append(selectedIDs, s.turns[i].turn.EntryID)
	}

	qVec := s.embedder.Embed(query)
	type scored struct {
		idx int
		sim float64
	}
	pool := make([]scored, 0, recentStart)
	for i := 0; i < recentStart; i++ {
		pool = append(pool, scored{idx: i, sim: index.Cosine(qVec, s.turns[i].vec)})
	}
	sort.Slice(pool, func(i, j int) bool {
		return pool[i].sim > pool[j].sim
	})

	recallBudget := s.tokenBudget - recentTokens
	if recallBudget < 100 {
		recallBudget = s.tokenBudget / 2
	}
	var chosen []algo.Candidate
	usedTok := 0
	for _, p := range pool {
		st := s.turns[p.idx]
		if st.tokens <= 0 || usedTok+st.tokens > recallBudget {
			continue
		}
		chosen = append(chosen, algo.Candidate{
			TurnIdx: p.idx,
			EntryID: st.turn.EntryID,
			Weight:  p.sim,
			Tokens:  st.tokens,
			Vec:     st.vec,
		})
		usedTok += st.tokens
		selectedIDs = uniqueAppend(selectedIDs, st.turn.EntryID)
	}

	wsText := formatWorkingSet(chosen, s.turns)
	asm := PromptAssembly{
		Segments: []PromptSegment{
			makeSystemSegment(s.systemPrompt, s.cfg.BaseToolsTokens),
			makeSegment(SegWorkingSet, wsText),
			makeSegment(SegRecentWindow, recentBuf.String()),
			makeSegment(SegQuery, query),
		},
		SelectedEntryIDs: selectedIDs,
		PollutedEntryIDs: selectedIDs,
		WorkingSetTokens: usedTok,
		WorkingSetCands:  chosen,
	}
	return asm, BuildMetrics{TotalMs: float64(time.Since(start).Microseconds()) / 1000.0}, nil
}
