package strategy

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/npc1607/gp-context-lab/internal/algo"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/ledger"
	"github.com/npc1607/gp-context-lab/internal/tokenizer"
)

type expVariant struct {
	name         string
	useGraph     bool
	useHeat      bool
	useBM25      bool
	useMMR       bool
	useKDPP      bool
	useLedger    bool
	useACTR      bool
	useEpochMask bool
	recallLast   bool // E4-B puts working set after epoch-frozen recent window
	useSurprisal bool // E5 negative control
}

// ExperimentalStrategy implements E1..E5 cleanly via parameterized pipeline toggles.
type ExperimentalStrategy struct {
	cfg          Config
	variant      expVariant
	systemPrompt string
	tokenBudget  int

	embedder  index.Embedder
	entExt    index.EntityExtractor
	ledgerExt ledger.Extractor

	bm25       *index.BM25
	graph      *index.Graph
	actr       *algo.ACTRTracker
	ctxLedger  *ledger.ContextLedger
	turns      []storedTurn
	entityHist map[string][]int // entity -> turnIdxs mentioning it (for ACT-R reinforcement)

	// Cached ledger prefix to preserve byte stability when unchanged.
	lastLedgerVer  int
	lastLedgerText string

	// Cached working set across non-query continuation steps within an Epoch (E4).
	lastWSQuery  string
	lastWSEpoch  int
	lastWSChosen []algo.Candidate
	lastWSText   string
	lastWSToks   int
}

func NewE1MMR(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E1-mmr", useMMR: true})
}
func NewE1KDPP(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E1-kdpp", useKDPP: true})
}
func NewE2PPR(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E2-ppr", useGraph: true, useBM25: true, useMMR: true})
}
func NewE2Heat(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E2-heat", useGraph: true, useHeat: true, useBM25: true, useMMR: true})
}
func NewE3(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E3", useGraph: true, useBM25: true, useMMR: true, useLedger: true, useACTR: true})
}
func NewE4A(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E4-A", useGraph: true, useBM25: true, useMMR: true, useLedger: true, useACTR: true, useEpochMask: true, recallLast: false})
}
func NewE4B(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E4-B", useGraph: true, useBM25: true, useMMR: true, useLedger: true, useACTR: true, useEpochMask: true, recallLast: true})
}
func NewE5Surprisal(cfg Config) *ExperimentalStrategy {
	return newExp(cfg, expVariant{name: "E5", useGraph: true, useBM25: true, useMMR: true, useSurprisal: true})
}

func newExp(cfg Config, v expVariant) *ExperimentalStrategy {
	return &ExperimentalStrategy{
		cfg:      cfg,
		variant:  v,
		embedder: index.NewHashNGramEmbedder(128),
	}
}

func (s *ExperimentalStrategy) Name() string { return s.variant.name }

func (s *ExperimentalStrategy) Reset(systemPrompt string, tokenBudget int) {
	s.systemPrompt = systemPrompt
	s.tokenBudget = tokenBudget
	if s.cfg.ExtractorMode == index.ModeOracle {
		s.entExt = index.OracleExtractor{}
		s.ledgerExt = ledger.OracleExtractor{}
	} else {
		s.entExt = index.HeuristicExtractor{}
		s.ledgerExt = ledger.NewHeuristicExtractor()
	}
	s.bm25 = index.NewBM25()
	s.graph = index.NewGraph()
	s.actr = algo.NewACTRTracker(s.cfg.ACTRDecay, -1.2, 1.0)
	s.ctxLedger = ledger.NewContextLedger()
	s.turns = nil
	s.entityHist = make(map[string][]int)
	s.lastLedgerVer = -1
	s.lastLedgerText = ""
	s.lastWSQuery = ""
	s.lastWSEpoch = -1
	s.lastWSChosen = nil
	s.lastWSText = ""
	s.lastWSToks = 0
}

func (s *ExperimentalStrategy) OnTurn(turn dataset.Turn) error {
	turnIdx := len(s.turns)
	txt := turn.RenderText()
	tokIDs := tokenizer.Encode(txt)
	vec := s.embedder.Embed(txt)
	ents := s.entExt.ExtractTurn(turn)

	entSet := make(map[string]struct{}, len(ents))
	for _, e := range ents {
		entSet[e] = struct{}{}
	}

	st := storedTurn{
		turn:     turn,
		text:     txt,
		tokens:   len(tokIDs),
		tokIDs:   tokIDs,
		vec:      vec,
		entities: ents,
		entSet:   entSet,
		density:  algo.ZlibDensityPenalty(txt),
		entropy:  algo.UnigramSurprisal(txt),
	}
	s.turns = append(s.turns, st)

	if s.variant.useBM25 {
		s.bm25.AddDoc(txt)
	}
	if s.variant.useGraph {
		s.graph.AddTurn(turnIdx, turn, ents, vec)
	}
	if s.variant.useACTR {
		s.actr.EnsureTurn(turnIdx, turn.TurnID)
		for _, e := range ents {
			for _, prevIdx := range s.entityHist[e] {
				s.actr.RecordAccess(prevIdx, turn.TurnID)
			}
			s.entityHist[e] = append(s.entityHist[e], turnIdx)
		}
	}
	if s.variant.useLedger {
		s.ledgerExt.ProcessTurn(turn, s.ctxLedger)
	}
	return nil
}

func (s *ExperimentalStrategy) BuildPrompt(query string, atTurn int) (PromptAssembly, BuildMetrics, error) {
	start := time.Now()
	n := len(s.turns)

	// 1. Determine recent verbatim window & epoch-frozen masking boundary.
	recentStart := n - s.cfg.RecentWindow
	if recentStart < 0 {
		recentStart = 0
	}
	var recentBuf strings.Builder
	recentTokens := 0
	var recentIDs []string
	if s.variant.useEpochMask && s.cfg.Epoch > 1 {
		// In Epoch-frozen mode, keep an append-only block from the latest epoch checkpoint
		// (frozen older tool outputs in prior epochs stay masked; current epoch stays verbatim).
		epochStep := s.cfg.Epoch
		epochStart := ((n - 1) / epochStep) * epochStep
		if epochStart < 0 {
			epochStart = 0
		}
		if n-epochStart < s.cfg.RecentWindow && epochStart >= epochStep {
			epochStart -= epochStep
		}
		recentStart = epochStart
		freezeBoundary := ((n - 1) / epochStep) * epochStep
		for i := recentStart; i < n; i++ {
			st := s.turns[i]
			override := ""
			if i < freezeBoundary && shouldMaskObservation(st.turn) {
				override = maskedPlaceholder(st.turn)
			}
			line := formatTurn(st.turn, override)
			recentBuf.WriteString(line)
			recentTokens += tokenizer.Count(line)
			if override == "" {
				recentIDs = append(recentIDs, st.turn.EntryID)
			}
		}
	} else {
		for i := recentStart; i < n; i++ {
			line := formatTurn(s.turns[i].turn, "")
			recentBuf.WriteString(line)
			recentTokens += s.turns[i].tokens
			recentIDs = append(recentIDs, s.turns[i].turn.EntryID)
		}
	}

	// 2. Render canonical L1 ContextLedger (only active constraints + minimal supersedes footnotes).
	ledgerText := ""
	ledgerToks := 0
	var activeLedgerIDs []string
	var supersededSet map[string]struct{}
	if s.variant.useLedger {
		if s.ctxLedger.Version != s.lastLedgerVer {
			s.lastLedgerText = s.ctxLedger.CanonicalRender()
			s.lastLedgerVer = s.ctxLedger.Version
		}
		ledgerText = s.lastLedgerText
		ledgerToks = tokenizer.Count(ledgerText)
		activeLedgerIDs = s.ctxLedger.ActiveProvenanceEntryIDs()
		supersededSet = s.ctxLedger.SupersededEntryIDs()
	}

	// 3. Score historical candidate turns before recentStart.
	qVec := s.embedder.Embed(query)
	denseScores := make([]float64, n)
	for i := 0; i < n; i++ {
		denseScores[i] = math.Max(0, index.Cosine(qVec, s.turns[i].vec))
	}

	var bm25Norm []float64
	if s.variant.useBM25 {
		bm25Norm = algo.NormalizeMinMax(s.bm25.Score(query))
	}

	var graphNorm []float64
	if s.variant.useGraph {
		qEnts := s.entExt.ExtractQuery(query)
		recentIdxs := make([]int, 0, n-recentStart)
		for i := n - 1; i >= recentStart; i-- {
			recentIdxs = append(recentIdxs, i)
		}
		seed := s.graph.BuildSeedDistribution(qEnts, recentIdxs)
		var diffused []float64
		if s.variant.useHeat {
			diffused = algo.HeatKernel(s.graph, seed, s.cfg.HeatT, 8, s.cfg.DegreeBeta)
		} else {
			diffused = algo.PersonalizedPageRank(s.graph, seed, s.cfg.Alpha, 20, s.cfg.DegreeBeta)
		}
		graphNorm = algo.NormalizeMinMax(s.graph.ProjectMessageScores(diffused, 120.0))
	}

	var rrfScores []float64
	if s.variant.useGraph || s.variant.useBM25 {
		signals := [][]float64{algo.NormalizeMinMax(denseScores)}
		weights := []float64{0.25}
		if s.variant.useBM25 {
			signals = append(signals, bm25Norm)
			weights = append(weights, 0.35)
		}
		if s.variant.useGraph {
			signals = append(signals, graphNorm)
			weights = append(weights, 0.55)
		}
		rrfScores = algo.WeightedRRF(signals, weights, 15.0)
	}

	// Compute max entropy for E5 negative control normalization.
	maxEntropy := 1.0
	if s.variant.useSurprisal {
		maxEntropy = 1e-6
		for i := 0; i < recentStart; i++ {
			if s.turns[i].entropy > maxEntropy {
				maxEntropy = s.turns[i].entropy
			}
		}
	}

	cands := make([]algo.Candidate, 0, recentStart)
	for i := 0; i < recentStart; i++ {
		st := s.turns[i]
		// E3/E4 suppresses entries belonging to superseded constraints from working-set recall.
		if s.variant.useLedger {
			if _, isSup := supersededSet[st.turn.EntryID]; isSup {
				continue
			}
		}

		var w float64
		if s.variant.useSurprisal {
			// Negative control E5: raw surprisal dominates candidate ranking (demonstrating high-entropy noise takeover).
			w = st.entropy / maxEntropy
		} else if s.variant.useGraph || s.variant.useBM25 {
			baseRel := 0.20*denseScores[i] + 0.35*bm25Norm[i] + 0.45*graphNorm[i]
			if len(rrfScores) == n {
				baseRel = 0.55*baseRel + 0.45*rrfScores[i]
			}
			timeW := 1.0
			if s.variant.useACTR {
				timeW = s.actr.TimeWeight(i, atTurn)
			}
			w = (0.3 + 0.7*timeW) * st.density * baseRel
		} else {
			// E1 (MMR / k-DPP over pure dense similarity).
			w = denseScores[i]
		}

		cands = append(cands, algo.Candidate{
			TurnIdx:  i,
			EntryID:  st.turn.EntryID,
			Weight:   w,
			Tokens:   st.tokens,
			Vec:      st.vec,
			Entities: st.entSet,
		})
	}

	// Two-stage candidate truncation to N <= 200 and min-max normalize weights into [0, 1]
	// so MMR relevance vs redundancy trade-off is scale-invariant.
	sort.Slice(cands, func(i, j int) bool {
		return cands[i].Weight > cands[j].Weight
	})
	if len(cands) > 200 {
		cands = cands[:200]
	}
	if len(cands) > 0 && cands[0].Weight > 0 {
		maxW := cands[0].Weight
		for i := range cands {
			cands[i].Weight /= maxW
		}
	}

	// 4. Select diverse subset within recall budget (E4 reuses working set during continuation steps inside the same Epoch).
	recallBudget := s.tokenBudget - ledgerToks - recentTokens
	if recallBudget < 160 {
		recallBudget = s.tokenBudget / 2
	}
	curEpoch := 0
	if s.cfg.Epoch > 0 && n > 0 {
		curEpoch = (n - 1) / s.cfg.Epoch
	}
	var chosen []algo.Candidate
	var wsText string
	var wsToks int
	if s.variant.useEpochMask && query == "continue" && s.lastWSEpoch == curEpoch && s.lastWSChosen != nil {
		chosen = s.lastWSChosen
		wsText = s.lastWSText
		wsToks = s.lastWSToks
	} else {
		switch {
		case s.variant.useKDPP:
			chosen = algo.SelectBudgetedKDPP(cands, recallBudget, s.cfg.Eta)
		case s.variant.useMMR:
			chosen = algo.SelectBudgetedMMR(cands, recallBudget, s.cfg.Lambda, s.cfg.Eta)
		default:
			used := 0
			for _, c := range cands {
				if c.Tokens > 0 && used+c.Tokens <= recallBudget {
					chosen = append(chosen, c)
					used += c.Tokens
				}
			}
		}

		// Sort chosen working-set entries in chronological turn order for coherent reading.
		sort.Slice(chosen, func(i, j int) bool {
			return chosen[i].TurnIdx < chosen[j].TurnIdx
		})
		for _, c := range chosen {
			wsToks += c.Tokens
		}
		wsText = formatWorkingSet(chosen, s.turns)
		if s.variant.useEpochMask {
			s.lastWSQuery = query
			s.lastWSEpoch = curEpoch
			s.lastWSChosen = chosen
			s.lastWSText = wsText
			s.lastWSToks = wsToks
		}
	}

	wsIDs := make([]string, 0, len(chosen))
	for _, c := range chosen {
		wsIDs = append(wsIDs, c.EntryID)
	}

	// 5. Assemble segments (respecting E4-A vs E4-B cache layout ordering).
	segs := []PromptSegment{makeSystemSegment(s.systemPrompt, s.cfg.BaseToolsTokens)}
	if ledgerText != "" {
		segs = append(segs, makeSegment(SegLedger, ledgerText))
	}
	if s.variant.recallLast {
		// E4-B: stable prefix through recent window, dynamic working set placed right before query.
		segs = append(segs,
			makeSegment(SegRecentWindow, recentBuf.String()),
			makeSegment(SegWorkingSet, wsText),
			makeSegment(SegQuery, query),
		)
	} else {
		segs = append(segs,
			makeSegment(SegWorkingSet, wsText),
			makeSegment(SegRecentWindow, recentBuf.String()),
			makeSegment(SegQuery, query),
		)
	}

	// SelectedEntryIDs includes active ledger provenance; PollutedEntryIDs excludes ledger footnotes.
	pollutedIDs := uniqueAppend(nil, wsIDs...)
	pollutedIDs = uniqueAppend(pollutedIDs, recentIDs...)
	selectedIDs := uniqueAppend(nil, activeLedgerIDs...)
	selectedIDs = uniqueAppend(selectedIDs, pollutedIDs...)

	asm := PromptAssembly{
		Segments:         segs,
		SelectedEntryIDs: selectedIDs,
		PollutedEntryIDs: pollutedIDs,
		WorkingSetTokens: wsToks,
		WorkingSetCands:  chosen,
	}
	return asm, BuildMetrics{TotalMs: float64(time.Since(start).Microseconds()) / 1000.0}, nil
}
