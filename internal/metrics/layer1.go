package metrics

import (
	"math"
	"sort"

	"github.com/npc1607/gp-context-lab/internal/algo"
	"github.com/npc1607/gp-context-lab/internal/dataset"
)

// ProbeMetrics holds Layer 1 deterministic retrieval metrics for a single EvalProbe.
type ProbeMetrics struct {
	CaseID             string  `json:"case_id"`
	Suite              string  `json:"suite"`
	ProbeID            string  `json:"probe_id"`
	EvidenceRecall     float64 `json:"evidence_recall"`
	BridgeCompleteness float64 `json:"bridge_completeness"`
	Contamination      float64 `json:"contamination"`
	TokenUtilization   float64 `json:"token_utilization"`
	Redundancy         float64 `json:"redundancy"`
	BuildLatencyMs     float64 `json:"build_latency_ms"`
}

// EvalLayer1 evaluates Layer 1 retrieval quality metrics strictly following docs/05 Section 2.
func EvalLayer1(
	c dataset.Case,
	probe dataset.EvalProbe,
	selectedEntryIDs []string,
	pollutedEntryIDs []string,
	workingSetTokens int,
	workingSetCands []algo.Candidate,
	latencyMs float64,
) ProbeMetrics {
	selSet := make(map[string]struct{}, len(selectedEntryIDs))
	for _, id := range selectedEntryIDs {
		selSet[id] = struct{}{}
	}
	polSet := make(map[string]struct{}, len(pollutedEntryIDs))
	for _, id := range pollutedEntryIDs {
		polSet[id] = struct{}{}
	}

	reqHit := 0
	for _, req := range probe.RequiredEvidenceEntryIDs {
		if _, ok := selSet[req]; ok {
			reqHit++
		}
	}
	recall := 1.0
	bridge := 1.0
	if len(probe.RequiredEvidenceEntryIDs) > 0 {
		recall = float64(reqHit) / float64(len(probe.RequiredEvidenceEntryIDs))
		if reqHit < len(probe.RequiredEvidenceEntryIDs) {
			bridge = 0.0
		}
	}

	contam := 0.0
	for _, forb := range probe.ForbiddenSupersededEntryIDs {
		if _, ok := polSet[forb]; ok {
			contam = 1.0
			break
		}
	}

	util := 0.0
	if c.TokenBudget > 0 {
		util = float64(workingSetTokens) / float64(c.TokenBudget)
	}

	redundancy := MeanPairwiseSimilarity(workingSetCands)

	return ProbeMetrics{
		CaseID:             c.CaseID,
		Suite:              string(c.Suite),
		ProbeID:            probe.ProbeID,
		EvidenceRecall:     recall,
		BridgeCompleteness: bridge,
		Contamination:      contam,
		TokenUtilization:   util,
		Redundancy:         redundancy,
		BuildLatencyMs:     latencyMs,
	}
}

// MeanPairwiseSimilarity computes average pairwise similarity across selected working-set items.
func MeanPairwiseSimilarity(cands []algo.Candidate) float64 {
	n := len(cands)
	if n < 2 {
		return 0
	}
	var sum float64
	pairs := 0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sum += algo.PairwiseSimilarity(cands[i], cands[j])
			pairs++
		}
	}
	if pairs == 0 {
		return 0
	}
	return sum / float64(pairs)
}

// Percentiles computes p50, p95, and p99 from a slice of latency measurements (in ms).
func Percentiles(values []float64) (p50, p95, p99 float64) {
	if len(values) == 0 {
		return 0, 0, 0
	}
	cp := append([]float64(nil), values...)
	sort.Float64s(cp)
	return percentileSorted(cp, 0.50), percentileSorted(cp, 0.95), percentileSorted(cp, 0.99)
}

func percentileSorted(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	pos := p * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1.0-frac) + sorted[hi]*frac
}
