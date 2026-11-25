package report

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/metrics"
)

// StrategyResult summarizes one strategy's offline evaluation across all cases and probes.
type StrategyResult struct {
	Strategy string `json:"strategy"`

	SuiteAMultihopRecall   float64 `json:"suite_a_recall"`
	SuiteABridgeComplete   float64 `json:"suite_a_bridge_complete"`
	SuiteBRecall           float64 `json:"suite_b_recall"`
	SuiteBContamination    float64 `json:"suite_b_contamination"`
	SuiteCRecall           float64 `json:"suite_c_recall"`
	SuiteDRecall           float64 `json:"suite_d_recall"`
	SuiteDInputTokens      int     `json:"suite_d_input_tokens"`
	SuiteDCacheReadTokens  int     `json:"suite_d_cache_read_tokens"`
	SuiteDCacheHitRatio    float64 `json:"suite_d_cache_hit_ratio"`
	SuiteDAnthropicCost    float64 `json:"suite_d_anthropic_cost"`
	SuiteDCostVsB0         float64 `json:"suite_d_cost_reduction_vs_b0"`
	HasSuiteD              bool    `json:"has_suite_d"`
	OverallRecall          float64 `json:"overall_recall"`
	MeanRedundancy         float64 `json:"mean_redundancy"`

	TotalInputTokens       int     `json:"total_input_tokens"`
	TotalLCPTokens         int     `json:"total_lcp_tokens"`
	TotalCacheReadTokens   int     `json:"total_cache_read_tokens"`
	TokenDropVsB0          float64 `json:"token_drop_vs_b0"`
	RawLCPRatio            float64 `json:"raw_lcp_ratio"`
	SimulatedCacheHitRatio float64 `json:"simulated_cache_hit_ratio"`
	AnthropicCostIndex     float64 `json:"anthropic_cost_index"`
	OpenAICostIndex        float64 `json:"openai_cost_index"`
	AnthropicCostVsB0      float64 `json:"anthropic_cost_reduction_vs_b0"`
	OpenAICostVsB0         float64 `json:"openai_cost_reduction_vs_b0"`

	LatencyP50Ms float64 `json:"latency_p50_ms"`
	LatencyP95Ms float64 `json:"latency_p95_ms"`
	LatencyP99Ms float64 `json:"latency_p99_ms"`

	ProbeDetails []metrics.ProbeMetrics `json:"probe_details,omitempty"`
}

// ComputeRelativeCosts fills token and cost reduction vs B0 across all strategy results.
func ComputeRelativeCosts(results []StrategyResult) {
	var b0Anthropic, b0OpenAI, b0SuiteD float64
	var b0Tokens int
	for _, r := range results {
		if r.Strategy == "B0" {
			b0Anthropic = r.AnthropicCostIndex
			b0OpenAI = r.OpenAICostIndex
			b0SuiteD = r.SuiteDAnthropicCost
			b0Tokens = r.TotalInputTokens
			break
		}
	}
	if b0Anthropic <= 0 || b0OpenAI <= 0 {
		return
	}
	for i := range results {
		if b0Tokens > 0 {
			results[i].TokenDropVsB0 = 1.0 - float64(results[i].TotalInputTokens)/float64(b0Tokens)
		}
		results[i].AnthropicCostVsB0 = 1.0 - results[i].AnthropicCostIndex/b0Anthropic
		results[i].OpenAICostVsB0 = 1.0 - results[i].OpenAICostIndex/b0OpenAI
		if b0SuiteD > 0 {
			results[i].SuiteDCostVsB0 = 1.0 - results[i].SuiteDAnthropicCost/b0SuiteD
		}
	}
}

// AggregateSuiteMetrics populates suite averages from ProbeDetails.
func (r *StrategyResult) AggregateSuiteMetrics(latencies []float64) {
	var aSum, aBridgeSum, bRecSum, bContamSum, cSum, dSum, allSum, redSum float64
	var aN, bN, cN, dN int
	for _, p := range r.ProbeDetails {
		allSum += p.EvidenceRecall
		redSum += p.Redundancy
		switch p.Suite {
		case string(dataset.SuiteMultihop):
			aSum += p.EvidenceRecall
			aBridgeSum += p.BridgeCompleteness
			aN++
		case string(dataset.SuiteConflict):
			bRecSum += p.EvidenceRecall
			bContamSum += p.Contamination
			bN++
		case string(dataset.SuiteNoise):
			cSum += p.EvidenceRecall
			cN++
		case string(dataset.SuiteReplay):
			r.HasSuiteD = true
			dSum += p.EvidenceRecall
			dN++
		}
	}
	if aN > 0 {
		r.SuiteAMultihopRecall = aSum / float64(aN)
		r.SuiteABridgeComplete = aBridgeSum / float64(aN)
	}
	if bN > 0 {
		r.SuiteBRecall = bRecSum / float64(bN)
		r.SuiteBContamination = bContamSum / float64(bN)
	}
	if cN > 0 {
		r.SuiteCRecall = cSum / float64(cN)
	}
	if dN > 0 {
		r.SuiteDRecall = dSum / float64(dN)
	}
	if len(r.ProbeDetails) > 0 {
		r.OverallRecall = allSum / float64(len(r.ProbeDetails))
		r.MeanRedundancy = redSum / float64(len(r.ProbeDetails))
	}
	if r.TotalInputTokens > 0 {
		r.RawLCPRatio = float64(r.TotalLCPTokens) / float64(r.TotalInputTokens)
		r.SimulatedCacheHitRatio = float64(r.TotalCacheReadTokens) / float64(r.TotalInputTokens)
	}
	if r.SuiteDInputTokens > 0 {
		r.SuiteDCacheHitRatio = float64(r.SuiteDCacheReadTokens) / float64(r.SuiteDInputTokens)
	}
	r.LatencyP50Ms, r.LatencyP95Ms, r.LatencyP99Ms = metrics.Percentiles(latencies)
}

// PrintTable writes a formatted comparison table and Go/No-Go gate check to w.
func PrintTable(w io.Writer, results []StrategyResult) {
	fmt.Fprintln(w, "====================================================================================================================================================")
	fmt.Fprintf(w, "%-10s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s | %-8s | %-8s | %-8s | %-7s\n",
		"Strategy", "SuiteA%", "Bridge%", "SuiteB%", "Contam%", "SuiteC%", "SuiteD%", "TokDrop", "CacheHit%", "CostDrop", "D-CostDp", "p95(ms)")
	fmt.Fprintln(w, "----------------------------------------------------------------------------------------------------------------------------------------------------")
	for _, r := range results {
		fmt.Fprintf(w, "%-10s | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %7.2f\n",
			r.Strategy,
			r.SuiteAMultihopRecall*100,
			r.SuiteABridgeComplete*100,
			r.SuiteBRecall*100,
			r.SuiteBContamination*100,
			r.SuiteCRecall*100,
			r.SuiteDRecall*100,
			r.TokenDropVsB0*100,
			r.SimulatedCacheHitRatio*100,
			r.AnthropicCostVsB0*100,
			r.SuiteDCostVsB0*100,
			r.LatencyP95Ms,
		)
	}
	fmt.Fprintln(w, "====================================================================================================================================================")
	printGoNoGo(w, results)
}

func printGoNoGo(w io.Writer, results []StrategyResult) {
	byName := make(map[string]StrategyResult, len(results))
	for _, r := range results {
		byName[r.Strategy] = r
	}
	a, hasA := byName["E4-A"]
	b, hasB := byName["E4-B"]
	var target StrategyResult
	var ok bool
	switch {
	case hasA && hasB:
		if a.SuiteDCostVsB0 >= b.SuiteDCostVsB0 {
			target = a
		} else {
			target = b
		}
		ok = true
	case hasA:
		target, ok = a, true
	case hasB:
		target, ok = b, true
	}
	if !ok {
		return
	}
	b1, hasB1 := byName["B1"]
	b2, hasB2 := byName["B2"]

	passRecall := target.SuiteAMultihopRecall >= 0.85
	passLift := !hasB2 || (target.SuiteAMultihopRecall-b2.SuiteAMultihopRecall) >= 0.25
	passContam := target.SuiteBContamination < 0.05
	passCost := target.SuiteDCostVsB0 >= 0.45
	passCache := !hasB1 || target.SuiteDCacheHitRatio >= b1.SuiteDCacheHitRatio
	passLatency := target.LatencyP95Ms <= 30.0

	mark := func(ok bool) string {
		if ok {
			return "[PASS]"
		}
		return "[FAIL]"
	}
	fmt.Fprintf(w, "\nGo / No-Go Gates (%s):\n", target.Strategy)
	fmt.Fprintf(w, "  %s Suite A Recall >= 85%% (actual %.1f%%) & +25pp vs B2 (delta %+.1fpp)\n",
		mark(passRecall && passLift), target.SuiteAMultihopRecall*100, (target.SuiteAMultihopRecall-b2.SuiteAMultihopRecall)*100)
	fmt.Fprintf(w, "  %s Suite B Contamination < 5%% (actual %.1f%%)\n",
		mark(passContam), target.SuiteBContamination*100)
	if target.HasSuiteD {
		fmt.Fprintf(w, "  %s Suite D Cost vs B0 >= 45%% drop (actual %.1f%%) & CacheHit >= B1 (%.1f%% vs %.1f%%)\n",
			mark(passCost && passCache), target.SuiteDCostVsB0*100, target.SuiteDCacheHitRatio*100, b1.SuiteDCacheHitRatio*100)
	} else {
		fmt.Fprintf(w, "  [INFO] Suite D not present yet (current synthetic 34-46t suites: TokDrop %.1f%%, CacheHit %.1f%% vs B1 %.1f%%, CostDrop %.1f%%)\n",
			target.TokenDropVsB0*100, target.SimulatedCacheHitRatio*100, b1.SimulatedCacheHitRatio*100, target.AnthropicCostVsB0*100)
	}
	fmt.Fprintf(w, "  %s Context Build p95 <= 30.00ms (actual %.2fms)\n",
		mark(passLatency), target.LatencyP95Ms)
}

// WriteJSON persists full StrategyResult slice to path (creating parent directories if needed).
func WriteJSON(path string, results []StrategyResult) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
