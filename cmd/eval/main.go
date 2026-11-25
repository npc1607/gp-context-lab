package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/cache"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/metrics"
	"github.com/npc1607/gp-context-lab/internal/report"
	"github.com/npc1607/gp-context-lab/internal/strategy"
	"github.com/npc1607/gp-context-lab/internal/tokenizer"
)

func main() {
	var (
		datasetsDir     = flag.String("datasets", "./datasets", "directory containing benchmark json cases")
		strategiesFlag  = flag.String("strategies", "all", "comma-separated strategy names or 'all'")
		extractorFlag   = flag.String("extractor", "heuristic", "entity/ledger extraction mode: heuristic | oracle")
		includeSample   = flag.Bool("include-sample", false, "include sample-multihop-case.json in evaluation")
		cacheMinTokens  = flag.Int("cache-min-tokens", 1024, "minimum prefix length threshold for Prompt Caching")
		baseToolsTokens = flag.Int("base-tools-tokens", 1024, "synthetic system/tools schema prefix tokens")
		alphaFlag       = flag.Float64("alpha", 0.75, "PPR damping/teleport parameter alpha")
		lambdaFlag      = flag.Float64("lambda", 0.60, "MMR relevance-diversity tradeoff lambda")
		etaFlag         = flag.Float64("eta", 0.50, "token-cost normalization exponent eta")
		decayFlag       = flag.Float64("decay", 0.50, "ACT-R power-law decay exponent d")
		epochFlag       = flag.Int("epoch", 8, "epoch stride for frozen observation masking in E4")
		epochsSweep     = flag.String("epochs", "", "optional comma-separated epoch curve sweep (e.g. 8,12,16,24,32)")
		calibratePath   = flag.String("calibrate-e2e", "", "optional path to e2e JSON report to calibrate tokenizer vs real API input_tokens")
		outPath         = flag.String("out", "", "optional JSON report output path")
	)
	flag.Parse()

	cases, err := dataset.LoadDir(*datasetsDir, *includeSample)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load datasets: %v\n", err)
		os.Exit(1)
	}
	if len(cases) == 0 {
		fmt.Fprintf(os.Stderr, "no benchmark cases found in %s\n", *datasetsDir)
		os.Exit(1)
	}

	cfg := strategy.DefaultConfig()
	cfg.Alpha = *alphaFlag
	cfg.Lambda = *lambdaFlag
	cfg.Eta = *etaFlag
	cfg.ACTRDecay = *decayFlag
	cfg.Epoch = *epochFlag
	cfg.BaseToolsTokens = *baseToolsTokens
	if strings.EqualFold(*extractorFlag, string(index.ModeOracle)) {
		cfg.ExtractorMode = index.ModeOracle
	} else {
		cfg.ExtractorMode = index.ModeHeuristic
	}

	selectedStrategies := buildStrategies(*strategiesFlag, cfg)
	if len(selectedStrategies) == 0 {
		fmt.Fprintf(os.Stderr, "no matching strategies for %q\n", *strategiesFlag)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d benchmark cases | extractor=%s | cache_min_tokens=%d | base_tools_tokens=%d\n\n",
		len(cases), cfg.ExtractorMode, *cacheMinTokens, *baseToolsTokens)

	if *calibratePath != "" {
		if err := runTokenizerCalibration(*calibratePath, cases, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "calibration failed: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if strings.TrimSpace(*epochsSweep) != "" {
		runEpochSweep(*epochsSweep, *strategiesFlag, cases, cfg, *cacheMinTokens)
		return
	}

	results := make([]report.StrategyResult, 0, len(selectedStrategies))
	for _, strat := range selectedStrategies {
		res := runStrategy(strat, cases, *cacheMinTokens)
		results = append(results, res)
	}
	report.ComputeRelativeCosts(results)
	report.PrintTable(os.Stdout, results)

	if *outPath != "" {
		if err := report.WriteJSON(*outPath, results); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write JSON report: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("\nReport written to %s\n", *outPath)
	}
}

func buildStrategies(spec string, cfg strategy.Config) []strategy.ContextStrategy {
	all := []strategy.ContextStrategy{
		strategy.NewB0Full(cfg),
		strategy.NewB1GPCurrent(cfg),
		strategy.NewB2NaiveRAG(cfg),
		strategy.NewE1MMR(cfg),
		strategy.NewE1KDPP(cfg),
		strategy.NewE2PPR(cfg),
		strategy.NewE2Heat(cfg),
		strategy.NewE3(cfg),
		strategy.NewE4A(cfg),
		strategy.NewE4B(cfg),
		strategy.NewE5Surprisal(cfg),
	}
	if strings.EqualFold(strings.TrimSpace(spec), "all") || strings.TrimSpace(spec) == "" {
		return all
	}
	want := make(map[string]bool)
	for _, part := range strings.Split(spec, ",") {
		p := strings.TrimSpace(strings.ToLower(part))
		if p != "" {
			want[p] = true
		}
	}
	var out []strategy.ContextStrategy
	for _, s := range all {
		if want[strings.ToLower(s.Name())] {
			out = append(out, s)
		}
	}
	return out
}

func runStrategy(strat strategy.ContextStrategy, cases []dataset.Case, minCacheTokens int) report.StrategyResult {
	res := report.StrategyResult{
		Strategy: strat.Name(),
	}
	var latencies []float64
	sim := cache.NewSimulator(minCacheTokens)

	for _, c := range cases {
		strat.Reset(c.SystemPrompt, c.TokenBudget)
		sim.Reset()

		probesByTurn := make(map[int][]dataset.EvalProbe)
		for _, p := range c.EvalProbes {
			probesByTurn[p.AtTurn] = append(probesByTurn[p.AtTurn], p)
		}

		for i, turn := range c.Turns {
			if err := strat.OnTurn(turn); err != nil {
				fmt.Fprintf(os.Stderr, "[%s/%s] OnTurn error: %v\n", strat.Name(), c.CaseID, err)
				continue
			}

			// Simulate an LLM request at every user turn or tool response turn (mirroring ReAct loop steps).
			isLastTurn := i == len(c.Turns)-1
			_, hasProbe := probesByTurn[turn.TurnID]
			if turn.Role == "user" || turn.Role == "tool" || isLastTurn {
				if !hasProbe {
					asm, bm, err := strat.BuildPrompt("continue", turn.TurnID)
					if err == nil {
						latencies = append(latencies, bm.TotalMs)
						cs := sim.Step(asm.FlattenTokens())
						accumulateCache(&res, c.Suite, cs)
					}
				}
			}

			for _, probe := range probesByTurn[turn.TurnID] {
				asm, bm, err := strat.BuildPrompt(probe.Query, turn.TurnID)
				if err != nil {
					fmt.Fprintf(os.Stderr, "[%s/%s/%s] BuildPrompt error: %v\n", strat.Name(), c.CaseID, probe.ProbeID, err)
					continue
				}
				latencies = append(latencies, bm.TotalMs)
				cs := sim.Step(asm.FlattenTokens())
				accumulateCache(&res, c.Suite, cs)

				pm := metrics.EvalLayer1(
					c, probe,
					asm.SelectedEntryIDs,
					asm.PollutedEntryIDs,
					asm.WorkingSetTokens,
					asm.WorkingSetCands,
					bm.TotalMs,
				)
				res.ProbeDetails = append(res.ProbeDetails, pm)
			}
		}
	}
	res.AggregateSuiteMetrics(latencies)
	return res
}

func accumulateCache(res *report.StrategyResult, suite dataset.SuiteType, cs cache.TurnCacheStats) {
	res.TotalInputTokens += cs.TotalInputTokens
	res.TotalLCPTokens += cs.LCPTokens
	res.TotalCacheReadTokens += cs.CacheReadTokens
	res.AnthropicCostIndex += cs.AnthropicCost
	res.OpenAICostIndex += cs.OpenAICost
	if suite == dataset.SuiteReplay {
		res.SuiteDInputTokens += cs.TotalInputTokens
		res.SuiteDCacheReadTokens += cs.CacheReadTokens
		res.SuiteDAnthropicCost += cs.AnthropicCost
	}
}

func runEpochSweep(spec, stratSpec string, cases []dataset.Case, baseCfg strategy.Config, minCacheTokens int) {
	fmt.Println("Epoch Sensitivity Curve:")
	fmt.Printf("%-6s | %-8s | %-7s | %-7s | %-7s | %-9s | %-8s | %-8s\n",
		"Epoch", "Strategy", "SuiteA%", "SuiteD%", "TokDrop", "D-Cache%", "D-CostDp", "p95(ms)")
	fmt.Println("-----------------------------------------------------------------------------------")
	for _, part := range strings.Split(spec, ",") {
		ep, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || ep <= 0 {
			continue
		}
		cfg := baseCfg
		cfg.Epoch = ep
		b0 := runStrategy(strategy.NewB0Full(cfg), cases, minCacheTokens)
		b1 := runStrategy(strategy.NewB1GPCurrent(cfg), cases, minCacheTokens)
		_ = b1
		strats := buildStrategies(stratSpec, cfg)
		var results []report.StrategyResult
		results = append(results, b0)
		for _, s := range strats {
			if s.Name() == "B0" {
				continue
			}
			results = append(results, runStrategy(s, cases, minCacheTokens))
		}
		report.ComputeRelativeCosts(results)
		for _, r := range results {
			if r.Strategy == "B0" {
				continue
			}
			fmt.Printf("%-6d | %-8s | %6.1f%% | %6.1f%% | %6.1f%% | %8.1f%% | %7.1f%% | %7.2f\n",
				ep, r.Strategy, r.SuiteAMultihopRecall*100, r.SuiteDRecall*100, r.TokenDropVsB0*100,
				r.SuiteDCacheHitRatio*100, r.SuiteDCostVsB0*100, r.LatencyP95Ms)
		}
	}
}

func runTokenizerCalibration(e2ePath string, cases []dataset.Case, cfg strategy.Config) error {
	raw, err := os.ReadFile(e2ePath)
	if err != nil {
		return err
	}
	var e2e []struct {
		Strategy string `json:"strategy"`
		Probes   []struct {
			CaseID   string `json:"case_id"`
			ProbeID  string `json:"probe_id"`
			GenUsage struct {
				InputTokens int `json:"input_tokens"`
			} `json:"gen_usage"`
		} `json:"probes"`
	}
	if err := json.Unmarshal(raw, &e2e); err != nil {
		return err
	}

	caseMap := make(map[string]dataset.Case, len(cases))
	for _, c := range cases {
		caseMap[c.CaseID] = c
	}

	// Disable synthetic base tools padding so we compare pure prompt text tokens directly.
	cfg.BaseToolsTokens = 0
	var relErrors, ratios []float64
	var estSum, realSum float64
	for _, sr := range e2e {
		strats := buildStrategies(sr.Strategy, cfg)
		if len(strats) == 0 {
			continue
		}
		strat := strats[0]
		probeMap := make(map[string]int)
		for _, p := range sr.Probes {
			if p.GenUsage.InputTokens > 0 {
				probeMap[p.CaseID+"/"+p.ProbeID] = p.GenUsage.InputTokens
			}
		}
		for _, c := range cases {
			strat.Reset(c.SystemPrompt, c.TokenBudget)
			byTurn := make(map[int][]dataset.EvalProbe)
			for _, p := range c.EvalProbes {
				byTurn[p.AtTurn] = append(byTurn[p.AtTurn], p)
			}
			for _, t := range c.Turns {
				_ = strat.OnTurn(t)
				for _, p := range byTurn[t.TurnID] {
					realTok, ok := probeMap[c.CaseID+"/"+p.ProbeID]
					if !ok {
						continue
					}
					asm, _, err := strat.BuildPrompt(p.Query, t.TurnID)
					if err != nil {
						continue
					}
					// Estimate total tokens across system + message segments (+ ~12 message-framing tokens).
					estTok := 0
					for _, seg := range asm.Segments {
						if strings.TrimSpace(seg.Content) != "" {
							estTok += tokenizer.Count(seg.Content) + 4
						}
					}
					ratio := float64(estTok) / float64(realTok)
					relErr := math.Abs(float64(estTok-realTok)) / float64(realTok)
					ratios = append(ratios, ratio)
					relErrors = append(relErrors, relErr)
					estSum += float64(estTok)
					realSum += float64(realTok)
				}
			}
		}
	}
	if len(ratios) == 0 {
		return fmt.Errorf("no matching probes found")
	}
	sort.Float64s(ratios)
	sort.Float64s(relErrors)
	var sumErr, sumRatio float64
	for i := range ratios {
		sumRatio += ratios[i]
		sumErr += relErrors[i]
	}
	n := len(ratios)
	fmt.Printf("Tokenizer Calibration against %s (%d probes):\n", e2ePath, n)
	fmt.Printf("  Total Estimated / Real InputTokens : %.0f / %.0f (overall ratio = %.3fx)\n", estSum, realSum, estSum/realSum)
	fmt.Printf("  Mean Ratio (est/real)              : %.3fx (median = %.3fx)\n", sumRatio/float64(n), ratios[n/2])
	fmt.Printf("  Mean Absolute Relative Error (MAPE): %.1f%% (median = %.1f%%, p95 = %.1f%%)\n",
		sumErr/float64(n)*100, relErrors[n/2]*100, relErrors[int(0.95*float64(n-1))]*100)
	return nil
}
