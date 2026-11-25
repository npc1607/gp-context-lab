package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/cache"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/metrics"
	"github.com/npc1607/gp-context-lab/internal/report"
	"github.com/npc1607/gp-context-lab/internal/strategy"
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
