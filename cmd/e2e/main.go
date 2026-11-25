package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/judge"
	"github.com/npc1607/gp-context-lab/internal/llm"
	"github.com/npc1607/gp-context-lab/internal/metrics"
	"github.com/npc1607/gp-context-lab/internal/strategy"
)

type probeE2EResult struct {
	CaseID        string             `json:"case_id"`
	Suite         string             `json:"suite"`
	ProbeID       string             `json:"probe_id"`
	RunID         string             `json:"run_id"`
	Layer1        metrics.ProbeMetrics `json:"layer1"`
	Layer2        judge.Layer2Result `json:"layer2"`
	Layer3        judge.Layer3Result `json:"layer3"`
	Pass          bool               `json:"pass"`
	Quadrant      judge.Quadrant     `json:"quadrant"`
	Answer        string             `json:"answer"`
	GenUsage      llm.Usage          `json:"gen_usage"`
	GenLatencyMs  float64            `json:"gen_latency_ms"`
}

type strategyE2EReport struct {
	Strategy        string           `json:"strategy"`
	PassRate        float64          `json:"pass_rate"`
	MeanRubricScore float64          `json:"mean_rubric_score"`
	MeanRecall      float64          `json:"mean_recall"`
	ContamRate      float64          `json:"contam_rate"`
	TotalInputTok   int              `json:"total_input_tokens"`
	TotalCacheRead  int              `json:"total_cache_read_tokens"`
	TotalCacheWrite int              `json:"total_cache_write_tokens"`
	TotalOutputTok  int              `json:"total_output_tokens"`
	CacheReadRatio  float64          `json:"cache_read_ratio"`
	EstAnthropicUSD float64          `json:"est_anthropic_cost_units"`
	Utility         float64          `json:"utility"`
	Quadrants       map[string]int   `json:"quadrants"`
	Probes          []probeE2EResult `json:"probes"`
}

func main() {
	var (
		datasetsDir   = flag.String("datasets", "./datasets", "benchmark dataset directory")
		strategiesStr = flag.String("strategies", "B0,B1,B2,E4-A,E4-B", "comma-separated strategies to evaluate end-to-end")
		providerFlag  = flag.String("provider", "mock", "generation LLM provider: mock | openai | anthropic | deepseek | qcoder")
		baseURLFlag   = flag.String("base-url", "", "optional custom base URL override for generation provider")
		modelFlag     = flag.String("model", "gpt-4o-mini", "generation model name")
		judgeProvFlag = flag.String("judge-provider", "", "judge LLM provider (defaults to -provider)")
		judgeBaseURL  = flag.String("judge-base-url", "", "optional custom base URL override for judge provider")
		judgeModel    = flag.String("judge-model", "", "judge model name (defaults to -model)")
		extractorFlag = flag.String("extractor", "heuristic", "heuristic | oracle")
		epochFlag     = flag.Int("epoch", 16, "epoch stride for E4")
		cacheDir      = flag.String("cache-dir", ".cache/llm", "disk response cache directory (empty disables)")
		limitCases    = flag.Int("limit", 0, "optional max number of cases to run (0 = all)")
		pairwiseFlag  = flag.String("pairwise", "E4-A,E4-B", "optional pair of strategies for symmetric position-swap comparison")
		outPath       = flag.String("out", "", "optional JSON report output path")
	)
	flag.Parse()

	if *judgeProvFlag == "" {
		*judgeProvFlag = *providerFlag
	}
	if *judgeModel == "" {
		*judgeModel = *modelFlag
	}

	cases, err := dataset.LoadDir(*datasetsDir, false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load datasets: %v\n", err)
		os.Exit(1)
	}
	if *limitCases > 0 && *limitCases < len(cases) {
		cases = cases[:*limitCases]
	}

	genClient, err := buildClient(*providerFlag, *baseURLFlag, *cacheDir, false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init gen client: %v\n", err)
		os.Exit(1)
	}
	judgeClient, err := buildClient(*judgeProvFlag, *judgeBaseURL, *cacheDir, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init judge client: %v\n", err)
		os.Exit(1)
	}
	atomicJudge := judge.NewAtomicJudge(judgeClient, *judgeModel)

	cfg := strategy.DefaultConfig()
	cfg.Epoch = *epochFlag
	if strings.EqualFold(*extractorFlag, "oracle") {
		cfg.ExtractorMode = index.ModeOracle
	}

	strats := selectStrategies(*strategiesStr, cfg)
	if len(strats) == 0 {
		fmt.Fprintf(os.Stderr, "no strategies matched %q\n", *strategiesStr)
		os.Exit(1)
	}

	ctx := context.Background()
	reports := make([]strategyE2EReport, 0, len(strats))
	answersByStrat := make(map[string]map[string]string) // strat -> case/probe -> answer
	passVectors := make(map[string][]bool)

	for _, strat := range strats {
		rep := strategyE2EReport{
			Strategy:  strat.Name(),
			Quadrants: map[string]int{},
		}
		ansMap := make(map[string]string)
		var passVec []bool
		var rubricSum, recallSum, contamSum float64

		for _, c := range cases {
			strat.Reset(c.SystemPrompt, c.TokenBudget)
			probesByTurn := make(map[int][]dataset.EvalProbe)
			for _, p := range c.EvalProbes {
				probesByTurn[p.AtTurn] = append(probesByTurn[p.AtTurn], p)
			}

			for _, turn := range c.Turns {
				_ = strat.OnTurn(turn)
				for _, probe := range probesByTurn[turn.TurnID] {
					asm, bm, err := strat.BuildPrompt(probe.Query, turn.TurnID)
					if err != nil {
						fmt.Fprintf(os.Stderr, "[%s] BuildPrompt err: %v\n", strat.Name(), err)
						continue
					}
					l1 := metrics.EvalLayer1(c, probe, asm.SelectedEntryIDs, asm.PollutedEntryIDs, asm.WorkingSetTokens, asm.WorkingSetCands, bm.TotalMs)

					req := toChatRequest(*modelFlag, c.SystemPrompt, asm)
					resp, err := genClient.Chat(ctx, req)
					if err != nil {
						fmt.Fprintf(os.Stderr, "[%s/%s/%s] LLM chat err: %v\n", strat.Name(), c.CaseID, probe.ProbeID, err)
						continue
					}

					l2 := judge.EvalLayer2(resp.Content, probe.DeterministicChecks)
					var l3 judge.Layer3Result
					if l2.Passed {
						runID := opaqueRunID(strat.Name(), c.CaseID, probe.ProbeID)
						l3, err = atomicJudge.JudgeProbe(ctx, runID, probe.Query, resp.Content, probe.ExpectedRubric)
						if err != nil {
							fmt.Fprintf(os.Stderr, "[%s/%s/%s] Judge err: %v\n", strat.Name(), c.CaseID, probe.ProbeID, err)
						}
					}

					passed := l2.Passed && l3.RubricScore == 1.0
					l1OK := l1.BridgeCompleteness == 1.0 && l1.Contamination == 0.0
					quad := judge.ClassifyQuadrant(l1OK, passed)

					rep.Quadrants[string(quad)]++
					rep.TotalInputTok += resp.Usage.InputTokens
					rep.TotalCacheRead += resp.Usage.CacheReadTokens
					rep.TotalCacheWrite += resp.Usage.CacheWriteTokens
					rep.TotalOutputTok += resp.Usage.OutputTokens
					uncached := resp.Usage.InputTokens - resp.Usage.CacheReadTokens - resp.Usage.CacheWriteTokens
					if uncached < 0 {
						uncached = 0
					}
					rep.EstAnthropicUSD += float64(uncached)*1.0 + float64(resp.Usage.CacheWriteTokens)*1.25 + float64(resp.Usage.CacheReadTokens)*0.10

					if passed {
						rep.PassRate++
					}
					rubricSum += l3.RubricScore
					recallSum += l1.EvidenceRecall
					contamSum += l1.Contamination
					passVec = append(passVec, passed)
					ansMap[c.CaseID+"/"+probe.ProbeID] = resp.Content

					rep.Probes = append(rep.Probes, probeE2EResult{
						CaseID:       c.CaseID,
						Suite:        string(c.Suite),
						ProbeID:      probe.ProbeID,
						RunID:        opaqueRunID(strat.Name(), c.CaseID, probe.ProbeID),
						Layer1:       l1,
						Layer2:       l2,
						Layer3:       l3,
						Pass:         passed,
						Quadrant:     quad,
						Answer:       resp.Content,
						GenUsage:     resp.Usage,
						GenLatencyMs: resp.LatencyMs,
					})
				}
			}
		}

		nProbes := float64(len(rep.Probes))
		if nProbes > 0 {
			rep.PassRate /= nProbes
			rep.MeanRubricScore = rubricSum / nProbes
			rep.MeanRecall = recallSum / nProbes
			rep.ContamRate = contamSum / nProbes
		}
		if rep.TotalInputTok > 0 {
			rep.CacheReadRatio = float64(rep.TotalCacheRead) / float64(rep.TotalInputTok)
		}
		reports = append(reports, rep)
		answersByStrat[strat.Name()] = ansMap
		passVectors[strat.Name()] = passVec
	}

	// Compute utility relative to B0 cost (or first strategy if B0 absent).
	b0Cost := 0.0
	for _, r := range reports {
		if r.Strategy == "B0" {
			b0Cost = r.EstAnthropicUSD
			break
		}
	}
	if b0Cost == 0 && len(reports) > 0 {
		b0Cost = reports[0].EstAnthropicUSD
	}
	for i := range reports {
		reports[i].Utility = judge.ComputeUtility(reports[i].PassRate, reports[i].MeanRecall, reports[i].EstAnthropicUSD, b0Cost)
	}

	printE2ESummary(reports, passVectors)

	// Optional symmetric pairwise comparison (e.g. E4-A vs E4-B).
	if parts := strings.Split(*pairwiseFlag, ","); len(parts) == 2 {
		sA, sB := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		if mapA, okA := answersByStrat[sA]; okA {
			if mapB, okB := answersByStrat[sB]; okB {
				runPairwise(ctx, atomicJudge, cases, sA, sB, mapA, mapB)
			}
		}
	}

	if *outPath != "" {
		if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err == nil {
			if data, err := json.MarshalIndent(reports, "", "  "); err == nil {
				_ = os.WriteFile(*outPath, data, 0o644)
				fmt.Printf("\nE2E report saved to %s\n", *outPath)
			}
		}
	}
}

func toChatRequest(model, system string, asm strategy.PromptAssembly) llm.ChatRequest {
	msgs := make([]llm.Message, 0, len(asm.Segments))
	for _, seg := range asm.Segments {
		if seg.Kind == strategy.SegSystem || strings.TrimSpace(seg.Content) == "" {
			continue
		}
		cacheCtrl := seg.Kind == strategy.SegLedger || seg.Kind == strategy.SegWorkingSet
		msgs = append(msgs, llm.Message{
			Role:         "user",
			Content:      seg.Content,
			CacheControl: cacheCtrl,
		})
	}
	return llm.ChatRequest{
		Model:       model,
		System:      system,
		Messages:    msgs,
		Temperature: 0,
		MaxTokens:   1024,
	}
}

func buildClient(provider, customBaseURL, cacheDir string, isJudge bool) (llm.Client, error) {
	var inner llm.Client
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "mock":
		if isJudge {
			inner = &llm.MockClient{Responder: mockJudgeResponder}
		} else {
			inner = &llm.MockClient{Responder: mockGenResponder}
		}
		return inner, nil
	case "deepseek":
		key := firstNonEmpty(os.Getenv("DEEPSEEK_API_KEY"), os.Getenv("OPENAI_API_KEY"))
		base := firstNonEmpty(customBaseURL, os.Getenv("DEEPSEEK_BASE_URL"), "https://api.deepseek.com/v1")
		if isJudge && os.Getenv("JUDGE_API_KEY") != "" {
			key = os.Getenv("JUDGE_API_KEY")
			base = firstNonEmpty(customBaseURL, os.Getenv("JUDGE_BASE_URL"), base)
		}
		if key == "" {
			return nil, fmt.Errorf("DEEPSEEK_API_KEY (or OPENAI_API_KEY) is required for provider=deepseek")
		}
		inner = llm.NewOpenAIClient(base, key)
	case "qcoder", "qwen":
		key := firstNonEmpty(os.Getenv("QCODER_API_KEY"), os.Getenv("DASHSCOPE_API_KEY"), os.Getenv("OPENAI_API_KEY"))
		base := firstNonEmpty(customBaseURL, os.Getenv("QCODER_BASE_URL"), os.Getenv("DASHSCOPE_BASE_URL"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
		if isJudge && os.Getenv("JUDGE_API_KEY") != "" {
			key = os.Getenv("JUDGE_API_KEY")
			base = firstNonEmpty(customBaseURL, os.Getenv("JUDGE_BASE_URL"), base)
		}
		if key == "" {
			return nil, fmt.Errorf("QCODER_API_KEY (or DASHSCOPE_API_KEY) is required for provider=qcoder")
		}
		inner = llm.NewOpenAIClient(base, key)
	case "openai":
		key := os.Getenv("OPENAI_API_KEY")
		base := firstNonEmpty(customBaseURL, os.Getenv("OPENAI_BASE_URL"))
		if isJudge && os.Getenv("JUDGE_API_KEY") != "" {
			key = os.Getenv("JUDGE_API_KEY")
			base = firstNonEmpty(customBaseURL, os.Getenv("JUDGE_BASE_URL"), base)
		}
		if key == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY (or JUDGE_API_KEY) is required for provider=openai")
		}
		inner = llm.NewOpenAIClient(base, key)
	case "anthropic":
		key := os.Getenv("ANTHROPIC_API_KEY")
		base := firstNonEmpty(customBaseURL, os.Getenv("ANTHROPIC_BASE_URL"))
		if isJudge && os.Getenv("JUDGE_API_KEY") != "" {
			key = os.Getenv("JUDGE_API_KEY")
		}
		if key == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY is required for provider=anthropic")
		}
		inner = llm.NewAnthropicClient(base, key)
	default:
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	return llm.NewCachedClient(inner, cacheDir), nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func mockGenResponder(req llm.ChatRequest) string {
	// Dry-run responder reflects retrieved working-set facts present in the prompt.
	var joined strings.Builder
	for _, m := range req.Messages {
		joined.WriteString(m.Content)
		joined.WriteByte('\n')
	}
	ctxText := joined.String()
	var notes []string
	if strings.Contains(ctxText, "离线") || strings.Contains(ctxText, "VPC") {
		notes = append(notes, "禁止公网下载，构建节点处于离线 VPC。")
	}
	if strings.Contains(ctxText, "只读") || strings.Contains(ctxText, "readOnlyRootFilesystem") {
		notes = append(notes, "由于 readOnlyRootFilesystem 只读根文件系统限制，请改为写入 /tmp。")
	}
	if strings.Contains(ctxText, "PostgreSQL 16") {
		notes = append(notes, "请使用 PostgreSQL 16 JSONB 语法 @> 查询。")
	}
	if strings.Contains(ctxText, "CGO_ENABLED=0") || strings.Contains(ctxText, "纯静态") {
		notes = append(notes, "不可切换为 mattn/go-sqlite3，受限于 CGO_ENABLED=0 与纯静态交叉编译。")
	}
	if strings.Contains(ctxText, "Redis Cluster") {
		notes = append(notes, "CROSSSLOT 报错源于跨槽位多键操作，请使用 Hash Tag {tenant}。")
	}
	if len(notes) == 0 {
		return "根据当前上下文，暂无命中特定约束，建议补充核查。"
	}
	return strings.Join(notes, " ")
}

func mockJudgeResponder(req llm.ChatRequest) string {
	// Dry-run judge returns structured JSON with exact substring quotes from the candidate answer.
	if len(req.Messages) == 0 {
		return `{"items":[]}`
	}
	prompt := req.Messages[0].Content
	if strings.Contains(prompt, "Candidate 1:") {
		return `{"winner":"TIE","reason":"mock dry-run"}`
	}
	const ansMarker = "Candidate Answer:\n"
	const rubMarker = "\n\nAtomic Rubric Items:\n"
	aStart := strings.Index(prompt, ansMarker)
	rStart := strings.Index(prompt, rubMarker)
	if aStart < 0 || rStart < 0 {
		return `{"items":[]}`
	}
	ans := strings.TrimSpace(prompt[aStart+len(ansMarker) : rStart])
	rubBlock := strings.TrimSpace(prompt[rStart+len(rubMarker):])
	lines := strings.Split(rubBlock, "\n")
	type item struct {
		RubricIndex   int    `json:"rubric_index"`
		Satisfied     bool   `json:"satisfied"`
		EvidenceQuote string `json:"evidence_quote"`
		Reason        string `json:"reason"`
	}
	var items []item
	for i := range lines {
		ok := len(ans) > 10 && !strings.Contains(ans, "暂无命中特定约束")
		quote := ""
		if ok {
			quote = ans
			if len([]rune(quote)) > 48 {
				quote = string([]rune(quote)[:48])
			}
		}
		items = append(items, item{
			RubricIndex:   i,
			Satisfied:     ok,
			EvidenceQuote: quote,
			Reason:        "mock quote check",
		})
	}
	out, _ := json.Marshal(map[string]any{"items": items})
	return string(out)
}

func opaqueRunID(strat, caseID, probeID string) string {
	sum := sha256.Sum256([]byte(strat + "|" + caseID + "|" + probeID))
	return "run_" + hex.EncodeToString(sum[:6])
}

func selectStrategies(spec string, cfg strategy.Config) []strategy.ContextStrategy {
	all := []strategy.ContextStrategy{
		strategy.NewB0Full(cfg),
		strategy.NewB1GPCurrent(cfg),
		strategy.NewB2NaiveRAG(cfg),
		strategy.NewE1MMR(cfg),
		strategy.NewE2PPR(cfg),
		strategy.NewE3(cfg),
		strategy.NewE4A(cfg),
		strategy.NewE4B(cfg),
	}
	want := make(map[string]bool)
	for _, p := range strings.Split(spec, ",") {
		want[strings.ToLower(strings.TrimSpace(p))] = true
	}
	var out []strategy.ContextStrategy
	for _, s := range all {
		if want["all"] || want[strings.ToLower(s.Name())] {
			out = append(out, s)
		}
	}
	return out
}

func printE2ESummary(reports []strategyE2EReport, passVectors map[string][]bool) {
	fmt.Println("========================================================================================================")
	fmt.Printf("%-10s | %-8s | %-8s | %-8s | %-8s | %-9s | %-8s | %-22s\n",
		"Strategy", "PassRate", "Rubric", "Recall", "Contam", "CacheRead", "Utility", "Quadrants (Q1/Q2/Q3/Q4)")
	fmt.Println("--------------------------------------------------------------------------------------------------------")
	for _, r := range reports {
		quadStr := fmt.Sprintf("%d/%d/%d/%d",
			r.Quadrants[string(judge.Q1Ideal)],
			r.Quadrants[string(judge.Q2SelectorFailure)],
			r.Quadrants[string(judge.Q3PromptLayoutFailure)],
			r.Quadrants[string(judge.Q4WeakProbe)],
		)
		fmt.Printf("%-10s | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %8.1f%% | %8.3f | %-22s\n",
			r.Strategy,
			r.PassRate*100,
			r.MeanRubricScore*100,
			r.MeanRecall*100,
			r.ContamRate*100,
			r.CacheReadRatio*100,
			r.Utility,
			quadStr,
		)
	}
	fmt.Println("========================================================================================================")

	// McNemar + Bootstrap significance tests for E4-A vs B1 / B2 when present.
	if vE4, ok := passVectors["E4-A"]; ok {
		for _, base := range []string{"B1", "B2"} {
			if vBase, okB := passVectors[base]; okB {
				b, c, p := judge.McNemarTest(vE4, vBase)
				diff, lo, hi := judge.BootstrapDiffCI(vE4, vBase, 1000, 42)
				fmt.Printf("Significance E4-A vs %-3s: McNemar(b=%d, c=%d, p=%.4f) | Bootstrap ΔPass=%.1f%% [95%% CI: %.1f%%, %.1f%%]\n",
					base, b, c, p, diff*100, lo*100, hi*100)
			}
		}
	}
}

func runPairwise(ctx context.Context, j *judge.AtomicJudge, cases []dataset.Case, sA, sB string, mapA, mapB map[string]string) {
	winsA, winsB, ties := 0, 0, 0
	for _, c := range cases {
		for _, p := range c.EvalProbes {
			key := c.CaseID + "/" + p.ProbeID
			ansA, okA := mapA[key]
			ansB, okB := mapB[key]
			if !okA || !okB {
				continue
			}
			out, err := j.ComparePairwise(ctx, p.Query, p.ExpectedRubric, ansA, ansB)
			if err != nil {
				continue
			}
			switch out.Winner {
			case "A":
				winsA++
			case "B":
				winsB++
			default:
				ties++
			}
		}
	}
	fmt.Printf("Symmetric Position-Swap Pairwise (%s vs %s): %s=%d, %s=%d, TIE=%d\n",
		sA, sB, sA, winsA, sB, winsB, ties)
}
