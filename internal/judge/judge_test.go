package judge_test

import (
	"context"
	"testing"

	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/judge"
	"github.com/npc1607/gp-context-lab/internal/llm"
)

func TestEvalLayer2HardFailGate(t *testing.T) {
	checks := &dataset.DeterministicChecks{
		MustMatchAnyRegex: []string{`Bearer`, `/v2/jobs`},
		MustNotMatchRegex: []string{`X-API-Key`, `/v1/tasks`},
	}

	okRes := judge.EvalLayer2("Use Authorization: Bearer <jwt> on /v2/jobs.", checks)
	if !okRes.Passed {
		t.Fatalf("expected Layer 2 pass, got %+v", okRes)
	}

	badRes := judge.EvalLayer2("Use Authorization: Bearer with fallback X-API-Key.", checks)
	if badRes.Passed {
		t.Fatalf("expected Layer 2 hard fail on forbidden X-API-Key")
	}

	missRes := judge.EvalLayer2("Plain explanation without required anchor.", checks)
	if missRes.Passed {
		t.Fatalf("expected Layer 2 fail when must_match_any_regex not satisfied")
	}
}

func TestAtomicJudgeRejectsHallucinatedQuotes(t *testing.T) {
	mock := &llm.MockClient{
		Responder: func(req llm.ChatRequest) string {
			return `{"items":[
				{"rubric_index":0,"satisfied":true,"evidence_quote":"Use PostgreSQL 16 JSONB","reason":"quoted"},
				{"rubric_index":1,"satisfied":true,"evidence_quote":"fabricated non-existent sentence","reason":"hallucinated"}
			]}`
		},
	}
	j := judge.NewAtomicJudge(mock, "mock-judge")
	res, err := j.JudgeProbe(context.Background(), "run_123", "Write SQL", "Use PostgreSQL 16 JSONB operators.", []string{"item0", "item1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Items[0].Satisfied || !res.Items[0].QuoteVerified {
		t.Fatalf("expected item 0 verified and satisfied, got %+v", res.Items[0])
	}
	if res.Items[1].Satisfied || res.Items[1].QuoteVerified {
		t.Fatalf("expected item 1 invalidated due to hallucinated quote, got %+v", res.Items[1])
	}
	if res.RubricScore != 0.5 {
		t.Fatalf("expected rubric score 0.5, got %v", res.RubricScore)
	}
}

func TestPairwisePositionSwapRequiresAgreement(t *testing.T) {
	// Always picks FIRST -> position bias -> swapped disagreement -> TIE.
	biased := &llm.MockClient{
		Responder: func(req llm.ChatRequest) string {
			return `{"winner":"FIRST","reason":"position bias"}`
		},
	}
	j := judge.NewAtomicJudge(biased, "mock-judge")
	out, err := j.ComparePairwise(context.Background(), "q", []string{"r1"}, "ansA", "ansB")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Agreement || out.Winner != "TIE" {
		t.Fatalf("expected position-biased judge to result in disagreement TIE, got %+v", out)
	}
}

func TestQuadrantAndMcNemarBootstrap(t *testing.T) {
	if q := judge.ClassifyQuadrant(true, false); q != judge.Q3PromptLayoutFailure {
		t.Fatalf("expected Q3_PromptLayoutFailure, got %s", q)
	}
	if q := judge.ClassifyQuadrant(false, true); q != judge.Q4WeakProbe {
		t.Fatalf("expected Q4_WeakProbe, got %s", q)
	}

	a := []bool{true, true, true, true, true, true, true, true, true, true}
	b := []bool{false, false, false, false, false, false, false, false, true, true}
	bc, cc, p := judge.McNemarTest(a, b)
	if bc != 8 || cc != 0 || p >= 0.05 {
		t.Fatalf("expected significant McNemar p<0.05, got b=%d c=%d p=%f", bc, cc, p)
	}
	diff, lo, hi := judge.BootstrapDiffCI(a, b, 1000, 42)
	if diff <= 0.5 || lo <= 0 || hi > 1.0 {
		t.Fatalf("unexpected bootstrap CI: diff=%f lo=%f hi=%f", diff, lo, hi)
	}
}
