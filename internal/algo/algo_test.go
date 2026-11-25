package algo_test

import (
	"math"
	"testing"

	"github.com/npc1607/gp-context-lab/internal/algo"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
)

func TestPPRMultiHopAndDegreePenalty(t *testing.T) {
	g := index.NewGraph()
	ext := index.HeuristicExtractor{}
	emb := index.NewHashNGramEmbedder(64)

	t1 := dataset.Turn{TurnID: 1, EntryID: "e1", Role: "user", Content: "forbidden public network access on build nodes; check deploy/Dockerfile"}
	t2 := dataset.Turn{TurnID: 2, EntryID: "e2", Role: "tool", Content: "FROM internal-registry.local/go:1.23 in deploy/Dockerfile"}
	t3 := dataset.Turn{TurnID: 3, EntryID: "e3", Role: "tool", Content: "scripts/build_image.sh invokes docker build -f deploy/Dockerfile"}
	t4 := dataset.Turn{TurnID: 4, EntryID: "e4", Role: "user", Content: "unrelated go.mod check"}

	for idx, turn := range []dataset.Turn{t1, t2, t3, t4} {
		g.AddTurn(idx, turn, ext.ExtractTurn(turn), emb.Embed(turn.RenderText()))
	}

	seed := g.BuildSeedDistribution(ext.ExtractQuery("fix scripts/build_image.sh 403"), []int{3})
	r := algo.PersonalizedPageRank(g, seed, 0.75, 20, 0.4)
	scores := g.ProjectMessageScores(r, 120.0)

	if scores[0] <= scores[3] {
		t.Fatalf("expected multihop entry e1 (%f) > unrelated e4 (%f)", scores[0], scores[3])
	}
	if scores[2] <= scores[3] {
		t.Fatalf("expected direct bridge e3 (%f) > unrelated e4 (%f)", scores[2], scores[3])
	}
}

func TestACTRSpacingEffect(t *testing.T) {
	tr := algo.NewACTRTracker(0.5, -1.0, 1.0)
	tr.EnsureTurn(0, 1) // reinforced repeatedly
	tr.RecordAccess(0, 20)
	tr.RecordAccess(0, 45)
	tr.RecordAccess(0, 70)

	tr.EnsureTurn(1, 5) // one-off mention

	wReinforced := tr.TimeWeight(0, 90)
	wOneOff := tr.TimeWeight(1, 90)
	if wReinforced <= wOneOff {
		t.Fatalf("expected reinforced weight (%f) > one-off weight (%f)", wReinforced, wOneOff)
	}
}

func TestBudgetedMMRReducesDuplicates(t *testing.T) {
	emb := index.NewHashNGramEmbedder(64)
	dupVec := emb.Embed("panic nil pointer dispatcher.go line 141")
	uniqVec := emb.Embed("worker concurrency limit is strictly 2")

	cands := []algo.Candidate{
		{TurnIdx: 0, EntryID: "dup1", Weight: 0.90, Tokens: 100, Vec: dupVec, Entities: map[string]struct{}{"panic": {}}},
		{TurnIdx: 1, EntryID: "dup2", Weight: 0.89, Tokens: 100, Vec: dupVec, Entities: map[string]struct{}{"panic": {}}},
		{TurnIdx: 2, EntryID: "uniq", Weight: 0.75, Tokens: 50, Vec: uniqVec, Entities: map[string]struct{}{"concurrency_2": {}}},
	}

	sel := algo.SelectBudgetedMMR(cands, 160, 0.6, 0.5)
	hasUniq := false
	for _, s := range sel {
		if s.EntryID == "uniq" {
			hasUniq = true
		}
	}
	if !hasUniq {
		t.Fatalf("expected MMR to select unique high-density entry within budget, got %+v", sel)
	}
}

func TestZlibDensityPenalizesRepeatedBoilerplate(t *testing.T) {
	noise := "panic: nil pointer 0xc0001234\n"
	longNoise := ""
	for i := 0; i < 60; i++ {
		longNoise += noise
	}
	conciseRule := "硬性限制：下游税务网关单商户并发上限固定为 2，超过会被拉黑 24 小时。"
	if algo.ZlibDensityPenalty(conciseRule) <= algo.ZlibDensityPenalty(longNoise) {
		t.Fatalf("expected concise rule density > repeated panic dump density")
	}
	if math.IsNaN(algo.UnigramSurprisal(longNoise)) {
		t.Fatalf("unexpected NaN surprisal")
	}
}
