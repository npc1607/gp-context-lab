package index_test

import (
	"testing"

	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
)

func TestCodeTokenizeAndBM25ExactSymbolDistinction(t *testing.T) {
	bm := index.NewBM25()
	d0 := bm.AddDoc("handler for /v1/tasks returning legacy envelope and E1042")
	d1 := bm.AddDoc("handler for /v2/jobs returning ProblemDetails and E1043")

	scores := bm.Score("fix /v2/jobs E1043 handler")
	if scores[d1] <= scores[d0] {
		t.Fatalf("expected exact /v2/jobs E1043 doc (%f) > /v1/tasks doc (%f)", scores[d1], scores[d0])
	}
}

func TestHeuristicExtractorExtractsPathsAndCamelSymbols(t *testing.T) {
	ext := index.HeuristicExtractor{}
	ents := ext.ExtractTurn(dataset.Turn{
		TurnID:  1,
		EntryID: "e1",
		Role:    "tool",
		Content: "import \"analytics/audit_meta.proto\"\nfunc MapOrderExtra() { /* proto/order/v1/order.proto */ }",
	})
	want := map[string]bool{
		"analytics/audit_meta.proto": false,
		"proto/order/v1/order.proto": false,
		"maporderextra":              false,
	}
	for _, e := range ents {
		if _, ok := want[e]; ok {
			want[e] = true
		}
	}
	for k, seen := range want {
		if !seen {
			t.Fatalf("expected entity %q in %v", k, ents)
		}
	}
}
