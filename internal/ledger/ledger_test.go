package ledger_test

import (
	"strings"
	"testing"

	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/ledger"
)

func TestContextLedgerLifecycleAndCanonicalOrder(t *testing.T) {
	l := ledger.NewContextLedger()
	l.Apply("add", "cst_b", "net", "offline VPC", "", "entry_002", 2)
	l.Apply("add", "cst_a", "db", "use MySQL 8.0", "", "entry_001", 1)

	render1 := l.CanonicalRender()
	// Deterministic sorted order: cst_a must appear before cst_b.
	idxA := strings.Index(render1, "cst_a")
	idxB := strings.Index(render1, "cst_b")
	if idxA < 0 || idxB < 0 || idxA > idxB {
		t.Fatalf("expected deterministic sorted IDs (cst_a before cst_b), got:\n%s", render1)
	}

	// Superseding cst_a moves entry_001 to SupersededEntryIDs and hides MySQL from active render.
	l.Apply("supersede", "cst_a2", "db", "use PostgreSQL 16", "cst_a", "entry_010", 10)
	render2 := l.CanonicalRender()
	if strings.Contains(render2, "use MySQL 8.0") {
		t.Fatalf("superseded statement leaked into canonical render:\n%s", render2)
	}
	if !strings.Contains(render2, "(supersedes:cst_a)") {
		t.Fatalf("expected minimal supersedes footnote in render:\n%s", render2)
	}
	sup := l.SupersededEntryIDs()
	if _, ok := sup["entry_001"]; !ok {
		t.Fatalf("expected entry_001 to be marked superseded")
	}

	// Resolving cst_b removes it from active render.
	l.Apply("resolve", "cst_b", "net", "offline VPC", "", "entry_012", 12)
	if strings.Contains(l.CanonicalRender(), "cst_b") {
		t.Fatalf("resolved constraint should not appear in active render")
	}
}

func TestSupersededEntryRetainedIfBackedByOtherActiveConstraint(t *testing.T) {
	l := ledger.NewContextLedger()
	l.Apply("add", "cst_mq", "mq", "Kafka", "", "entry_001", 1)
	l.Apply("add", "cst_codec", "codec", "JSON", "", "entry_001", 1)
	l.Apply("supersede", "cst_mq2", "mq", "NATS", "cst_mq", "entry_007", 7)

	sup := l.SupersededEntryIDs()
	if _, ok := sup["entry_001"]; ok {
		t.Fatalf("entry_001 still backs active cst_codec and should not be marked fully superseded yet")
	}

	l.Apply("supersede", "cst_codec2", "codec", "Protobuf", "cst_codec", "entry_013", 13)
	sup = l.SupersededEntryIDs()
	if _, ok := sup["entry_001"]; !ok {
		t.Fatalf("once both cst_mq and cst_codec are superseded, entry_001 must be superseded")
	}
}

func TestHeuristicExtractorHandlesExplicitVsImplicitChanges(t *testing.T) {
	ext := ledger.NewHeuristicExtractor()
	l := ledger.NewContextLedger()
	ext.ProcessTurn(dataset.Turn{TurnID: 1, EntryID: "e1", Role: "user", Content: "初期采用 MySQL 8.0"}, l)
	ext.ProcessTurn(dataset.Turn{TurnID: 5, EntryID: "e5", Role: "user", Content: "正式废除 MySQL，全面改用 PostgreSQL 16"}, l)
	if len(l.ActiveProvenanceEntryIDs()) != 1 || l.ActiveProvenanceEntryIDs()[0] != "e5" {
		t.Fatalf("expected explicit supersede to leave only e5 active, got %v", l.ActiveProvenanceEntryIDs())
	}
}
