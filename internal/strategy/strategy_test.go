package strategy_test

import (
	"strings"
	"testing"

	"github.com/npc1607/gp-context-lab/internal/cache"
	"github.com/npc1607/gp-context-lab/internal/dataset"
	"github.com/npc1607/gp-context-lab/internal/index"
	"github.com/npc1607/gp-context-lab/internal/strategy"
)

func TestE4AGoldenPromptPrefixStabilityWithinEpoch(t *testing.T) {
	cfg := strategy.DefaultConfig()
	cfg.Epoch = 8
	cfg.RecentWindow = 4
	cfg.BaseToolsTokens = 1024
	cfg.ExtractorMode = index.ModeOracle

	s := strategy.NewE4A(cfg)
	s.Reset("You are gp.", 1200)

	sim := cache.NewSimulator(1024)

	// Add initial turns and verify append-only prefix stability across continuation turns inside Epoch 0.
	turns := []dataset.Turn{
		{TurnID: 1, EntryID: "e1", Role: "user", Content: "离线 VPC 禁止公网访问", ConstraintEvent: &dataset.ConstraintEvent{Action: "add", ConstraintID: "c1", Statement: "离线 VPC 禁止公网"}},
		{TurnID: 2, EntryID: "e2", Role: "assistant", Content: "收到。"},
		{TurnID: 3, EntryID: "e3", Role: "assistant", Content: "", ToolCalls: []dataset.ToolCall{{ID: "c1", Name: "read", Arguments: "{}"}}},
		{TurnID: 4, EntryID: "e4", Role: "tool", ToolCallID: "c1", Content: "FROM internal-registry/go:1.23"},
		{TurnID: 5, EntryID: "e5", Role: "assistant", Content: "读取完成。"},
		{TurnID: 6, EntryID: "e6", Role: "user", Content: "继续检查构建脚本。"},
	}
	for _, tr := range turns {
		if err := s.OnTurn(tr); err != nil {
			t.Fatal(err)
		}
	}

	asm1, _, err := s.BuildPrompt("continue", 6)
	if err != nil {
		t.Fatal(err)
	}
	tok1 := asm1.FlattenTokens()
	st1 := sim.Step(tok1)
	if st1.TotalInputTokens < 1024 {
		t.Fatalf("expected base tokens >= 1024, got %d", st1.TotalInputTokens)
	}

	// Verify canonical golden segment order: System -> Ledger -> WorkingSet -> RecentWindow -> Query.
	var kinds []strategy.SegmentKind
	for _, seg := range asm1.Segments {
		kinds = append(kinds, seg.Kind)
	}
	if len(kinds) < 4 || kinds[0] != strategy.SegSystem || kinds[1] != strategy.SegLedger || kinds[len(kinds)-2] != strategy.SegRecentWindow || kinds[len(kinds)-1] != strategy.SegQuery {
		t.Fatalf("unexpected E4-A segment ordering: %v", kinds)
	}

	// Turn 7 stays within the same Epoch (Epoch 0): full previous prompt prefix up to Query must hit cache!
	if err := s.OnTurn(dataset.Turn{TurnID: 7, EntryID: "e7", Role: "tool", ToolCallID: "c2", Content: "docker build -f Dockerfile ."}); err != nil {
		t.Fatal(err)
	}
	asm2, _, err := s.BuildPrompt("continue", 7)
	if err != nil {
		t.Fatal(err)
	}
	tok2 := asm2.FlattenTokens()
	st2 := sim.Step(tok2)
	if st2.CacheReadTokens < 1024 {
		t.Fatalf("expected E4-A within-epoch continuation to hit prefix cache >= 1024, got %+v", st2)
	}
}

func TestHonestB1PreservesFullLinesAndMutatesSlidingPrefix(t *testing.T) {
	cfg := strategy.DefaultConfig()
	cfg.BaseToolsTokens = 1024

	b1 := strategy.NewB1GPCurrent(cfg)
	b1.Reset("You are gp.", 400)

	bigToolOut := strings.Repeat("log line payload check ok\n", 25)
	turns := []dataset.Turn{
		{TurnID: 1, EntryID: "e1", Role: "user", Content: "第一条完整长决策：必须保证在十个以上汉字之后的核心约束语句也不被截断丢失。"},
		{TurnID: 2, EntryID: "e2", Role: "tool", ToolCallID: "tc1", Content: bigToolOut},
		{TurnID: 3, EntryID: "e3", Role: "assistant", Content: "收到。"},
		{TurnID: 4, EntryID: "e4", Role: "user", Content: "第 4 轮"},
		{TurnID: 5, EntryID: "e5", Role: "user", Content: "第 5 轮"},
		{TurnID: 6, EntryID: "e6", Role: "user", Content: "第 6 轮"},
		{TurnID: 7, EntryID: "e7", Role: "user", Content: "第 7 轮"},
	}
	for _, tr := range turns {
		if err := b1.OnTurn(tr); err != nil {
			t.Fatal(err)
		}
	}
	asm, _, err := b1.BuildPrompt("query", 7)
	if err != nil {
		t.Fatal(err)
	}
	joined := ""
	for _, seg := range asm.Segments {
		joined += seg.Content
	}
	if !strings.Contains(joined, "不被截断丢失") {
		t.Fatalf("expected honest B1 compaction to preserve full user constraint text, got:\n%s", joined)
	}
}

func TestE4AWorkingSetBudgetFloorPreventsStarvation(t *testing.T) {
	cfg := strategy.DefaultConfig()
	cfg.Epoch = 16
	cfg.RecentWindow = 4
	cfg.WorkingSetFloorRatio = 0.65

	s := strategy.NewE4A(cfg)
	s.Reset("You are gp.", 1000)

	// Turn 1: early critical decision that must be recalled.
	if err := s.OnTurn(dataset.Turn{
		TurnID: 1, EntryID: "e_early", Role: "user",
		Content: "critical constraint: proto slot 12 is permanently frozen in audit_meta.proto",
	}); err != nil {
		t.Fatal(err)
	}
	// Turns 2..20: fill Epoch 1 with large tool outputs so recentTokens exceeds 900.
	bigTool := strings.Repeat("verbose routine output line from check_runner\n", 18)
	for tid := 2; tid <= 20; tid++ {
		if err := s.OnTurn(dataset.Turn{
			TurnID: tid, EntryID: "e_pad_" + string(rune('a'+tid-2)), Role: "tool",
			ToolCallID: "c", Content: bigTool,
		}); err != nil {
			t.Fatal(err)
		}
	}

	asm, _, err := s.BuildPrompt("which proto slot is frozen in audit_meta.proto?", 20)
	if err != nil {
		t.Fatal(err)
	}
	if asm.WorkingSetTokens == 0 {
		t.Fatalf("expected non-zero WorkingSetTokens under budget floor even when recent window is full, got 0")
	}
	foundEarly := false
	for _, id := range asm.SelectedEntryIDs {
		if id == "e_early" {
			foundEarly = true
			break
		}
	}
	if !foundEarly {
		t.Fatalf("expected e_early to be recalled with WorkingSet budget floor, got %v", asm.SelectedEntryIDs)
	}
}
