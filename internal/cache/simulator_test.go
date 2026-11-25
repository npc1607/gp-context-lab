package cache_test

import (
	"testing"

	"github.com/npc1607/gp-context-lab/internal/cache"
)

func TestSimulatorLCPAndThreshold(t *testing.T) {
	sim := cache.NewSimulator(1024)

	base := make([]int, 1500)
	for i := range base {
		base[i] = i + 1
	}

	s1 := sim.Step(base)
	if s1.CacheReadTokens != 0 || s1.UncachedTokens != 1500 {
		t.Fatalf("unexpected first step stats: %+v", s1)
	}

	// Append 100 tokens while keeping prefix intact.
	req2 := append(append([]int(nil), base...), 9001, 9002)
	s2 := sim.Step(req2)
	if s2.LCPTokens != 1500 || s2.CacheReadTokens != 1500 || s2.UncachedTokens != 2 {
		t.Fatalf("expected 1500 cached prefix hit, got %+v", s2)
	}

	// Mutating token index 100 invalidates prefix beyond 100 (below 1024 threshold).
	req3 := append([]int(nil), req2...)
	req3[100] = 999999
	s3 := sim.Step(req3)
	if s3.LCPTokens != 100 || s3.CacheReadTokens != 0 {
		t.Fatalf("expected cache miss below 1024 threshold, got %+v", s3)
	}
}
