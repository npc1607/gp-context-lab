package cache

// TurnCacheStats captures per-request token counts, LCP cache hit metrics, and normalized USD-equivalent costs.
type TurnCacheStats struct {
	TotalInputTokens int
	LCPTokens        int
	CacheReadTokens  int
	CacheWriteTokens int
	UncachedTokens   int
	AnthropicCost    float64 // Normalized cost index (c_base = 1.0): uncached*1.0 + write*1.25 + read*0.10
	OpenAICost       float64 // Normalized cost index (c_base = 1.0): uncached*1.0 + read*0.25
}

// Simulator computes exact token-level Longest Common Prefix (LCP) across consecutive LLM requests.
type Simulator struct {
	MinCacheTokens int
	prevTokens     []int
	hasPrev        bool
}

// NewSimulator creates a Prompt Caching simulator with the specified minimum cacheable prefix length (default 1024).
func NewSimulator(minCacheTokens int) *Simulator {
	if minCacheTokens <= 0 {
		minCacheTokens = 1024
	}
	return &Simulator{MinCacheTokens: minCacheTokens}
}

// Reset clears previous request prefix state for a new conversation replay.
func (s *Simulator) Reset() {
	s.prevTokens = nil
	s.hasPrev = false
}

// Step evaluates a serialized prompt token sequence against the immediately preceding request.
func (s *Simulator) Step(tokens []int) TurnCacheStats {
	total := len(tokens)
	lcp := 0
	if s.hasPrev {
		limit := len(s.prevTokens)
		if total < limit {
			limit = total
		}
		for i := 0; i < limit; i++ {
			if s.prevTokens[i] != tokens[i] {
				break
			}
			lcp++
		}
	}

	readToks := 0
	if lcp >= s.MinCacheTokens {
		readToks = lcp
	}
	uncached := total - readToks

	// Anthropic-style cache write applies to the newly cacheable prefix extension beyond readToks.
	writeToks := 0
	if total >= s.MinCacheTokens && uncached > 0 {
		writeToks = uncached
	}

	anthropicCost := float64(uncached)*1.0 + float64(writeToks)*0.25 + float64(readToks)*0.10
	// Wait: uncached already charged 1.0x; write premium adds +0.25x (total 1.25x) on newly written cache tokens.
	// When writeToks == 0 (below threshold), cost is uncached*1.0.
	openAICost := float64(uncached)*1.0 + float64(readToks)*0.25

	s.prevTokens = append(s.prevTokens[:0], tokens...)
	s.hasPrev = true

	return TurnCacheStats{
		TotalInputTokens: total,
		LCPTokens:        lcp,
		CacheReadTokens:  readToks,
		CacheWriteTokens: writeToks,
		UncachedTokens:   uncached,
		AnthropicCost:    anthropicCost,
		OpenAICost:       openAICost,
	}
}
