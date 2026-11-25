package llm_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/npc1607/gp-context-lab/internal/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestOpenAIClientParsesDeepSeekCacheHitTokens(t *testing.T) {
	c := llm.NewOpenAIClient("https://api.deepseek.com/v1", "test-key")
	c.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(`{
					"choices": [{"message": {"content": "ok"}}],
					"usage": {
						"prompt_tokens": 2000,
						"completion_tokens": 50,
						"prompt_cache_hit_tokens": 1600
					}
				}`)),
			}, nil
		}),
	}

	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Model:    "deepseek-chat",
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Usage.InputTokens != 2000 || resp.Usage.CacheReadTokens != 1600 {
		t.Fatalf("expected Input=2000 CacheRead=1600, got %+v", resp.Usage)
	}
}
