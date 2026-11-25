package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Usage mirrors gp's ai.TokenUsage for real API cache accounting.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

// Message represents a single prompt segment/message sent to an LLM provider.
type Message struct {
	Role         string `json:"role"`
	Content      string `json:"content"`
	CacheControl bool   `json:"cache_control,omitempty"` // emits Anthropic ephemeral breakpoint when true
}

// ChatRequest encapsulates a provider-agnostic LLM completion call.
type ChatRequest struct {
	Model       string    `json:"model"`
	System      string    `json:"system,omitempty"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// ChatResponse returns generated text, token/cache usage, and wall-clock latency.
type ChatResponse struct {
	Content   string  `json:"content"`
	Usage     Usage   `json:"usage"`
	LatencyMs float64 `json:"latency_ms"`
	FromCache bool    `json:"from_cache"`
}

// Client abstracts real or mock LLM providers.
type Client interface {
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

// OpenAIClient calls OpenAI-compatible /v1/chat/completions endpoints.
type OpenAIClient struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

func NewOpenAIClient(baseURL, apiKey string) *OpenAIClient {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	return &OpenAIClient{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *OpenAIClient) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	type oaiMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	msgs := make([]oaiMsg, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, oaiMsg{Role: "system", Content: req.System})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, oaiMsg{Role: m.Role, Content: m.Content})
	}
	payload := map[string]any{
		"model":       req.Model,
		"messages":    msgs,
		"temperature": req.Temperature,
	}
	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return ChatResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	start := time.Now()
	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatResponse{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ChatResponse{}, fmt.Errorf("openai API %d: %s", resp.StatusCode, string(raw))
	}

	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens         int `json:"prompt_tokens"`
			CompletionTokens     int `json:"completion_tokens"`
			PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"` // DeepSeek prefix cache hit field
			PromptTokensDetails  struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return ChatResponse{}, fmt.Errorf("parse openai response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return ChatResponse{}, fmt.Errorf("openai response had no choices")
	}
	cached := parsed.Usage.PromptTokensDetails.CachedTokens
	if cached == 0 && parsed.Usage.PromptCacheHitTokens > 0 {
		cached = parsed.Usage.PromptCacheHitTokens
	}
	return ChatResponse{
		Content: parsed.Choices[0].Message.Content,
		Usage: Usage{
			InputTokens:     parsed.Usage.PromptTokens,
			OutputTokens:    parsed.Usage.CompletionTokens,
			CacheReadTokens: cached,
		},
		LatencyMs: float64(time.Since(start).Microseconds()) / 1000.0,
	}, nil
}

// AnthropicClient calls /v1/messages with cache_control ephemeral breakpoints on stable prefix segments.
type AnthropicClient struct {
	BaseURL    string
	APIKey     string
	Version    string
	HTTPClient *http.Client
}

func NewAnthropicClient(baseURL, apiKey string) *AnthropicClient {
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}
	return &AnthropicClient{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		Version:    "2023-06-01",
		HTTPClient: &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *AnthropicClient) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	type textBlock struct {
		Type         string            `json:"type"`
		Text         string            `json:"text"`
		CacheControl map[string]string `json:"cache_control,omitempty"`
	}
	type anthMsg struct {
		Role    string      `json:"role"`
		Content []textBlock `json:"content"`
	}

	msgs := make([]anthMsg, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := m.Role
		if role == "system" || role == "tool" {
			role = "user"
		}
		blk := textBlock{Type: "text", Text: m.Content}
		if m.CacheControl {
			blk.CacheControl = map[string]string{"type": "ephemeral"}
		}
		msgs = append(msgs, anthMsg{Role: role, Content: []textBlock{blk}})
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = 1024
	}
	payload := map[string]any{
		"model":       req.Model,
		"max_tokens":  maxTok,
		"temperature": req.Temperature,
		"messages":    msgs,
	}
	if req.System != "" {
		payload["system"] = []textBlock{{
			Type:         "text",
			Text:         req.System,
			CacheControl: map[string]string{"type": "ephemeral"},
		}}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return ChatResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.APIKey)
	httpReq.Header.Set("anthropic-version", c.Version)

	start := time.Now()
	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatResponse{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ChatResponse{}, fmt.Errorf("anthropic API %d: %s", resp.StatusCode, string(raw))
	}

	var parsed struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return ChatResponse{}, fmt.Errorf("parse anthropic response: %w", err)
	}
	var out strings.Builder
	for _, b := range parsed.Content {
		if b.Type == "text" {
			out.WriteString(b.Text)
		}
	}
	return ChatResponse{
		Content: out.String(),
		Usage: Usage{
			InputTokens:      parsed.Usage.InputTokens,
			OutputTokens:     parsed.Usage.OutputTokens,
			CacheReadTokens:  parsed.Usage.CacheReadInputTokens,
			CacheWriteTokens: parsed.Usage.CacheCreationInputTokens,
		},
		LatencyMs: float64(time.Since(start).Microseconds()) / 1000.0,
	}, nil
}

// CachedClient wraps an underlying Client with deterministic SHA256 response caching on disk.
type CachedClient struct {
	Inner    Client
	CacheDir string
}

func NewCachedClient(inner Client, cacheDir string) *CachedClient {
	return &CachedClient{Inner: inner, CacheDir: cacheDir}
}

func (c *CachedClient) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if c.CacheDir == "" {
		return c.Inner.Chat(ctx, req)
	}
	keyBytes, err := json.Marshal(req)
	if err != nil {
		return c.Inner.Chat(ctx, req)
	}
	sum := sha256.Sum256(keyBytes)
	path := filepath.Join(c.CacheDir, hex.EncodeToString(sum[:])+".json")

	if raw, err := os.ReadFile(path); err == nil {
		var cached ChatResponse
		if json.Unmarshal(raw, &cached) == nil {
			cached.FromCache = true
			return cached, nil
		}
	}

	resp, err := c.Inner.Chat(ctx, req)
	if err != nil {
		return ChatResponse{}, err
	}
	if err := os.MkdirAll(c.CacheDir, 0o755); err == nil {
		if data, err := json.MarshalIndent(resp, "", "  "); err == nil {
			if tmp, err := os.CreateTemp(c.CacheDir, ".tmp-*"); err == nil {
				tmpName := tmp.Name()
				_, wErr := tmp.Write(data)
				cErr := tmp.Close()
				if wErr == nil && cErr == nil {
					_ = os.Rename(tmpName, path)
				} else {
					_ = os.Remove(tmpName)
				}
			}
		}
	}
	return resp, nil
}

// MockClient provides deterministic offline responses for dry-run pipelines and unit tests.
type MockClient struct {
	Responder func(req ChatRequest) string
}

func (m *MockClient) Chat(_ context.Context, req ChatRequest) (ChatResponse, error) {
	text := "mock response"
	if m.Responder != nil {
		text = m.Responder(req)
	}
	return ChatResponse{
		Content: text,
		Usage: Usage{
			InputTokens:  len(req.System)/4 + 128,
			OutputTokens: len(text)/4 + 16,
		},
		LatencyMs: 1.0,
	}, nil
}
