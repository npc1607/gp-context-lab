package judge

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/npc1607/gp-context-lab/internal/llm"
)

// RubricItemResult captures the verification status of one atomic rubric criterion.
type RubricItemResult struct {
	RubricIndex   int    `json:"rubric_index"`
	Satisfied     bool   `json:"satisfied"`
	EvidenceQuote string `json:"evidence_quote"`
	QuoteVerified bool   `json:"quote_verified"`
	Reason        string `json:"reason"`
}

// Layer3Result captures Atomic Rubric Judge scoring for a single probe response.
type Layer3Result struct {
	Items       []RubricItemResult `json:"items"`
	RubricScore float64            `json:"rubric_score"`
	RawOutput   string             `json:"raw_output,omitempty"`
	Usage       llm.Usage          `json:"usage"`
}

// PairwiseOutcome records symmetric A/B position-swapped comparison results.
type PairwiseOutcome struct {
	Winner        string `json:"winner"` // "A" | "B" | "TIE"
	ForwardWinner string `json:"forward_winner"`
	SwappedWinner string `json:"swapped_winner"`
	Agreement     bool   `json:"agreement"`
}

// AtomicJudge evaluates answers via anonymous, quote-grounded atomic rubric verification.
type AtomicJudge struct {
	Client llm.Client
	Model  string
}

func NewAtomicJudge(client llm.Client, model string) *AtomicJudge {
	return &AtomicJudge{Client: client, Model: model}
}

// JudgeProbe sends an anonymous run_id + answer + rubric checklist to the judge LLM at temperature=0,
// then verifies every positive claim is backed by a verbatim quote present in answer.
func (j *AtomicJudge) JudgeProbe(ctx context.Context, runID, query, answer string, rubric []string) (Layer3Result, error) {
	if len(rubric) == 0 {
		return Layer3Result{RubricScore: 1.0}, nil
	}
	var rubricBuf strings.Builder
	for i, r := range rubric {
		rubricBuf.WriteString(fmt.Sprintf("%d. %s\n", i, r))
	}

	sysPrompt := `You are a strict, impartial software engineering evaluator.
Verify the candidate answer against each atomic rubric item independently.
Output ONLY valid JSON matching:
{"items":[{"rubric_index":0,"satisfied":true,"evidence_quote":"exact verbatim substring from candidate answer or null","reason":"<= 30 chars"}]}
Rules:
1. If satisfied is true, evidence_quote MUST be an exact verbatim substring copied from the candidate answer.
2. Do not infer unstated facts or give partial credit.`

	userPrompt := fmt.Sprintf("Run ID: %s\nQuestion:\n%s\n\nCandidate Answer:\n%s\n\nAtomic Rubric Items:\n%s",
		runID, query, answer, rubricBuf.String())

	resp, err := j.Client.Chat(ctx, llm.ChatRequest{
		Model:       j.Model,
		System:      sysPrompt,
		Messages:    []llm.Message{{Role: "user", Content: userPrompt}},
		Temperature: 0,
		MaxTokens:   1024,
	})
	if err != nil {
		return Layer3Result{}, err
	}

	cleaned := stripJSONCodeFence(resp.Content)
	var parsed struct {
		Items []RubricItemResult `json:"items"`
	}
	if err := json.Unmarshal([]byte(cleaned), &parsed); err != nil {
		return Layer3Result{RawOutput: resp.Content, Usage: resp.Usage}, fmt.Errorf("parse judge json: %w", err)
	}

	byIdx := make(map[int]RubricItemResult, len(parsed.Items))
	for _, it := range parsed.Items {
		if it.RubricIndex < 0 || it.RubricIndex >= len(rubric) {
			continue
		}
		if it.Satisfied {
			verified := VerifyQuoteGrounded(answer, it.EvidenceQuote)
			it.QuoteVerified = verified
			if !verified {
				// Enforce quote-grounded safeguard from docs/05 Section 4.3.
				it.Satisfied = false
				if it.Reason == "" {
					it.Reason = "missing verbatim quote"
				} else {
					it.Reason += " (quote unverified)"
				}
			}
		}
		byIdx[it.RubricIndex] = it
	}

	finalItems := make([]RubricItemResult, len(rubric))
	satisfiedCount := 0
	for i := range rubric {
		if it, ok := byIdx[i]; ok {
			finalItems[i] = it
			if it.Satisfied {
				satisfiedCount++
			}
		} else {
			finalItems[i] = RubricItemResult{RubricIndex: i, Satisfied: false, Reason: "missing from judge output"}
		}
	}

	return Layer3Result{
		Items:       finalItems,
		RubricScore: float64(satisfiedCount) / float64(len(rubric)),
		RawOutput:   resp.Content,
		Usage:       resp.Usage,
	}, nil
}

// ComparePairwise runs symmetric A/B evaluation ((A,B) and (B,A)) to cancel position bias.
func (j *AtomicJudge) ComparePairwise(ctx context.Context, query string, rubric []string, ansA, ansB string) (PairwiseOutcome, error) {
	fwd, err := j.judgePairOnce(ctx, query, rubric, ansA, ansB)
	if err != nil {
		return PairwiseOutcome{}, err
	}
	swpRaw, err := j.judgePairOnce(ctx, query, rubric, ansB, ansA)
	if err != nil {
		return PairwiseOutcome{}, err
	}
	// Translate swapped winner ("FIRST"/"SECOND") back to original A/B labels.
	swp := swpRaw
	switch swpRaw {
	case "A":
		swp = "B"
	case "B":
		swp = "A"
	}

	winner := "TIE"
	agree := fwd == swp
	if agree {
		winner = fwd
	}
	return PairwiseOutcome{
		Winner:        winner,
		ForwardWinner: fwd,
		SwappedWinner: swp,
		Agreement:     agree,
	}, nil
}

func (j *AtomicJudge) judgePairOnce(ctx context.Context, query string, rubric []string, first, second string) (string, error) {
	sysPrompt := `Compare Candidate 1 and Candidate 2 strictly against the rubric items.
Output ONLY JSON: {"winner":"FIRST"|"SECOND"|"TIE","reason":"short"}`
	userPrompt := fmt.Sprintf("Question:\n%s\n\nRubric:\n- %s\n\nCandidate 1:\n%s\n\nCandidate 2:\n%s",
		query, strings.Join(rubric, "\n- "), first, second)
	resp, err := j.Client.Chat(ctx, llm.ChatRequest{
		Model:       j.Model,
		System:      sysPrompt,
		Messages:    []llm.Message{{Role: "user", Content: userPrompt}},
		Temperature: 0,
		MaxTokens:   256,
	})
	if err != nil {
		return "", err
	}
	var parsed struct {
		Winner string `json:"winner"`
	}
	if err := json.Unmarshal([]byte(stripJSONCodeFence(resp.Content)), &parsed); err != nil {
		return "", fmt.Errorf("parse pairwise judge json: %w", err)
	}
	switch strings.ToUpper(strings.TrimSpace(parsed.Winner)) {
	case "FIRST":
		return "A", nil
	case "SECOND":
		return "B", nil
	default:
		return "TIE", nil
	}
}

// VerifyQuoteGrounded verifies that quote appears in answer after whitespace normalization.
func VerifyQuoteGrounded(answer, quote string) bool {
	nq := normalizeWS(quote)
	if nq == "" || strings.EqualFold(nq, "null") {
		return false
	}
	na := normalizeWS(answer)
	return strings.Contains(na, nq)
}

func normalizeWS(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inSpace := false
	for _, r := range strings.TrimSpace(s) {
		if unicode.IsSpace(r) {
			if !inSpace {
				b.WriteByte(' ')
				inSpace = true
			}
			continue
		}
		inSpace = false
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

func stripJSONCodeFence(s string) string {
	t := strings.TrimSpace(s)
	if strings.HasPrefix(t, "```") {
		if idx := strings.IndexByte(t, '\n'); idx >= 0 {
			t = t[idx+1:]
		}
		if end := strings.LastIndex(t, "```"); end >= 0 {
			t = t[:end]
		}
	}
	return strings.TrimSpace(t)
}
