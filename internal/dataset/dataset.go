package dataset

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SuiteType identifies the benchmark suite category defined in case.schema.json.
type SuiteType string

const (
	SuiteMultihop SuiteType = "suite_a_multihop"
	SuiteConflict SuiteType = "suite_b_conflict"
	SuiteNoise    SuiteType = "suite_c_noise"
	SuiteReplay   SuiteType = "suite_d_replay"
)

// ToolCall represents a tool invocation emitted by an assistant turn.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ConstraintEvent represents ground-truth constraint lifecycle annotations on a turn.
type ConstraintEvent struct {
	Action       string `json:"action"` // add | supersede | resolve
	ConstraintID string `json:"constraint_id"`
	Supersedes   string `json:"supersedes,omitempty"`
	Statement    string `json:"statement"`
}

// Turn represents a single session history entry.
type Turn struct {
	TurnID            int              `json:"turn_id"`
	EntryID           string           `json:"entry_id"`
	Role              string           `json:"role"` // user | assistant | tool
	Content           string           `json:"content"`
	ToolCalls         []ToolCall       `json:"tool_calls,omitempty"`
	ToolCallID        string           `json:"tool_call_id,omitempty"`
	MentionedEntities []string         `json:"mentioned_entities,omitempty"`
	ConstraintEvent   *ConstraintEvent `json:"constraint_event,omitempty"`
}

// RenderText returns the full searchable text of a turn including tool call arguments.
func (t Turn) RenderText() string {
	var b strings.Builder
	b.WriteString(t.Content)
	for _, tc := range t.ToolCalls {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(tc.Name)
		b.WriteByte(' ')
		b.WriteString(tc.Arguments)
	}
	return b.String()
}

// DeterministicChecks configures Layer 2 regex assertions for end-to-end LLM evaluation.
type DeterministicChecks struct {
	MustMatchAnyRegex []string `json:"must_match_any_regex,omitempty"`
	MustNotMatchRegex []string `json:"must_not_match_regex,omitempty"`
}

// EvalProbe defines an evaluation query anchored at a specific turn index.
type EvalProbe struct {
	ProbeID                     string               `json:"probe_id"`
	AtTurn                      int                  `json:"at_turn"`
	Query                       string               `json:"query"`
	RequiredEvidenceEntryIDs    []string             `json:"required_evidence_entry_ids"`
	ForbiddenSupersededEntryIDs []string             `json:"forbidden_superseded_entry_ids,omitempty"`
	ExpectedRubric              []string             `json:"expected_rubric"`
	DeterministicChecks         *DeterministicChecks `json:"deterministic_checks,omitempty"`
}

// Case represents a single benchmark case conforming to case.schema.json.
type Case struct {
	CaseID       string      `json:"case_id"`
	Suite        SuiteType   `json:"suite"`
	Title        string      `json:"title"`
	Description  string      `json:"description"`
	SystemPrompt string      `json:"system_prompt"`
	TokenBudget  int         `json:"token_budget"`
	Turns        []Turn      `json:"turns"`
	EvalProbes   []EvalProbe `json:"eval_probes"`
}

// LoadFile reads and validates a single benchmark case JSON file.
func LoadFile(path string) (Case, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Case{}, fmt.Errorf("read %s: %w", path, err)
	}
	var c Case
	if err := json.Unmarshal(raw, &c); err != nil {
		return Case{}, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := c.Validate(); err != nil {
		return Case{}, fmt.Errorf("validate %s: %w", path, err)
	}
	if c.TokenBudget <= 0 {
		c.TokenBudget = 8000
	}
	return c, nil
}

// LoadDir loads all benchmark case JSON files from a directory, excluding schema files.
func LoadDir(dir string, includeSample bool) ([]Case, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir %s: %w", dir, err)
	}
	var paths []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".json") || name == "case.schema.json" {
			continue
		}
		if !includeSample && name == "sample-multihop-case.json" {
			continue
		}
		paths = append(paths, filepath.Join(dir, name))
	}
	sort.Strings(paths)

	cases := make([]Case, 0, len(paths))
	for _, p := range paths {
		c, err := LoadFile(p)
		if err != nil {
			return nil, err
		}
		cases = append(cases, c)
	}
	return cases, nil
}

// Validate checks structural invariants required by the offline runner.
func (c Case) Validate() error {
	if c.CaseID == "" {
		return fmt.Errorf("missing case_id")
	}
	if len(c.Turns) == 0 {
		return fmt.Errorf("turns must be non-empty")
	}
	if len(c.EvalProbes) == 0 {
		return fmt.Errorf("eval_probes must be non-empty")
	}
	seenEntries := make(map[string]struct{}, len(c.Turns))
	for i, t := range c.Turns {
		if t.EntryID == "" {
			return fmt.Errorf("turn[%d] missing entry_id", i)
		}
		if _, dup := seenEntries[t.EntryID]; dup {
			return fmt.Errorf("duplicate entry_id %q", t.EntryID)
		}
		seenEntries[t.EntryID] = struct{}{}
	}
	return nil
}
