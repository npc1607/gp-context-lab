package ledger

import (
	"fmt"
	"sort"
	"strings"
)

// Status represents a constraint lifecycle state.
type Status string

const (
	StatusActive     Status = "active"
	StatusSuperseded Status = "superseded"
	StatusResolved   Status = "resolved"
)

// Constraint represents a typed invariant/decision in L1 ContextLedger.
type Constraint struct {
	ID           string   `json:"id"`
	Category     string   `json:"category"`
	Statement    string   `json:"statement"`
	Status       Status   `json:"status"`
	Supersedes   []string `json:"supersedes,omitempty"`
	SupersededBy string   `json:"superseded_by,omitempty"`
	Provenance   []string `json:"provenance"`
	Turn         int      `json:"turn"`
}

// ContextLedger maintains structured low-frequency invariants with provenance pointers.
type ContextLedger struct {
	Version     int
	Constraints map[string]*Constraint
	Order       []string
}

// NewContextLedger initializes an empty structured constraint ledger.
func NewContextLedger() *ContextLedger {
	return &ContextLedger{
		Constraints: make(map[string]*Constraint),
	}
}

// Apply adds, supersedes, or resolves a constraint and tracks provenance entry IDs.
func (l *ContextLedger) Apply(action, id, category, statement, supersedesID, entryID string, turn int) {
	if id == "" || statement == "" {
		return
	}
	l.Version++
	switch action {
	case "resolve":
		if c, ok := l.Constraints[id]; ok {
			c.Status = StatusResolved
			c.Provenance = appendUnique(c.Provenance, entryID)
			c.Turn = turn
		}
	case "supersede":
		if supersedesID != "" {
			if old, ok := l.Constraints[supersedesID]; ok {
				old.Status = StatusSuperseded
				old.SupersededBy = id
			}
		}
		c := &Constraint{
			ID:         id,
			Category:   category,
			Statement:  statement,
			Status:     StatusActive,
			Provenance: []string{entryID},
			Turn:       turn,
		}
		if supersedesID != "" {
			c.Supersedes = []string{supersedesID}
		}
		if _, exists := l.Constraints[id]; !exists {
			l.Order = append(l.Order, id)
		}
		l.Constraints[id] = c
	default: // "add"
		if existing, exists := l.Constraints[id]; exists {
			existing.Statement = statement
			existing.Status = StatusActive
			existing.Provenance = appendUnique(existing.Provenance, entryID)
			existing.Turn = turn
			return
		}
		l.Order = append(l.Order, id)
		l.Constraints[id] = &Constraint{
			ID:         id,
			Category:   category,
			Statement:  statement,
			Status:     StatusActive,
			Provenance: []string{entryID},
			Turn:       turn,
		}
	}
}

// ActiveProvenanceEntryIDs returns entry IDs backing currently active constraints.
func (l *ContextLedger) ActiveProvenanceEntryIDs() []string {
	seen := make(map[string]struct{})
	var out []string
	for _, c := range l.sortedActive() {
		for _, eid := range c.Provenance {
			if _, ok := seen[eid]; !ok {
				seen[eid] = struct{}{}
				out = append(out, eid)
			}
		}
	}
	return out
}

// SupersededEntryIDs returns entry IDs that belong exclusively to superseded constraints.
func (l *ContextLedger) SupersededEntryIDs() map[string]struct{} {
	active := make(map[string]struct{})
	for _, eid := range l.ActiveProvenanceEntryIDs() {
		active[eid] = struct{}{}
	}
	sup := make(map[string]struct{})
	for _, c := range l.Constraints {
		if c.Status == StatusSuperseded {
			for _, eid := range c.Provenance {
				if _, isActive := active[eid]; !isActive {
					sup[eid] = struct{}{}
				}
			}
		}
	}
	return sup
}

// CanonicalRender produces a deterministic, byte-stable representation sorted by constraint ID
// so unchanged ledgers preserve Prompt Caching prefixes.
func (l *ContextLedger) CanonicalRender() string {
	active := l.sortedActive()
	if len(active) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[CONTEXT_LEDGER]\n")
	for _, c := range active {
		b.WriteString(fmt.Sprintf("- %s (%s): %s [provenance:%s]",
			c.ID, c.Category, c.Statement, strings.Join(c.Provenance, ",")))
		if len(c.Supersedes) > 0 {
			b.WriteString(fmt.Sprintf(" (supersedes:%s)", strings.Join(c.Supersedes, ",")))
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func (l *ContextLedger) sortedActive() []*Constraint {
	ids := make([]string, 0, len(l.Constraints))
	for id, c := range l.Constraints {
		if c.Status == StatusActive {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	out := make([]*Constraint, 0, len(ids))
	for _, id := range ids {
		out = append(out, l.Constraints[id])
	}
	return out
}

func appendUnique(list []string, v string) []string {
	if v == "" {
		return list
	}
	for _, x := range list {
		if x == v {
			return list
		}
	}
	return append(list, v)
}
