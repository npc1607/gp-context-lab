package judge

import (
	"fmt"
	"regexp"

	"github.com/npc1607/gp-context-lab/internal/dataset"
)

// Layer2Result records deterministic hard-fail regex gate outcomes.
type Layer2Result struct {
	Passed         bool     `json:"passed"`
	FailureReasons []string `json:"failure_reasons,omitempty"`
}

// EvalLayer2 runs Layer 2 deterministic assertions from probe.DeterministicChecks.
// Any forbidden match or missing required match produces an immediate Hard Fail.
func EvalLayer2(answer string, checks *dataset.DeterministicChecks) Layer2Result {
	if checks == nil {
		return Layer2Result{Passed: true}
	}
	var reasons []string

	if len(checks.MustMatchAnyRegex) > 0 {
		matchedAny := false
		for _, pat := range checks.MustMatchAnyRegex {
			re, err := regexp.Compile(pat)
			if err != nil {
				reasons = append(reasons, fmt.Sprintf("invalid must_match_any_regex %q: %v", pat, err))
				continue
			}
			if re.MatchString(answer) {
				matchedAny = true
				break
			}
		}
		if !matchedAny {
			reasons = append(reasons, fmt.Sprintf("answer matched none of required patterns: %v", checks.MustMatchAnyRegex))
		}
	}

	for _, pat := range checks.MustNotMatchRegex {
		re, err := regexp.Compile(pat)
		if err != nil {
			reasons = append(reasons, fmt.Sprintf("invalid must_not_match_regex %q: %v", pat, err))
			continue
		}
		if loc := re.FindString(answer); loc != "" {
			reasons = append(reasons, fmt.Sprintf("answer triggered forbidden pattern %q (matched %q)", pat, loc))
		}
	}

	return Layer2Result{
		Passed:         len(reasons) == 0,
		FailureReasons: reasons,
	}
}
