package algo

import (
	"math"

	"github.com/npc1607/gp-context-lab/internal/index"
)

// SelectBudgetedKDPP performs greedy MAP inference over a quality-diversity kernel L_ij = q_i * <phi_i, phi_j> * q_j
// using incremental Cholesky decomposition (Chen et al., NeurIPS 2018) with token-cost normalization c_i^eta.
func SelectBudgetedKDPP(cands []Candidate, budget int, eta float64) []Candidate {
	n := len(cands)
	if budget <= 0 || n == 0 {
		return nil
	}
	if eta < 0 {
		eta = 0.5
	}

	// Quality q_i = max(eps, Weight_i) / c_i^(eta/2) so det(L_S) incorporates length efficiency.
	q := make([]float64, n)
	diag := make([]float64, n)
	for i, c := range cands {
		w := math.Max(1e-6, c.Weight)
		cost := math.Pow(math.Max(1.0, float64(c.Tokens)), eta*0.5)
		q[i] = w / cost
		diag[i] = q[i] * q[i]
	}

	used := make([]bool, n)
	selected := make([]Candidate, 0, 16)
	// Cholesky factor rows C[step][i] for orthogonalized residual updates.
	cRows := make([][]float64, 0, 16)
	rem := budget

	for {
		bestIdx := -1
		bestGain := 1e-12
		for i, c := range cands {
			if used[i] || c.Tokens <= 0 || c.Tokens > rem {
				continue
			}
			if diag[i] > bestGain {
				bestGain = diag[i]
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}

		chosen := cands[bestIdx]
		used[bestIdx] = true
		selected = append(selected, chosen)
		rem -= chosen.Tokens

		invSqrt := 1.0 / math.Sqrt(diag[bestIdx])
		row := make([]float64, n)
		for i, c := range cands {
			if used[i] || c.Tokens > rem {
				continue
			}
			lIJ := q[bestIdx] * math.Max(0, index.Cosine(chosen.Vec, c.Vec)) * q[i]
			dotPrev := 0.0
			for _, prevRow := range cRows {
				dotPrev += prevRow[bestIdx] * prevRow[i]
			}
			eVal := (lIJ - dotPrev) * invSqrt
			row[i] = eVal
			diag[i] -= eVal * eVal
			if diag[i] < 0 {
				diag[i] = 0
			}
		}
		cRows = append(cRows, row)
	}
	return selected
}
