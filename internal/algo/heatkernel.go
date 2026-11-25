package algo

import (
	"math"

	"github.com/npc1607/gp-context-lab/internal/index"
)

// HeatKernel approximates h(t) = exp(-t * L_sym) * q using scaling-and-squaring truncated Taylor steps
// over the degree-penalized symmetric normalized graph Laplacian.
func HeatKernel(g *index.Graph, seed []float64, t float64, order int, beta float64) []float64 {
	n := len(g.Kinds)
	if n == 0 || len(seed) != n {
		return nil
	}
	if t <= 0 {
		t = 3.0
	}
	if order <= 0 {
		order = 8
	}
	if beta < 0 {
		beta = 0.4
	}

	// Precompute penalized weighted degree D_i.
	degW := make([]float64, n)
	for u := 0; u < n; u++ {
		var s float64
		for _, e := range g.Adj[u] {
			s += penalizedEdgeWeight(g, u, e, beta)
		}
		degW[u] = s
	}
	invSqrtDeg := make([]float64, n)
	for i, d := range degW {
		if d > 0 {
			invSqrtDeg[i] = 1.0 / math.Sqrt(d)
		}
	}

	// Sub-step dt = t / steps keeps ||dt * L_sym|| small so truncated Taylor converges cleanly.
	steps := 4
	dt := t / float64(steps)

	cur := make([]float64, n)
	copy(cur, seed)
	term := make([]float64, n)
	next := make([]float64, n)
	tmp := make([]float64, n)

	for s := 0; s < steps; s++ {
		copy(term, cur)
		copy(next, cur)
		for m := 1; m <= order; m++ {
			// tmp = L_sym * term = term - D^{-1/2} W D^{-1/2} term
			for i := 0; i < n; i++ {
				tmp[i] = term[i]
			}
			for u := 0; u < n; u++ {
				if term[u] == 0 || invSqrtDeg[u] == 0 {
					continue
				}
				uScaled := term[u] * invSqrtDeg[u]
				for _, e := range g.Adj[u] {
					w := penalizedEdgeWeight(g, u, e, beta)
					tmp[e.To] -= invSqrtDeg[e.To] * w * uScaled
				}
			}
			scale := -dt / float64(m)
			for i := 0; i < n; i++ {
				term[i] = scale * tmp[i]
				next[i] += term[i]
			}
		}
		for i := 0; i < n; i++ {
			if next[i] < 0 {
				next[i] = 0
			}
			cur[i] = next[i]
		}
	}
	return cur
}

func penalizedEdgeWeight(g *index.Graph, u int, e index.Edge, beta float64) float64 {
	if beta == 0 {
		return e.Weight
	}
	// Symmetric penalty on entity hubs.
	pen := 1.0
	if g.Kinds[e.To] == index.NodeEntity && len(g.Adj[e.To]) > 1 {
		pen *= math.Pow(float64(len(g.Adj[e.To])), -beta*0.5)
	}
	if g.Kinds[u] == index.NodeEntity && len(g.Adj[u]) > 1 {
		pen *= math.Pow(float64(len(g.Adj[u])), -beta*0.5)
	}
	return e.Weight * pen
}
