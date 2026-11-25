package algo

import (
	"bytes"
	"compress/zlib"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// NormalizeMinMax scales a non-negative score slice into [0, 1] by max element.
func NormalizeMinMax(scores []float64) []float64 {
	out := make([]float64, len(scores))
	var maxVal float64
	for _, v := range scores {
		if v > maxVal {
			maxVal = v
		}
	}
	if maxVal <= 0 {
		return out
	}
	inv := 1.0 / maxVal
	for i, v := range scores {
		if v > 0 {
			out[i] = v * inv
		}
	}
	return out
}

// WeightedRRF computes Reciprocal Rank Fusion scores across multiple ranked signal arrays.
func WeightedRRF(signals [][]float64, weights []float64, k float64) []float64 {
	if len(signals) == 0 {
		return nil
	}
	n := len(signals[0])
	if k <= 0 {
		k = 60.0
	}
	out := make([]float64, n)
	type idxScore struct {
		idx int
		val float64
	}
	buf := make([]idxScore, n)
	for sIdx, sig := range signals {
		if len(sig) != n {
			continue
		}
		w := 1.0
		if sIdx < len(weights) {
			w = weights[sIdx]
		}
		hasNonZero := false
		for i := 0; i < n; i++ {
			buf[i] = idxScore{idx: i, val: sig[i]}
			if sig[i] > 0 {
				hasNonZero = true
			}
		}
		if !hasNonZero {
			continue
		}
		sort.Slice(buf, func(i, j int) bool {
			return buf[i].val > buf[j].val
		})
		for rank, item := range buf {
			if item.val <= 0 {
				break
			}
			out[item.idx] += w / (k + float64(rank+1))
		}
	}
	return NormalizeMinMax(out)
}

// ZlibDensityPenalty computes rho(d_i) in (0.15, 1.0] to penalize repetitive logs,
// boilerplate acknowledgments, and ultra-long repetitive hex/UUID dumps.
func ZlibDensityPenalty(text string) float64 {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return 0.15
	}
	// Very short boilerplate acknowledgments without technical tokens get downweighted.
	runeLen := utf8.RuneCountInString(trimmed)
	if runeLen < 12 {
		return 0.45
	}

	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	_, _ = zw.Write([]byte(trimmed))
	_ = zw.Close()

	compRatio := float64(buf.Len()) / float64(len(trimmed))
	// Check unique token ratio to catch repetitive stack traces / UUID arrays.
	fields := strings.Fields(trimmed)
	uniq := make(map[string]struct{}, len(fields))
	for _, f := range fields {
		uniq[f] = struct{}{}
	}
	uniqRatio := 1.0
	if len(fields) > 0 {
		uniqRatio = float64(len(uniq)) / float64(len(fields))
	}

	// Penalize highly compressible boilerplate (low compRatio) and massive hex/UUID tables.
	rho := 0.5*math.Min(1.0, compRatio*1.35) + 0.5*math.Min(1.0, uniqRatio*1.2)
	if len(trimmed) > 1200 && uniqRatio < 0.55 {
		rho *= 0.45
	}
	if rho < 0.15 {
		rho = 0.15
	}
	if rho > 1.0 {
		rho = 1.0
	}
	return rho
}

// UnigramSurprisal approximates bare -log P entropy per token for negative control E5.
// High-entropy UUIDs, random hex addresses, and minified JS produce high scores here.
func UnigramSurprisal(text string) float64 {
	if text == "" {
		return 0
	}
	counts := make(map[rune]int)
	total := 0
	for _, r := range text {
		counts[r]++
		total++
	}
	if total == 0 {
		return 0
	}
	var h float64
	fTotal := float64(total)
	for _, c := range counts {
		p := float64(c) / fTotal
		h -= p * math.Log2(p)
	}
	// Also reward unique high-entropy tokens (hex/UUIDs) to faithfully mirror raw -log P trap.
	fields := strings.Fields(text)
	hexLike := 0
	for _, f := range fields {
		if len(f) >= 16 && strings.ContainsAny(f, "0123456789") && strings.ContainsAny(f, "abcdef-") {
			hexLike++
		}
	}
	return h + 0.25*float64(hexLike)
}
