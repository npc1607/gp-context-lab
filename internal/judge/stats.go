package judge

import (
	"math"
	"math/rand"
)

// Quadrant classifies each probe into the 4-Quadrant Failure Attribution Matrix (docs/05 Section 5).
type Quadrant string

const (
	Q1Ideal               Quadrant = "Q1_Ideal"
	Q2SelectorFailure     Quadrant = "Q2_SelectorFailure"
	Q3PromptLayoutFailure Quadrant = "Q3_PromptLayoutFailure"
	Q4WeakProbe           Quadrant = "Q4_WeakProbe"
)

// ClassifyQuadrant maps Layer 1 retrieval success and Layer 2+3 Pass into a diagnostic quadrant.
func ClassifyQuadrant(retrievalSuccess, answerPass bool) Quadrant {
	switch {
	case retrievalSuccess && answerPass:
		return Q1Ideal
	case !retrievalSuccess && !answerPass:
		return Q2SelectorFailure
	case retrievalSuccess && !answerPass:
		return Q3PromptLayoutFailure
	default:
		return Q4WeakProbe
	}
}

// ComputeUtility computes the Pareto-gated utility score U = 0.55*PassRate + 0.25*MeanRecall - 0.20*(C/C_B0).
func ComputeUtility(passRate, meanRecall, cost, costB0 float64) float64 {
	costRatio := 1.0
	if costB0 > 0 {
		costRatio = cost / costB0
	}
	return 0.55*passRate + 0.25*meanRecall - 0.20*costRatio
}

// McNemarTest computes discordant counts (b: A pass & B fail, c: A fail & B pass) and two-sided p-value.
// Uses exact binomial test when b+c <= 25 and continuity-corrected chi-squared otherwise.
func McNemarTest(passA, passB []bool) (b, c int, pValue float64) {
	n := len(passA)
	if len(passB) < n {
		n = len(passB)
	}
	for i := 0; i < n; i++ {
		if passA[i] && !passB[i] {
			b++
		} else if !passA[i] && passB[i] {
			c++
		}
	}
	m := b + c
	if m == 0 {
		return 0, 0, 1.0
	}
	if m <= 25 {
		k := b
		if c < k {
			k = c
		}
		// Exact two-sided Binomial(m, 0.5): 2 * sum_{i=0..k} C(m, i) * 0.5^m
		var tail float64
		for i := 0; i <= k; i++ {
			tail += math.Exp(logChoose(m, i) - float64(m)*math.Ln2)
		}
		p := 2.0 * tail
		if p > 1.0 {
			p = 1.0
		}
		return b, c, p
	}
	diff := math.Abs(float64(b-c)) - 1.0
	if diff < 0 {
		diff = 0
	}
	chi2 := (diff * diff) / float64(m)
	// Survival function for chi-squared(df=1) = erfc(sqrt(chi2 / 2)).
	p := math.Erfc(math.Sqrt(chi2 / 2.0))
	return b, c, p
}

// BootstrapDiffCI computes mean PassRate(A) - PassRate(B) and its 95% confidence interval via deterministic bootstrap.
func BootstrapDiffCI(passA, passB []bool, iterations int, seed int64) (meanDiff, ciLow, ciHigh float64) {
	n := len(passA)
	if len(passB) < n {
		n = len(passB)
	}
	if n == 0 {
		return 0, 0, 0
	}
	if iterations <= 0 {
		iterations = 1000
	}
	rng := rand.New(rand.NewSource(seed))
	diffs := make([]float64, iterations)
	var sum float64
	for it := 0; it < iterations; it++ {
		var aHit, bHit int
		for i := 0; i < n; i++ {
			idx := rng.Intn(n)
			if passA[idx] {
				aHit++
			}
			if passB[idx] {
				bHit++
			}
		}
		d := float64(aHit-bHit) / float64(n)
		diffs[it] = d
		sum += d
	}
	sortFloat64s(diffs)
	loIdx := int(math.Floor(0.025 * float64(iterations-1)))
	hiIdx := int(math.Ceil(0.975 * float64(iterations-1)))
	return sum / float64(iterations), diffs[loIdx], diffs[hiIdx]
}

func logChoose(n, k int) float64 {
	ln, _ := math.Lgamma(float64(n + 1))
	lk, _ := math.Lgamma(float64(k + 1))
	lnk, _ := math.Lgamma(float64(n - k + 1))
	return ln - lk - lnk
}

func sortFloat64s(a []float64) {
	// Insertion/quick hybrid via standard simple sort for small/1000-element slices.
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}
