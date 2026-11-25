package tokenizer

import (
	"hash/fnv"
	"strings"
	"unicode"
)

// Encode splits text into deterministic subword-like chunks and hashes each chunk to an integer ID.
// This provides reproducible token counts and exact token-level LCP matching with zero external dependencies.
func Encode(text string) []int {
	if text == "" {
		return nil
	}
	tokens := make([]int, 0, len(text)/3+1)
	var cur strings.Builder
	flush := func() {
		if cur.Len() == 0 {
			return
		}
		s := cur.String()
		cur.Reset()
		// Split long ASCII chunks (~4 chars per subword) to mirror BPE token granularity.
		if isASCIIAlnumString(s) && len(s) > 4 {
			for i := 0; i < len(s); i += 4 {
				end := i + 4
				if end > len(s) {
					end = len(s)
				}
				tokens = append(tokens, hashToken(s[i:end]))
			}
			return
		}
		tokens = append(tokens, hashToken(s))
	}

	for _, r := range text {
		switch {
		case r <= unicode.MaxASCII && (unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'):
			cur.WriteRune(r)
		case unicode.Is(unicode.Han, r):
			flush()
			tokens = append(tokens, int(r)|0x40000000)
		case unicode.IsSpace(r):
			flush()
		default:
			flush()
			tokens = append(tokens, hashToken(string(r)))
		}
	}
	flush()
	return tokens
}

// Count returns the estimated token length of text.
func Count(text string) int {
	return len(Encode(text))
}

// PadTokens builds a deterministic synthetic token sequence of length n (used to simulate base tool schemas).
func PadTokens(prefix string, n int) []int {
	if n <= 0 {
		return nil
	}
	out := make([]int, n)
	base := hashToken(prefix)
	for i := 0; i < n; i++ {
		out[i] = base ^ (i + 1)*2654435761
	}
	return out
}

func isASCIIAlnumString(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
			return false
		}
	}
	return true
}

func hashToken(s string) int {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	// Keep positive 31-bit non-CJK range.
	return int(h.Sum64() & 0x3fffffff)
}
