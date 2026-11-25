package index

import (
	"math"

	"github.com/npc1607/gp-context-lab/internal/dataset"
)

// NodeKind distinguishes message nodes, entity nodes, and ledger rule nodes in the heterogeneous graph.
type NodeKind uint8

const (
	NodeMessage NodeKind = iota
	NodeEntity
	NodeRule
)

// Edge represents a weighted directed edge in the heterogeneous graph.
type Edge struct {
	To     int32
	Weight float64
}

// Graph is an incremental heterogeneous graph G = (V_msg ∪ V_ent ∪ V_rule, E, W).
type Graph struct {
	Kinds      []NodeKind
	Adj        [][]Edge
	MsgToNode  map[int]int32 // turnIdx -> nodeID
	NodeToMsg  map[int32]int // nodeID -> turnIdx
	EntToNode  map[string]int32
	CallToNode map[string]int32 // tool_call_id -> assistant nodeID
	MsgVecs    [][]float32      // normalized dense vectors indexed by turnIdx
	lastMsg    int32
}

// NewGraph initializes an empty incremental heterogeneous graph.
func NewGraph() *Graph {
	return &Graph{
		MsgToNode:  make(map[int]int32),
		NodeToMsg:  make(map[int32]int),
		EntToNode:  make(map[string]int32),
		CallToNode: make(map[string]int32),
		lastMsg:    -1,
	}
}

func (g *Graph) addNode(kind NodeKind) int32 {
	id := int32(len(g.Kinds))
	g.Kinds = append(g.Kinds, kind)
	g.Adj = append(g.Adj, nil)
	return id
}

func (g *Graph) addUndirectedEdge(u, v int32, w float64) {
	if u == v || w <= 0 {
		return
	}
	g.Adj[u] = append(g.Adj[u], Edge{To: v, Weight: w})
	g.Adj[v] = append(g.Adj[v], Edge{To: u, Weight: w})
}

func (g *Graph) ensureEntityNode(entity string) int32 {
	if id, ok := g.EntToNode[entity]; ok {
		return id
	}
	id := g.addNode(NodeEntity)
	g.EntToNode[entity] = id
	return id
}

// AddTurn incrementally inserts a message node, its entity mention edges, causal/tool edges,
// entity co-occurrence bridges, and semantic similarity bridge edges (>= 0.75).
func (g *Graph) AddTurn(turnIdx int, turn dataset.Turn, entities []string, vec []float32) int32 {
	msgNode := g.addNode(NodeMessage)
	g.MsgToNode[turnIdx] = msgNode
	g.NodeToMsg[msgNode] = turnIdx
	g.MsgVecs = append(g.MsgVecs, vec)

	// 1. Causal / adjacent turn chain edge (very light temporal continuity so entity bridges dominate).
	if g.lastMsg >= 0 {
		g.addUndirectedEdge(g.lastMsg, msgNode, 0.08)
	}
	g.lastMsg = msgNode

	// 2. Tool call <-> tool result edge.
	for _, tc := range turn.ToolCalls {
		if tc.ID != "" {
			g.CallToNode[tc.ID] = msgNode
		}
	}
	if turn.ToolCallID != "" {
		if callerNode, ok := g.CallToNode[turn.ToolCallID]; ok {
			g.addUndirectedEdge(callerNode, msgNode, 1.5)
		}
	}

	// 3. Entry <-> entity mention edges.
	entNodes := make([]int32, 0, len(entities))
	for _, e := range entities {
		en := g.ensureEntityNode(e)
		g.addUndirectedEdge(msgNode, en, 1.0)
		entNodes = append(entNodes, en)
	}

	// 4. Co-occurring file/symbol dependency edges within the same turn (e.g., import relations).
	if len(entNodes) > 1 && len(entNodes) <= 12 {
		for i := 0; i < len(entNodes); i++ {
			for j := i + 1; j < len(entNodes); j++ {
				g.addUndirectedEdge(entNodes[i], entNodes[j], 0.4)
			}
		}
	}

	// 5. High-similarity semantic bridge edges (cosine >= 0.75), capped to avoid dense blowup.
	if len(vec) > 0 {
		bridges := 0
		for prevIdx := turnIdx - 1; prevIdx >= 0 && bridges < 3; prevIdx-- {
			sim := Cosine(vec, g.MsgVecs[prevIdx])
			if sim >= 0.75 {
				g.addUndirectedEdge(msgNode, g.MsgToNode[prevIdx], sim*0.6)
				bridges++
			}
		}
	}
	return msgNode
}

// BuildSeedDistribution constructs a normalized personalization vector q over graph nodes
// from query entities and recent active messages.
func (g *Graph) BuildSeedDistribution(queryEntities []string, recentTurnIdxs []int) []float64 {
	n := len(g.Kinds)
	q := make([]float64, n)
	if n == 0 {
		return q
	}
	var sum float64
	matchedEntities := 0
	for _, e := range queryEntities {
		if id, ok := g.EntToNode[e]; ok {
			// Weight rarer entities slightly higher (IDF-like prior).
			deg := float64(len(g.Adj[id]))
			w := 1.0 / math.Sqrt(math.Max(1.0, deg))
			q[id] += w
			sum += w
			matchedEntities++
		}
	}
	// Only fall back to recency message seeds if query extracted no graph entities,
	// or add a very small prior so unrelated recent turns do not overshadow entity seeds.
	recencyScale := 0.02
	if matchedEntities == 0 {
		recencyScale = 0.25
	}
	for i, tidx := range recentTurnIdxs {
		if id, ok := g.MsgToNode[tidx]; ok {
			w := recencyScale / float64(i+1)
			q[id] += w
			sum += w
		}
	}
	if sum == 0 {
		return q
	}
	inv := 1.0 / sum
	for i := range q {
		q[i] *= inv
	}
	return q
}

// ProjectMessageScores extracts diffusion weights mapped to turn indices with log(1 + c*r) compression.
func (g *Graph) ProjectMessageScores(r []float64, logScale float64) []float64 {
	out := make([]float64, len(g.MsgVecs))
	if logScale <= 0 {
		logScale = 100.0
	}
	for nodeID, tidx := range g.NodeToMsg {
		if int(nodeID) < len(r) && tidx < len(out) {
			out[tidx] = math.Log1p(logScale * math.Max(0, r[nodeID]))
		}
	}
	return out
}
