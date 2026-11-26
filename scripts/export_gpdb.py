#!/usr/bin/env python3
"""
Export sanitized real long sessions from ~/.gp/gp.db into gp-context-lab Suite D JSON cases.
Matches datasets/case.schema.json and strips Feishu boilerplate wrappers & PII.
"""
import argparse
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Tuple

RE_FEISHU_WRAPPER = re.compile(
    r"^\[角色人设\].*?(?:展示到飞书卡片|跨群发消息时才调用）)\]\s*",
    re.DOTALL,
)
RE_CHAT_ID = re.compile(r"oc_[a-zA-Z0-9]+")
RE_OPEN_ID = re.compile(r"ou_[a-zA-Z0-9]+")
RE_FEISHU_URL = re.compile(r"https?://[^\s<>\"]*(?:feishu\.cn|larksuite\.com|x2robot\.com)[^\s<>\"]*")
RE_USER_PATH = re.compile(r"/Users/[^/\s\"']+")
RE_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
RE_SECRET = re.compile(r"\b(?:sk|pt)-[A-Za-z0-9_-]{16,}\b")
RE_CODE_TOKENS = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{3,}")


def sanitize_text(s: str, strip_wrapper: bool = False) -> str:
    if not s:
        return ""
    if strip_wrapper:
        s = RE_FEISHU_WRAPPER.sub("", s, count=1)
    s = RE_CHAT_ID.sub("chat_redacted", s)
    s = RE_OPEN_ID.sub("user_redacted", s)
    s = RE_FEISHU_URL.sub("https://internal.redacted/doc", s)
    s = RE_USER_PATH.sub("/workspace", s)
    s = RE_JWT.sub("<JWT_REDACTED>", s)
    s = RE_SECRET.sub("<SECRET_REDACTED>", s)
    return s.strip()


def extract_anchors(text: str) -> List[str]:
    anchors = []
    for tok in RE_CODE_TOKENS.findall(text):
        tl = tok.lower()
        if tl in {"http", "https", "internal.redacted", "workspace", "redacted", "user_redacted", "chat_redacted"}:
            continue
        if "/" in tok or "_" in tok or any(c.isupper() for c in tok[1:]):
            anchors.append(tl)
    for kw in ["跨租户", "分发参数", "采集租户", "质检租户", "创建租户", "自由数采", "格式", "会议纪要", "排期", "协议"]:
        if kw in text:
            anchors.append(kw)
    return list(dict.fromkeys(anchors))


def load_session_turns(conn: sqlite3.Connection, sid: str, max_msgs: int) -> Tuple[str, List[Dict[str, Any]]]:
    rows = conn.execute(
        "SELECT id, payload FROM entries WHERE session_id=? AND entry_type='message' ORDER BY rowid ASC",
        (sid,),
    ).fetchall()
    if max_msgs > 0:
        rows = rows[:max_msgs]

    sys_prompt = "You are gp, a high-performance Go coding and ops agent."
    turns: List[Dict[str, Any]] = []
    for idx, (eid, raw_payload) in enumerate(rows, start=1):
        obj = json.loads(raw_payload)
        msg = obj.get("message", {})
        role = msg.get("role", "user")
        if role not in ("user", "assistant", "tool"):
            continue
        raw_content = msg.get("content", "")
        if not isinstance(raw_content, str):
            raw_content = json.dumps(raw_content, ensure_ascii=False)
        content = sanitize_text(raw_content, strip_wrapper=(role == "user"))

        # Cap monstrous individual tool dumps (e.g., 500KB+ binary/AST dumps) at 12KB to keep replay tractable.
        if role == "tool" and len(content) > 12000:
            content = content[:12000] + "\n...[truncated during export]..."

        t: Dict[str, Any] = {
            "turn_id": idx,
            "entry_id": eid,
            "role": role,
            "content": content,
        }

        tcs = msg.get("tool_calls") or []
        if tcs and role == "assistant":
            norm_tcs = []
            for tc in tcs:
                fn = tc.get("function") or {}
                name = tc.get("name") or fn.get("name") or "tool"
                args = tc.get("arguments") if "arguments" in tc else fn.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                norm_tcs.append({
                    "id": tc.get("id", f"call_{idx}"),
                    "name": name,
                    "arguments": sanitize_text(args),
                })
            t["tool_calls"] = norm_tcs

        if msg.get("tool_call_id"):
            t["tool_call_id"] = msg["tool_call_id"]

        turns.append(t)
    return sys_prompt, turns


def build_probes(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(turns)
    if n < 20:
        return []
    # Index anchors in earlier turns (first 70% of session) to pair with late real user turns.
    probes: List[Dict[str, Any]] = []
    for i in range(int(n * 0.45), n):
        t = turns[i]
        if t["role"] != "user" or len(t["content"]) < 10:
            continue
        q_anchors = extract_anchors(t["content"])
        if not q_anchors:
            continue

        # Find earliest up to 3 distinct earlier turns (more than 15 turns before i) containing these anchors.
        ev_ids: List[str] = []
        for early in turns[: max(0, i - 15)]:
            txt = early["content"].lower()
            if any(a in txt for a in q_anchors):
                # Prefer substantive user declarations or tool read outputs.
                if early["role"] in ("user", "tool") and len(early["content"]) >= 25:
                    ev_ids.append(early["entry_id"])
                    if len(ev_ids) >= 3:
                        break
        if len(ev_ids) >= 2:
            probes.append({
                "probe_id": f"probe_real_{t['turn_id']}",
                "at_turn": t["turn_id"],
                "query": t["content"],
                "required_evidence_entry_ids": ev_ids,
                "forbidden_superseded_entry_ids": [],
                "expected_rubric": [f"回答需准确结合历史上下文 {', '.join(ev_ids)} 中的事实与约束"],
            })
        if len(probes) >= 3:
            break

    if not probes:
        # Fallback: anchor final user turn to earliest substantial user/tool entries.
        last_u = next((t for t in reversed(turns) if t["role"] == "user" and len(t["content"]) >= 8), turns[-1])
        ev = [t["entry_id"] for t in turns[: max(1, len(turns) // 2)] if t["role"] in ("user", "tool") and len(t["content"]) >= 40][:2]
        if ev:
            probes.append({
                "probe_id": f"probe_real_{last_u['turn_id']}",
                "at_turn": last_u["turn_id"],
                "query": last_u["content"],
                "required_evidence_entry_ids": ev,
                "forbidden_superseded_entry_ids": [],
                "expected_rubric": ["回答需保留早期核心目标与文件上下文"],
            })
    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.expanduser("~/.gp/gp.db"))
    parser.add_argument("--out", default="./datasets/real_replay")
    parser.add_argument("--min-msgs", type=int, default=80)
    parser.add_argument("--max-msgs", type=int, default=240)
    parser.add_argument("--token-budget", type=int, default=6000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    sids = [
        r[0]
        for r in conn.execute(
            """
            SELECT session_id
            FROM entries
            WHERE entry_type='message'
            GROUP BY session_id
            HAVING COUNT(*) >= ?
            ORDER BY COUNT(*) DESC
            LIMIT 6
            """,
            (args.min_msgs,),
        ).fetchall()
    ]

    exported = 0
    for idx, sid in enumerate(sids, start=1):
        sys_prompt, turns = load_session_turns(conn, sid, args.max_msgs)
        probes = build_probes(turns)
        if not turns or not probes:
            continue
        suffix = re.sub(r"[^a-zA-Z0-9]+", "_", sid[-14:]).strip("_").lower()
        cid = f"suite_d_real_{idx:02d}_{suffix}"
        case = {
            "case_id": cid,
            "suite": "suite_d_replay",
            "title": f"真实 gp.db 长会话脱敏回放 #{idx} ({len(turns)} msgs)",
            "description": f"从本地 gp.db 脱敏导出的真实生产长轨迹（原始消息 {len(turns)} 条，含 {len(probes)} 个真实追问探针）。",
            "system_prompt": sys_prompt,
            "token_budget": args.token_budget,
            "turns": turns,
            "eval_probes": probes,
        }
        out_path = os.path.join(args.out, f"{cid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=2)
        total_chars = sum(len(t["content"]) for t in turns)
        print(f"Exported {cid}: turns={len(turns)}, chars={total_chars}, probes={len(probes)}")
        for p in probes:
            print(f"   probe@{p['at_turn']}: q={p['query'][:60]}... req={p['required_evidence_entry_ids']}")
        exported += 1
    print(f"Total exported real replay cases: {exported}")


if __name__ == "__main__":
    main()
