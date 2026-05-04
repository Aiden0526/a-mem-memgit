from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARKS = [
    REPO_ROOT / "data/PersonaMem-v2-enhanced-release/benchmark_v34/text/benchmark_9p_nonood_v34.csv",
    REPO_ROOT / "data/PersonaMem-v2-enhanced-release/benchmark_v34/text/benchmark_9p_ood_v34.csv",
]
DEFAULT_OUTPUT = REPO_ROOT / "analysis/gold_persona_patches_9p_all.jsonl"
SIZE = "32k"


def sanitize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def flatten_snippet(snippet_raw: str) -> str:
    if not snippet_raw:
        return ""
    try:
        turns = json.loads(snippet_raw)
        if isinstance(turns, list):
            bits = []
            for turn in turns:
                role = str(turn.get("role", "")).upper()
                content = sanitize_text(turn.get("content", ""))
                if content:
                    bits.append(f"{role}: {content}")
            return " | ".join(bits)
    except Exception:
        pass
    return sanitize_text(snippet_raw)


def first_user_turn(snippet_raw: str) -> str:
    if not snippet_raw:
        return ""
    try:
        turns = json.loads(snippet_raw)
        if isinstance(turns, list):
            for turn in turns:
                if str(turn.get("role", "")).lower() == "user":
                    return sanitize_text(turn.get("content", ""))
    except Exception:
        pass
    return sanitize_text(snippet_raw)


def parse_snippet_turns(snippet_raw: str) -> List[Dict[str, str]]:
    try:
        turns = json.loads(snippet_raw)
        if isinstance(turns, list):
            return [
                {
                    "role": str(turn.get("role", "")).lower(),
                    "content": sanitize_text(turn.get("content", "")),
                }
                for turn in turns
            ]
    except Exception:
        pass
    return []


def resolve_chat_history_path(row: Dict[str, str], size: str, persona_root: Path) -> Path:
    link = row.get(f"chat_history_{size}_link", "")
    if link.startswith("data/"):
        link = link[len("data/"):]
    return persona_root / "data" / link


def build_sample_id(chat_history_path: Path, include_system_messages: bool = True) -> str:
    payload = f"{chat_history_path.resolve()}|include_system={include_system_messages}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def candidate_persona_roots() -> List[Path]:
    return [
        REPO_ROOT / "data/PersonaMem-v2-enhanced-release",
        REPO_ROOT / "data/Persona-release",
    ]


def resolve_sample_ids(row: Dict[str, str]) -> Dict[str, str]:
    sample_ids: Dict[str, str] = {}
    for root in candidate_persona_roots():
        chat_path = resolve_chat_history_path(row, SIZE, root)
        sample_ids[root.name] = build_sample_id(chat_path, include_system_messages=True)
    return sample_ids


def resolve_sample_id(row: Dict[str, str]) -> str:
    sample_ids = resolve_sample_ids(row)
    for root in candidate_persona_roots():
        chat_path = resolve_chat_history_path(row, SIZE, root)
        if chat_path.exists():
            return sample_ids[root.name]
    return next(iter(sample_ids.values()))


def resolve_existing_chat_path(row: Dict[str, str]) -> Path:
    for root in candidate_persona_roots():
        chat_path = resolve_chat_history_path(row, SIZE, root)
        if chat_path.exists():
            return chat_path
    return resolve_chat_history_path(row, SIZE, candidate_persona_roots()[0])


def event_key(row: Dict[str, str], benchmark_name: str) -> Tuple[str, ...]:
    return (
        benchmark_name,
        row.get("persona_id", ""),
        row.get("change_family", ""),
        row.get("change_step", ""),
        row.get("prev_pref", ""),
        row.get("preference", ""),
        row.get("related_conversation_snippet", ""),
    )


def build_retrieval_views(row: Dict[str, str], snippet_text: str, trigger_text: str, temporal_note: str) -> Dict[str, str]:
    change_family = sanitize_text(row.get("change_family", ""))
    prev_pref = sanitize_text(row.get("prev_pref", ""))
    preference = sanitize_text(row.get("preference", ""))
    family_query = sanitize_text(row.get("family_query", ""))
    family_correct = sanitize_text(row.get("family_correct", ""))
    topic_query = sanitize_text(row.get("topic_query", ""))
    pref_type = sanitize_text(row.get("pref_type", ""))
    who = sanitize_text(row.get("who", ""))

    before_after_doc = " ".join(
        part
        for part in [
            temporal_note,
            f"Change family: {change_family}." if change_family else "",
            f"Who changed: {who}." if who else "",
            f"Preference type: {pref_type}." if pref_type else "",
            f"Earlier preference: {prev_pref}." if prev_pref else "",
            f"Later preference: {preference}." if preference else "",
        ]
        if part
    )

    hybrid_doc = " ".join(
        part
        for part in [
            before_after_doc,
            f"Trigger snippet: {trigger_text}." if trigger_text else "",
            f"Related conversation: {snippet_text}." if snippet_text else "",
            f"Question form: {family_query}." if family_query else "",
            f"Gold answer form: {family_correct}." if family_correct else "",
            f"Topic query: {topic_query}." if topic_query else "",
        ]
        if part
    )

    trigger_doc = " ".join(
        part
        for part in [
            temporal_note,
            f"Trigger snippet: {trigger_text}." if trigger_text else "",
            f"Related conversation: {snippet_text}." if snippet_text else "",
        ]
        if part
    )

    query_alignment_doc = " ".join(
        part
        for part in [
            temporal_note,
            f"Question form: {family_query}." if family_query else "",
            f"Earlier preference: {prev_pref}." if prev_pref else "",
            f"Later preference: {preference}." if preference else "",
            f"Gold answer form: {family_correct}." if family_correct else "",
        ]
        if part
    )

    return {
        "before_after_doc": before_after_doc,
        "trigger_doc": trigger_doc,
        "hybrid_doc": hybrid_doc,
        "query_alignment_doc": query_alignment_doc,
    }


def load_chat_history(chat_path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(chat_path.read_text(encoding="utf-8"))
    return list(obj.get("chat_history") or [])


def find_snippet_span(chat_history: List[Dict[str, Any]], snippet_raw: str) -> Optional[Dict[str, Any]]:
    snippet_turns = parse_snippet_turns(snippet_raw)
    if not snippet_turns:
        return None
    history = [
        (str(turn.get("role", "")).lower(), sanitize_text(turn.get("content", "")))
        for turn in chat_history
    ]
    snippet = [(turn["role"], turn["content"]) for turn in snippet_turns]
    limit = len(history) - len(snippet) + 1
    for start in range(max(0, limit)):
        if all(snippet[j] == history[start + j] for j in range(len(snippet))):
            end = start + len(snippet) - 1
            trigger_offset = 0
            for offset, turn in enumerate(snippet_turns):
                if turn["role"] == "user":
                    trigger_offset = offset
                    break
            trigger_index = start + trigger_offset
            return {
                "session_id": 0,
                "snippet_start_turn_index": start,
                "snippet_end_turn_index": end,
                "snippet_turn_count": len(snippet),
                "trigger_turn_index": trigger_index,
                "trigger_turn_number": trigger_index + 1,
                "temporal_order_note": (
                    f"Temporal order: session 0, snippet turns {start}-{end}, trigger turn {trigger_index}. "
                    f"Later larger turn indices are temporally more recent within the same preference domain."
                ),
            }
    return None


def build_gold_patch(
    row: Dict[str, str],
    ordinal: int,
    benchmark_path: Path,
    split_name: str,
    temporal_span: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    snippet_raw = row.get("related_conversation_snippet", "")
    snippet_text = flatten_snippet(snippet_raw)
    trigger_text = first_user_turn(snippet_raw) or snippet_text
    sample_id = resolve_sample_id(row)
    sample_ids = resolve_sample_ids(row)
    patch_id = f"gold_patch_{split_name}_{ordinal:06d}"

    temporal_span = temporal_span or {
        "session_id": 0,
        "snippet_start_turn_index": None,
        "snippet_end_turn_index": None,
        "snippet_turn_count": len(parse_snippet_turns(snippet_raw)),
        "trigger_turn_index": None,
        "trigger_turn_number": None,
        "temporal_order_note": (
            "Temporal order: exact turn indices could not be recovered from the benchmark snippet. "
            "Use change_step as a coarse progression signal for relative recency."
        ),
    }
    retrieval_views = build_retrieval_views(row, snippet_text, trigger_text, temporal_span["temporal_order_note"])

    return {
        "patch_id": patch_id,
        "patch_source": "benchmark_ground_truth",
        "benchmark_file": str(benchmark_path),
        "benchmark_split": split_name,
        "persona_id": row.get("persona_id", ""),
        "sample_id": sample_id,
        "sample_ids": sample_ids,
        "chat_history_32k_link": row.get("chat_history_32k_link", ""),
        "raw_persona_file": row.get("raw_persona_file", ""),
        "change_family": row.get("change_family", ""),
        "change_k": row.get("change_k", ""),
        "change_step": row.get("change_step", ""),
        "change_enhanced": row.get("change_enhanced", ""),
        "updated": row.get("updated", ""),
        "who": row.get("who", ""),
        "pref_type": row.get("pref_type", ""),
        "topic_query": row.get("topic_query", ""),
        "topic_preference": row.get("topic_preference", ""),
        "family_query": row.get("family_query", ""),
        "family_correct": row.get("family_correct", ""),
        "family_incorrect": row.get("family_incorrect", ""),
        "prev_pref": row.get("prev_pref", ""),
        "preference": row.get("preference", ""),
        "related_conversation_snippet_raw": snippet_raw,
        "related_conversation_snippet_text": snippet_text,
        "trigger_text": trigger_text,
        "temporal_span": temporal_span,
        "gold_patch_overall": {
            "change_family": row.get("change_family", ""),
            "before_preference": row.get("prev_pref", ""),
            "after_preference": row.get("preference", ""),
            "summary": sanitize_text(
                f"User preference changed via {row.get('change_family', '')} from {row.get('prev_pref', '')} to {row.get('preference', '')}."
            ),
            "temporal_order_note": temporal_span["temporal_order_note"],
        },
        "gold_changed_item": {
            "item_kind": "gold_change",
            "before_state_text": row.get("prev_pref", ""),
            "after_state_text": row.get("preference", ""),
            "trigger_text": trigger_text,
            "snippet_text": snippet_text,
            "trigger_turn_index": temporal_span["trigger_turn_index"],
            "trigger_turn_number": temporal_span["trigger_turn_number"],
        },
        "retrieval_views": retrieval_views,
        "default_index_document": retrieval_views["hybrid_doc"],
    }


def load_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def split_name_for_path(path: Path) -> str:
    name = path.name
    if "nonood" in name:
        return "9p_nonood"
    if "ood" in name:
        return "9p_ood"
    return path.stem


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold patches from PersonaMem 9p benchmark ground truth.")
    parser.add_argument("--benchmarks", nargs="*", type=Path, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"splits": {}}

    for benchmark_path in args.benchmarks:
        split_name = split_name_for_path(benchmark_path)
        rows = [row for row in load_rows(benchmark_path) if row.get("change_family") and str(row.get("updated", "")).lower() == "true"]
        deduped: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            key = event_key(row, split_name)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        chat_cache: Dict[str, List[Dict[str, Any]]] = {}
        split_rows: List[Dict[str, Any]] = []
        recovered_spans = 0
        for idx, row in enumerate(deduped, 1):
            chat_path = resolve_existing_chat_path(row)
            chat_key = str(chat_path)
            if chat_key not in chat_cache:
                chat_cache[chat_key] = load_chat_history(chat_path)
            temporal_span = find_snippet_span(chat_cache[chat_key], row.get("related_conversation_snippet", ""))
            if temporal_span:
                recovered_spans += 1
            split_rows.append(build_gold_patch(row, idx, benchmark_path, split_name, temporal_span))

        split_output = args.output.with_name(f"gold_persona_patches_{split_name}.jsonl")
        write_jsonl(split_output, split_rows)
        all_rows.extend(split_rows)
        summary["splits"][split_name] = {
            "benchmark": str(benchmark_path),
            "output": str(split_output),
            "changed_rows": len(rows),
            "unique_gold_patches": len(split_rows),
            "recovered_temporal_spans": recovered_spans,
        }

    write_jsonl(args.output, all_rows)
    summary["combined_output"] = str(args.output)
    summary["combined_unique_gold_patches"] = len(all_rows)
    summary["fields"] = [
        "patch_id",
        "benchmark_split",
        "persona_id",
        "sample_id",
        "change_family",
        "change_step",
        "prev_pref",
        "preference",
        "related_conversation_snippet_text",
        "trigger_text",
        "temporal_span",
        "gold_patch_overall",
        "gold_changed_item",
        "retrieval_views",
        "default_index_document",
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
