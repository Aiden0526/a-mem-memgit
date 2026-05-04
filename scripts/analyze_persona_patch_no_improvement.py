"""
Offline diagnosis for PersonaMem patch no-improvement on non-OOD runs.

This script joins:
  - non-OOD benchmark rows
  - saved patch-eval result CSVs
  - cached patch stores / patch retriever embeddings
  - cached robust global-memory stores

It produces per-row attribution and summary reports under analysis/.

Usage:
    source a-mem/bin/activate
    python scripts/analyze_persona_patch_no_improvement.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK = REPO_ROOT / "data/PersonaMem-v2-enhanced-release/benchmark_v34/text/benchmark_49p_nonood.csv"
DEFAULT_RESULTS_GLOB = str(REPO_ROOT / "results/persona_patch_v7_nonood_045.worker_*.json")
DEFAULT_PATCH_CACHE = REPO_ROOT / "cached_memories_persona_patch_openai_gpt-5.4-mini-2026-03-17_32k_always_prefaware_v7"
DEFAULT_ROBUST_CACHE = REPO_ROOT / "cached_memories_persona_robust_openai_gpt-5.4-mini-2026-03-17_32k"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "analysis"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def sanitize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_user_query(user_query_raw: str) -> str:
    try:
        user_query = json.loads(user_query_raw)
    except (json.JSONDecodeError, TypeError):
        try:
            user_query = ast.literal_eval(user_query_raw)
        except (ValueError, SyntaxError):
            user_query = {"role": "user", "content": str(user_query_raw).strip('"').strip("'")}
    if not isinstance(user_query, dict):
        user_query = {"role": "user", "content": str(user_query)}
    content = sanitize_text(user_query.get("content", ""))
    if content:
        content += " Please recall my related preferences from our conversation history to give personalized responses."
    return content


def parse_incorrect_answers(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    try:
        values = json.loads(raw_value)
        return [str(value) for value in values]
    except json.JSONDecodeError:
        return []


def build_mcq_query_text(row: Dict[str, str]) -> str:
    question = parse_user_query(row.get("user_query", ""))
    options = [row.get("correct_answer", "")] + parse_incorrect_answers(row.get("incorrect_answers", ""))
    option_text = "\n".join(f"- {sanitize_text(opt)}" for opt in options if sanitize_text(opt))
    return sanitize_text(f"{question}\nOptions:\n{option_text}")


def extract_user_turn(snippet_str: str) -> str:
    if not snippet_str:
        return ""
    try:
        turns = json.loads(snippet_str)
        user_bits = [sanitize_text(t.get("content", "")) for t in turns if str(t.get("role", "")).lower() == "user"]
        return " ".join(bit for bit in user_bits if bit)
    except Exception:
        return sanitize_text(snippet_str)


def build_transition_query(row: Dict[str, str]) -> str:
    snippet = extract_user_turn(row.get("related_conversation_snippet", ""))
    return sanitize_text(
        f"Preference change family: {row.get('change_family', '')}. "
        f"Previous preference: {row.get('prev_pref', '')}. "
        f"Current preference: {row.get('preference', '')}. "
        f"Benchmark question: {parse_user_query(row.get('user_query', ''))}. "
        f"Conversation evidence: {snippet}"
    )


def resolve_chat_history_path(row: Dict[str, str], size: str, persona_root: Path) -> Path:
    link = row.get(f"chat_history_{size}_link", "")
    if link.startswith("data/"):
        link = link[len("data/"):]
    return persona_root / "data" / link


def build_sample_id(chat_history_path: Path, include_system_messages: bool = True) -> str:
    payload = f"{chat_history_path.resolve()}|include_system={include_system_messages}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def tokenize_keywords(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 4}


def lexical_overlap_score(a: str, b: str) -> float:
    a_tokens = tokenize_keywords(a)
    b_tokens = tokenize_keywords(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return overlap / max(1, min(len(a_tokens), len(b_tokens)))


def patch_before_after_texts(patch_view: PatchRecordView) -> tuple[str, str, str]:
    before_parts = []
    after_parts = []
    for block in patch_view.full_record.get("changed_nodes", []):
        before_parts.append(((block.get("before") or {}).get("context", "")))
        after_parts.append(((block.get("after") or {}).get("context", "")))
    trigger_text = str((patch_view.full_record.get("trigger_turn") or {}).get("text", ""))
    return " ".join(before_parts), " ".join(after_parts), trigger_text


def strict_grounded_patch_match(row: Dict[str, str], patch_view: PatchRecordView) -> tuple[bool, Dict[str, float]]:
    family = str(row.get("change_family", "")).lower()
    family_ok = (not family) or family_match(family, patch_view)
    before_text, after_text, trigger_text = patch_before_after_texts(patch_view)
    prev_overlap = lexical_overlap_score(str(row.get("prev_pref", "")), before_text)
    curr_overlap = lexical_overlap_score(str(row.get("preference", "")), after_text)
    trigger_overlap = lexical_overlap_score(extract_user_turn(row.get("related_conversation_snippet", "")), trigger_text)
    grounded = family_ok and max(prev_overlap, curr_overlap, trigger_overlap) >= 0.08 and (curr_overlap >= 0.08 or prev_overlap >= 0.08)
    return grounded, {
        "prev_overlap": prev_overlap,
        "curr_overlap": curr_overlap,
        "trigger_overlap": trigger_overlap,
    }


def contains_transition_signal(text: str) -> bool:
    lower = text.lower()
    markers = [
        "changed from",
        "evolved from",
        "instead of",
        "rather than",
        "later",
        "now prefers",
        "no longer",
        "switched to",
        "became conditional",
        "time restriction",
    ]
    return any(marker in lower for marker in markers)


def best_indices(scores: np.ndarray, top_k: int) -> List[int]:
    if scores.size == 0:
        return []
    top_k = min(top_k, scores.size)
    idx = np.argsort(scores)[-top_k:][::-1]
    return [int(i) for i in idx]


def format_rate(num: int, den: int) -> str:
    if den == 0:
        return "0/0 = 0.000"
    return f"{num}/{den} = {num / den:.3f}"


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_result_rows(glob_pattern: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(glob.glob(glob_pattern)):
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def build_row_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("persona_id", "")),
        sanitize_text(row.get("user_query", "")),
        str(row.get("change_family", "")),
        str(row.get("change_step", "")),
        str(row.get("ood_type", "")),
    )


def parse_injected_patch_refs(prompt: str) -> List[Tuple[str, str, str]]:
    if not prompt:
        return []
    pattern = re.compile(
        r"Session\s+(\d+)\s+TurnIndex\s+(\d+).*?change_type:\s*([a-z_]+)",
        re.IGNORECASE | re.DOTALL,
    )
    refs: List[Tuple[str, str, str]] = []
    for session_id, turn_index, change_type in pattern.findall(prompt):
        refs.append((session_id, turn_index, change_type.lower()))
    return refs


def build_patch_index_document_from_record(patch_record: Dict[str, Any]) -> str:
    overall = patch_record.get("patch_overall", {})
    pref_domain = patch_record.get("pref_domain", "")
    change_type = patch_record.get("change_type", "") or overall.get("change_type", "")

    before_contexts: List[str] = []
    after_contexts: List[str] = []
    link_summaries: List[str] = []
    for block in patch_record.get("changed_nodes", []):
        before = (block.get("before") or {}).get("context", "")
        after = (block.get("after") or {}).get("context", "")
        if before:
            before_contexts.append(before)
        if after:
            after_contexts.append(after)
        link_summary = block.get("link_change_summary", "")
        if link_summary:
            link_summaries.append(link_summary)

    parts: List[str] = []
    if pref_domain:
        parts.append(f"Preference domain: {pref_domain}.")
    if before_contexts:
        parts.append(f"Previously: {' '.join(before_contexts)}.")
    if after_contexts:
        parts.append(f"Now prefers: {' '.join(after_contexts)}.")
    if change_type and change_type != "none":
        parts.append(f"Change type: {change_type}.")
    summary = overall.get("overall_summary", "")
    if summary:
        parts.append(f"Summary: {summary}.")
    if link_summaries:
        parts.append(f"Related changes: {' '.join(link_summaries)}.")
    trigger_text = ((patch_record.get("trigger_turn") or {}).get("text", ""))[:120]
    if trigger_text:
        parts.append(f"Trigger snippet: {trigger_text}")
    return " ".join(parts).strip()


@dataclass
class PatchRecordView:
    patch_id: str
    index_record: Dict[str, Any]
    full_record: Dict[str, Any]
    cached_doc: str
    rebuilt_doc: str


@dataclass
class PatchSampleArtifacts:
    sample_id: str
    patch_views: List[PatchRecordView]
    cached_embeddings: np.ndarray
    rebuilt_embeddings: np.ndarray
    build_status: Dict[str, Any]
    ref_to_patch_ids: Dict[Tuple[str, str, str], List[str]]


@dataclass
class MemoryMatch:
    similarity: float
    content: str
    context: str
    change_type: str
    pref_domain: str
    is_preference: Optional[bool]
    has_transition_signal: bool


@dataclass
class RobustSampleArtifacts:
    sample_id: str
    corpus: List[str]
    embeddings: np.ndarray
    memory_lookup: Dict[str, Any]


class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)


def load_patch_artifacts(cache_root: Path, sample_id: str, embedder: Embedder) -> Optional[PatchSampleArtifacts]:
    sample_dir = cache_root / f"sample_{sample_id}"
    index_path = sample_dir / "patch_index_records.jsonl"
    emb_path = sample_dir / "patch_retriever_embeddings.npy"
    if not index_path.exists() or not emb_path.exists():
        return None

    index_records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cached_embeddings = np.load(emb_path).astype(np.float32)
    if cached_embeddings.size:
        norms = np.linalg.norm(cached_embeddings, axis=1, keepdims=True)
        cached_embeddings = cached_embeddings / np.where(norms == 0, 1.0, norms)

    patches_dir = sample_dir / "patches"
    full_patches: Dict[str, Dict[str, Any]] = {}
    if patches_dir.exists():
        for patch_path in sorted(patches_dir.glob("patch_*.json")):
            full_patches[patch_path.stem] = json.loads(patch_path.read_text(encoding="utf-8"))

    patch_views: List[PatchRecordView] = []
    rebuilt_docs: List[str] = []
    ref_to_patch_ids: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for record in index_records:
        patch_id = str(record.get("patch_id", ""))
        full_record = full_patches.get(patch_id, dict(record))
        cached_doc = str(record.get("index_document", ""))
        rebuilt_doc = build_patch_index_document_from_record(full_record)
        patch_views.append(PatchRecordView(patch_id, record, full_record, cached_doc, rebuilt_doc))
        rebuilt_docs.append(rebuilt_doc)
        trigger_turn = full_record.get("trigger_turn") or {}
        ref = (
            str(trigger_turn.get("session_id", record.get("session_id", ""))),
            str(trigger_turn.get("turn_position", record.get("turn_position", ""))),
            str(full_record.get("change_type", record.get("change_type", "none"))).lower(),
        )
        ref_to_patch_ids[ref].append(patch_id)

    rebuilt_embeddings = embedder.encode(rebuilt_docs)
    build_status_path = sample_dir / "build_status.json"
    build_status = json.loads(build_status_path.read_text(encoding="utf-8")) if build_status_path.exists() else {}
    return PatchSampleArtifacts(sample_id, patch_views, cached_embeddings, rebuilt_embeddings, build_status, dict(ref_to_patch_ids))


def load_robust_artifacts(cache_root: Path, sample_id: str) -> Optional[RobustSampleArtifacts]:
    sample_dir = cache_root / f"sample_{sample_id}"
    graph_dir = sample_dir / "global_graph"
    ret_path = graph_dir / f"retriever_cache_sample_{sample_id}.pkl"
    emb_path = graph_dir / f"retriever_cache_embeddings_sample_{sample_id}.npy"
    mem_path = graph_dir / f"memory_cache_sample_{sample_id}.pkl"
    if not ret_path.exists() or not emb_path.exists() or not mem_path.exists():
        return None

    with ret_path.open("rb") as f:
        retriever_payload = pickle.load(f)
    corpus = list(retriever_payload.get("corpus", []))
    embeddings = np.load(emb_path).astype(np.float32)
    if embeddings.size:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
    with mem_path.open("rb") as f:
        memories = pickle.load(f)

    memory_lookup: Dict[str, Any] = {}
    for note in memories.values():
        content = str(getattr(note, "content", ""))
        memory_lookup[content[:80]] = note
    return RobustSampleArtifacts(sample_id, corpus, embeddings, memory_lookup)


def best_memory_match(query_embedding: np.ndarray, artifacts: Optional[RobustSampleArtifacts]) -> Optional[MemoryMatch]:
    if artifacts is None or artifacts.embeddings.size == 0:
        return None
    scores = artifacts.embeddings @ query_embedding
    idx = int(np.argmax(scores))
    content = artifacts.corpus[idx] if idx < len(artifacts.corpus) else ""
    note = artifacts.memory_lookup.get(content[:80])
    context = str(getattr(note, "context", "")) if note else ""
    return MemoryMatch(
        similarity=float(scores[idx]),
        content=content,
        context=context,
        change_type=str(getattr(note, "change_type", "none")) if note else "none",
        pref_domain=str(getattr(note, "pref_domain", "")) if note else "",
        is_preference=getattr(note, "is_preference", None) if note else None,
        has_transition_signal=contains_transition_signal(context or content),
    )


def recall_at_k(indices: Iterable[int], target: int) -> bool:
    return target >= 0 and any(int(i) == int(target) for i in indices)


def family_match(row_family: str, patch_view: PatchRecordView) -> bool:
    ct = str(patch_view.full_record.get("change_type", patch_view.index_record.get("change_type", "none"))).lower()
    return ct == str(row_family).lower()


def row_result_status(is_correct_value: str) -> bool:
    return str(is_correct_value).strip().lower() == "true"


def classify_root_cause(
    sample_has_patches: bool,
    plausible_patch_idx: int,
    plausible_score: float,
    plausible_in_cached_topk: bool,
    injected_patch_ids: List[str],
    injected_contains_plausible: bool,
    run_threshold: float,
    top_cached_score: float,
) -> str:
    if not sample_has_patches:
        return "no_patch_inventory"
    if plausible_patch_idx < 0 or plausible_score < 0.40:
        return "patch_generation_miss"
    if injected_patch_ids:
        if injected_contains_plausible:
            return "plausible_patch_injected"
        return "wrong_patch_injected"
    if plausible_in_cached_topk:
        if top_cached_score < run_threshold:
            return "threshold_blocked_plausible_patch"
        return "retrieved_but_not_injected"
    return "retrieval_miss"


def gather_failure_examples(detail_rows: List[Dict[str, Any]], max_per_family: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        if row["root_cause"] not in {"wrong_patch_injected", "retrieval_miss", "patch_generation_miss"}:
            continue
        family = str(row["change_family"])
        if len(grouped[family]) >= max_per_family:
            continue
        grouped[family].append(row)
    return grouped


def detect_persona_root(benchmark_file: Path) -> Path:
    for candidate in [benchmark_file.parent, *benchmark_file.parents]:
        if (candidate / "EVALUATION_GUIDE.md").exists():
            return candidate
    return benchmark_file.parent.parent.parent


def candidate_persona_roots(primary_root: Path) -> List[Path]:
    candidates = [primary_root]
    persona_release_root = REPO_ROOT / "data/Persona-release"
    if persona_release_root not in candidates:
        candidates.append(persona_release_root)
    return candidates


def resolve_sample_id_for_row(
    row: Dict[str, str],
    size: str,
    roots: Sequence[Path],
    patch_cache_root: Path,
    robust_cache_root: Path,
) -> str:
    fallback_sample_id = ""
    for root in roots:
        chat_path = resolve_chat_history_path(row, size, root)
        sample_id = build_sample_id(chat_path, include_system_messages=True)
        if not fallback_sample_id:
            fallback_sample_id = sample_id
        if (patch_cache_root / f"sample_{sample_id}").exists() or (robust_cache_root / f"sample_{sample_id}").exists():
            return sample_id
    return fallback_sample_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PersonaMem patch no-improvement offline")
    parser.add_argument("--benchmark_file", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--results_glob", type=str, default=DEFAULT_RESULTS_GLOB)
    parser.add_argument("--patch_cache_root", type=Path, default=DEFAULT_PATCH_CACHE)
    parser.add_argument("--robust_cache_root", type=Path, default=DEFAULT_ROBUST_CACHE)
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--size", type=str, default="32k")
    parser.add_argument("--run_label", type=str, default="persona_patch_v7_nonood_045")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--run_threshold", type=float, default=0.45)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = [row for row in load_csv_rows(args.benchmark_file) if not row.get("ood_type", "")]
    result_rows = load_result_rows(args.results_glob)
    result_map = {build_row_key(row): row for row in result_rows}
    matched_benchmark_rows = [row for row in benchmark_rows if build_row_key(row) in result_map]
    if matched_benchmark_rows:
        benchmark_rows = matched_benchmark_rows
    persona_root = detect_persona_root(args.benchmark_file.resolve())
    persona_roots = candidate_persona_roots(persona_root)

    embedder = Embedder()

    benchmark_key_order: List[Tuple[str, str, str, str, str]] = []
    sample_ids: Dict[Tuple[str, str, str, str, str], str] = {}
    patch_artifacts: Dict[str, Optional[PatchSampleArtifacts]] = {}
    robust_artifacts: Dict[str, Optional[RobustSampleArtifacts]] = {}
    query_texts: List[str] = []
    transition_texts: List[str] = []

    for row in benchmark_rows:
        key = build_row_key(row)
        benchmark_key_order.append(key)
        sample_id = resolve_sample_id_for_row(
            row,
            args.size,
            persona_roots,
            args.patch_cache_root,
            args.robust_cache_root,
        )
        sample_ids[key] = sample_id
        if sample_id not in patch_artifacts:
            patch_artifacts[sample_id] = load_patch_artifacts(args.patch_cache_root, sample_id, embedder)
        if sample_id not in robust_artifacts:
            robust_artifacts[sample_id] = load_robust_artifacts(args.robust_cache_root, sample_id)
        query_texts.append(build_mcq_query_text(row))
        transition_texts.append(build_transition_query(row))

    query_embeddings = embedder.encode(query_texts)
    transition_embeddings = embedder.encode(transition_texts)

    detail_rows: List[Dict[str, Any]] = []
    family_stats: Dict[str, Counter[str]] = defaultdict(Counter)
    recall_stats: Dict[str, Counter[str]] = defaultdict(Counter)
    root_cause_stats: Counter[str] = Counter()

    for idx, row in enumerate(benchmark_rows):
        key = benchmark_key_order[idx]
        sample_id = sample_ids[key]
        patch_data = patch_artifacts.get(sample_id)
        robust_data = robust_artifacts.get(sample_id)
        result_row = result_map.get(key, {})

        query_embedding = query_embeddings[idx]
        transition_embedding = transition_embeddings[idx]

        cached_top_indices: List[int] = []
        rebuilt_top_indices: List[int] = []
        top_cached_score = -1.0
        plausible_patch_idx = -1
        plausible_patch_score = -1.0
        plausible_family_patch_idx = -1
        plausible_family_patch_score = -1.0
        cached_top_patch_ids: List[str] = []
        rebuilt_top_patch_ids: List[str] = []
        plausible_patch_id = ""
        plausible_family_patch_id = ""
        sample_has_patches = bool(patch_data and patch_data.patch_views)

        if patch_data and patch_data.patch_views:
            cached_scores = patch_data.cached_embeddings @ query_embedding
            rebuilt_scores = patch_data.rebuilt_embeddings @ query_embedding
            transition_scores = patch_data.rebuilt_embeddings @ transition_embedding

            cached_top_indices = best_indices(cached_scores, args.top_k)
            rebuilt_top_indices = best_indices(rebuilt_scores, args.top_k)
            top_cached_score = float(cached_scores[cached_top_indices[0]]) if cached_top_indices else -1.0
            cached_top_patch_ids = [patch_data.patch_views[i].patch_id for i in cached_top_indices]
            rebuilt_top_patch_ids = [patch_data.patch_views[i].patch_id for i in rebuilt_top_indices]

            plausible_patch_idx = int(np.argmax(transition_scores))
            plausible_patch_score = float(transition_scores[plausible_patch_idx])
            plausible_patch_id = patch_data.patch_views[plausible_patch_idx].patch_id

            family_candidates = [
                (i, float(transition_scores[i]))
                for i, patch_view in enumerate(patch_data.patch_views)
                if family_match(str(row.get("change_family", "")), patch_view)
            ]
            if family_candidates:
                plausible_family_patch_idx, plausible_family_patch_score = max(family_candidates, key=lambda item: item[1])
                plausible_family_patch_id = patch_data.patch_views[plausible_family_patch_idx].patch_id

        prompt = result_row.get(f"raw_input_prompt_mcq_{args.size}", "")
        injected_refs = parse_injected_patch_refs(prompt)
        injected_patch_ids: List[str] = []
        if patch_data:
            for ref in injected_refs:
                injected_patch_ids.extend(patch_data.ref_to_patch_ids.get(ref, []))
        injected_patch_ids = list(dict.fromkeys(injected_patch_ids))

        injected_contains_plausible = plausible_patch_id in injected_patch_ids if plausible_patch_id else False
        plausible_in_cached_topk = recall_at_k(cached_top_indices, plausible_patch_idx)
        plausible_family_in_cached_topk = recall_at_k(cached_top_indices, plausible_family_patch_idx)

        robust_match = best_memory_match(transition_embedding, robust_data)
        root_cause = classify_root_cause(
            sample_has_patches=sample_has_patches,
            plausible_patch_idx=plausible_patch_idx,
            plausible_score=plausible_patch_score,
            plausible_in_cached_topk=plausible_in_cached_topk,
            injected_patch_ids=injected_patch_ids,
            injected_contains_plausible=injected_contains_plausible,
            run_threshold=args.run_threshold,
            top_cached_score=top_cached_score,
        )

        is_correct = row_result_status(result_row.get(f"is_correct_mcq_{args.size}", "False"))
        family = str(row.get("change_family", ""))
        family_stats[family]["total"] += 1
        family_stats[family][f"root:{root_cause}"] += 1
        if sample_has_patches:
            family_stats[family]["sample_has_patches"] += 1
        if plausible_patch_idx >= 0 and plausible_patch_score >= 0.40:
            family_stats[family]["plausible_patch_present"] += 1
        if injected_patch_ids:
            family_stats[family]["any_patch_injected"] += 1
        if injected_contains_plausible:
            family_stats[family]["plausible_patch_injected"] += 1
        if is_correct:
            family_stats[family]["correct"] += 1

        recall_stats[family]["total"] += 1
        if plausible_patch_idx >= 0 and plausible_patch_score >= 0.40:
            recall_stats[family]["gold_present"] += 1
            if plausible_in_cached_topk:
                recall_stats[family]["gold_cached_topk"] += 1
            if plausible_patch_id in rebuilt_top_patch_ids:
                recall_stats[family]["gold_rebuilt_topk"] += 1

        root_cause_stats[root_cause] += 1
        plausible_patch_view = patch_data.patch_views[plausible_patch_idx] if patch_data and plausible_patch_idx >= 0 else None
        strict_match = False
        strict_scores = {"prev_overlap": 0.0, "curr_overlap": 0.0, "trigger_overlap": 0.0}
        if plausible_patch_view is not None:
            strict_match, strict_scores = strict_grounded_patch_match(row, plausible_patch_view)
            if strict_match:
                family_stats[family]["strict_grounded_patch_match"] += 1

        detail_rows.append(
            {
                "persona_id": row.get("persona_id", ""),
                "sample_id": sample_id,
                "change_family": family,
                "change_k": row.get("change_k", ""),
                "change_step": row.get("change_step", ""),
                "family_query": row.get("family_query", ""),
                "user_query": row.get("user_query", ""),
                "preference": row.get("preference", ""),
                "prev_pref": row.get("prev_pref", ""),
                "conversation_snippet": extract_user_turn(row.get("related_conversation_snippet", "")),
                "is_correct": is_correct,
                "root_cause": root_cause,
                "sample_patch_count": len(patch_data.patch_views) if patch_data else 0,
                "plausible_patch_present": plausible_patch_idx >= 0 and plausible_patch_score >= 0.40,
                "plausible_patch_id": plausible_patch_id,
                "plausible_patch_score": round(plausible_patch_score, 4) if plausible_patch_score >= 0 else "",
                "plausible_patch_change_type": plausible_patch_view.full_record.get("change_type", "") if plausible_patch_view else "",
                "plausible_patch_family_match": bool(plausible_patch_view and family_match(family, plausible_patch_view)),
                "strict_grounded_patch_match": strict_match,
                "strict_prev_overlap": round(strict_scores["prev_overlap"], 4),
                "strict_curr_overlap": round(strict_scores["curr_overlap"], 4),
                "strict_trigger_overlap": round(strict_scores["trigger_overlap"], 4),
                "plausible_family_patch_id": plausible_family_patch_id,
                "plausible_family_patch_score": round(plausible_family_patch_score, 4) if plausible_family_patch_score >= 0 else "",
                "cached_top_patch_ids": safe_json_dumps(cached_top_patch_ids),
                "rebuilt_top_patch_ids": safe_json_dumps(rebuilt_top_patch_ids),
                "top_cached_score": round(top_cached_score, 4) if top_cached_score >= 0 else "",
                "plausible_in_cached_topk": plausible_in_cached_topk,
                "plausible_family_in_cached_topk": plausible_family_in_cached_topk,
                "injected_patch_ids": safe_json_dumps(injected_patch_ids),
                "injected_contains_plausible": injected_contains_plausible,
                "result_prompt_has_patch_block": "[Relevant Historical Patches]" in prompt,
                "robust_best_context_similarity": round(robust_match.similarity, 4) if robust_match else "",
                "robust_best_context_has_transition_signal": robust_match.has_transition_signal if robust_match else "",
                "robust_best_context": robust_match.context if robust_match else "",
                "robust_best_change_type": robust_match.change_type if robust_match else "",
                "result_path_present": bool(result_row),
            }
        )

    detail_csv = analysis_dir / "nonood_patch_capture.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    manifest_lines = [
        "# Persona Patch Run Manifest",
        "",
        f"- Benchmark: `{args.benchmark_file}`",
        f"- Results glob: `{args.results_glob}`",
        f"- Patch cache root: `{args.patch_cache_root}`",
        f"- Robust cache root: `{args.robust_cache_root}`",
        f"- Run label: `{args.run_label}`",
        f"- Run threshold assumed: `{args.run_threshold}`",
        f"- Top-k replayed: `{args.top_k}`",
        f"- Benchmark rows loaded: `{len(benchmark_rows)}`",
        f"- Result rows loaded: `{len(result_rows)}`",
        "",
        "## Notes",
        "",
        "- Retrieval replay uses offline embedding over the benchmark question plus answer options; current `persona_patch_v7_nonood_045` outputs do not include saved retrieval keyword columns.",
        "- Plausible gold patch is estimated as the highest-similarity patch to a transition query built from `prev_pref`, `preference`, `change_family`, and the related conversation snippet.",
        "- Robust-memory comparison is based on the best matching cached global-memory node for that same transition query.",
    ]
    (analysis_dir / "persona_patch_run_manifest.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    family_report_lines = [
        "# Non-OOD Patch Family Report",
        "",
        f"Overall rows: {len(detail_rows)}",
        "",
        "## Root Cause Counts",
        "",
    ]
    for cause, count in root_cause_stats.most_common():
        family_report_lines.append(f"- `{cause}`: {count}")

    family_report_lines.extend(["", "## By Family", ""])
    for family in sorted(family_stats.keys()):
        stats = family_stats[family]
        recall = recall_stats[family]
        total = stats["total"]
        family_report_lines.append(f"### {family}")
        family_report_lines.append(f"- Accuracy: {format_rate(stats['correct'], total)}")
        family_report_lines.append(f"- Sample has any patches: {format_rate(stats['sample_has_patches'], total)}")
        family_report_lines.append(f"- Plausible patch present: {format_rate(stats['plausible_patch_present'], total)}")
        family_report_lines.append(f"- Any patch injected: {format_rate(stats['any_patch_injected'], total)}")
        family_report_lines.append(f"- Plausible patch injected: {format_rate(stats['plausible_patch_injected'], total)}")
        family_report_lines.append(f"- Strict grounded patch match: {format_rate(stats['strict_grounded_patch_match'], total)}")
        family_report_lines.append(f"- Gold-present recall@{args.top_k} with cached docs: {format_rate(recall['gold_cached_topk'], recall['gold_present'])}")
        family_report_lines.append(f"- Gold-present recall@{args.top_k} with rebuilt docs: {format_rate(recall['gold_rebuilt_topk'], recall['gold_present'])}")
        for cause_key, count in sorted(((k, v) for k, v in stats.items() if k.startswith('root:')), key=lambda item: (-item[1], item[0])):
            family_report_lines.append(f"- {cause_key[5:]}: {count}")
        family_report_lines.append("")
    (analysis_dir / "nonood_patch_family_report.md").write_text("\n".join(family_report_lines), encoding="utf-8")

    failure_examples = gather_failure_examples(detail_rows)
    failure_lines = ["# Non-OOD Failure Slices", ""]
    for family in sorted(failure_examples.keys()):
        failure_lines.append(f"## {family}")
        failure_lines.append("")
        for row in failure_examples[family]:
            failure_lines.append(f"- Persona `{row['persona_id']}` step `{row['change_step']}` root cause `{row['root_cause']}` correct=`{row['is_correct']}`")
            failure_lines.append(f"  Query: {sanitize_text(row['user_query'])[:220]}")
            failure_lines.append(f"  Prev -> Current: {sanitize_text(row['prev_pref'])[:120]} -> {sanitize_text(row['preference'])[:120]}")
            failure_lines.append(f"  Plausible patch: `{row['plausible_patch_id']}` score `{row['plausible_patch_score']}` cached_top `{row['cached_top_patch_ids']}` injected `{row['injected_patch_ids']}`")
            if row["robust_best_context"]:
                failure_lines.append(f"  Robust context: {sanitize_text(row['robust_best_context'])[:260]}")
            failure_lines.append("")
    (analysis_dir / "nonood_failure_slices.md").write_text("\n".join(failure_lines), encoding="utf-8")

    runtime_lines = [
        "# Runtime Mismatch Notes",
        "",
        f"- Evaluated run label: `{args.run_label}`",
        f"- Results glob matched `{len(result_rows)}` rows.",
        f"- Patch cache root exists: `{args.patch_cache_root.exists()}`",
        "",
        "## Observations",
        "",
        f"- The run appears to rely on prebuilt caches under `{args.patch_cache_root.name}` rather than rebuilding during evaluation.",
        "- Cached `patch_index_records.jsonl` are treated as the source of truth for what was actually indexed during the run.",
        "- The current offline replay also embeds rebuilt structured-first patch docs; differences between cached-doc recall and rebuilt-doc recall quantify stale/index-text effects.",
        "- `test_persona_patch.py` had `llm_patch_filter` objects threaded into the lower layers but not into `evaluate_persona_benchmark()` or the CLI path; this implementation fixes that wiring for future experiments.",
        "- Current result CSVs do not include debug retrieval keyword columns, so exact runtime keyword generation cannot be replayed from artifacts alone.",
    ]
    (analysis_dir / "runtime_mismatch_notes.md").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "run_label": args.run_label,
        "benchmark_file": str(args.benchmark_file),
        "results_glob": args.results_glob,
        "rows": len(detail_rows),
        "root_cause_counts": dict(root_cause_stats),
        "family_accuracy": {family: {"correct": stats["correct"], "total": stats["total"]} for family, stats in family_stats.items()},
    }
    (analysis_dir / "nonood_patch_capture_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {detail_csv}")
    print(f"Wrote {analysis_dir / 'persona_patch_run_manifest.md'}")
    print(f"Wrote {analysis_dir / 'nonood_patch_family_report.md'}")
    print(f"Wrote {analysis_dir / 'nonood_failure_slices.md'}")
    print(f"Wrote {analysis_dir / 'runtime_mismatch_notes.md'}")


if __name__ == "__main__":
    main()
