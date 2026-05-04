"""
Post-process existing Persona patch cache to apply quality gating and temporal ordering
WITHOUT re-running ingestion.

What this script does per sample:
  1. Reads all patch JSON files from the source cache
  2. FILTERS OUT noise patches (assistant-triggered, no preference signal)
  3. LABELS each kept patch as ADD_PREFERENCE, UPDATE_PREFERENCE, or REVOKE_PREFERENCE
  4. Writes filtered patches + updated patch_index_records.jsonl (with temporal_order field)
     to a new destination cache directory
  5. Does NOT copy retriever pkl/npy — the retriever will be rebuilt from index_records
     on the first QA run
  6. Copies global_graph/ and build_status.json unchanged

Usage:
    python scripts/postprocess_persona_patches.py \\
        --src cached_memories_persona_patch_openai_gpt-5.4-mini-2026-03-17_32k_always \\
        --dst cached_memories_persona_patch_openai_gpt-5.4-mini-2026-03-17_32k_always_postproc \\
        [--dry-run]

Heuristics used for filtering/labelling:
  - trigger speaker == "assistant"          → NOISE (discard)
  - trigger text contains "please forget"   → REVOKE_PREFERENCE (keep)
  - trigger text contains "no longer" / "i don't" / "stop" (preference revoke variants)
                                            → REVOKE_PREFERENCE (keep)
  - trigger speaker == "user" + any pref keyword present
                                            → ADD_PREFERENCE or UPDATE_PREFERENCE (keep)
  - trigger speaker == "user" + no pref keyword but summary mentions preference
                                            → keep as UPDATE_PREFERENCE
  - trigger speaker == "user" + truly task-only (code/rewrite/translate command)
                                            → NOISE (discard)
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Heuristic classifiers
# ---------------------------------------------------------------------------

REVOKE_PHRASES = [
    "please forget",
    "forget that i",
    "no longer prefer",
    "no longer like",
    "i don't enjoy",
    "i don't like",
    "i stopped",
    "i've stopped",
    "i changed my mind",
    "i'm not into",
    "i'm no longer",
]

PREFERENCE_KEYWORDS = [
    "prefer", "preference", "like", "enjoy", "love", "hate", "dislike",
    "favor", "favourite", "favorite", "tend to", "usually", "always",
    "habit", "hobby", "interest", "passion", "allergic", "condition",
    "sensitive", "diet", "avoid", "never", "health", "lifestyle",
]

TASK_ONLY_PATTERNS = [
    r"^(please )?(translate|rewrite|rephrase|summarize|summarise|edit|fix|correct|improve|shorten|expand|write|draft|generate|create|make|convert|format)\b",
    r"^(can you|could you|please) (translate|rewrite|rephrase|summarize|write|draft|generate|create|make|convert)\b",
    r"here is (a |the )?(translation|rewrite|revised|edited|corrected|improved)",
]
TASK_ONLY_RE = [re.compile(p, re.IGNORECASE) for p in TASK_ONLY_PATTERNS]


def classify_patch(patch: dict) -> tuple[str, bool]:
    """
    Returns (patch_type, keep).
    patch_type: one of ADD_PREFERENCE, UPDATE_PREFERENCE, REVOKE_PREFERENCE, NOISE
    keep: True if the patch should be retained in the filtered cache
    """
    trigger = patch.get("trigger_turn", {})
    speaker = str(trigger.get("speaker", "")).lower().strip()
    text = str(trigger.get("text", "")).strip()
    text_lower = text.lower()
    summary = str(patch.get("patch_overall", {}).get("overall_summary", "")).lower()
    signals = " ".join(patch.get("patch_overall", {}).get("selection_signals", [])).lower()

    # 1. Assistant-triggered → always noise
    if speaker == "assistant":
        return "NOISE", False

    # 2. Explicit revoke phrases
    for phrase in REVOKE_PHRASES:
        if phrase in text_lower:
            return "REVOKE_PREFERENCE", True

    # 3. Task-only user messages (translate, rewrite, etc.) with no preference keyword
    is_task_only = any(p.search(text) for p in TASK_ONLY_RE)
    has_pref_kw = any(kw in text_lower for kw in PREFERENCE_KEYWORDS)
    if is_task_only and not has_pref_kw:
        # Double-check: does summary/signals mention preference?
        has_pref_in_meta = any(kw in summary or kw in signals for kw in PREFERENCE_KEYWORDS)
        if not has_pref_in_meta:
            return "NOISE", False

    # 4. User message with preference keywords → keep
    if has_pref_kw:
        # Distinguish add vs update: if summary says "revoked/removed/no longer/forget"
        revoke_in_meta = any(w in summary for w in ["revok", "remov", "no longer", "forget", "retract", "withdraw"])
        if revoke_in_meta:
            return "REVOKE_PREFERENCE", True
        return "ADD_PREFERENCE", True

    # 5. User message with preference signal in summary/signals only
    has_pref_in_meta = any(kw in summary or kw in signals for kw in PREFERENCE_KEYWORDS)
    if has_pref_in_meta:
        return "UPDATE_PREFERENCE", True

    # 6. Default: unknown user message, keep but label generically
    return "UPDATE_PREFERENCE", True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def postprocess_sample(src_sample: Path, dst_sample: Path, dry_run: bool) -> dict:
    stats = Counter()

    # Read all patch JSON files
    src_patches_dir = src_sample / "patches"
    if not src_patches_dir.exists():
        return dict(stats)

    patches = []
    for pf in sorted(src_patches_dir.glob("patch_*.json")):
        with open(pf) as f:
            patches.append((pf, json.load(f)))

    stats["total"] = len(patches)

    # Classify and filter
    kept = []
    for pf, patch in patches:
        patch_type, keep = classify_patch(patch)
        if keep:
            # Inject the classified type into patch_overall
            patch["patch_overall"]["patch_type"] = patch_type
            kept.append((pf, patch))
            stats[f"kept_{patch_type}"] += 1
        else:
            stats["discarded_NOISE"] += 1

    stats["kept_total"] = len(kept)

    if dry_run:
        return dict(stats)

    # Write filtered patches
    dst_patches_dir = dst_sample / "patches"
    dst_patches_dir.mkdir(parents=True, exist_ok=True)
    for pf, patch in kept:
        with open(dst_patches_dir / pf.name, "w") as f:
            json.dump(patch, f, ensure_ascii=False, indent=2)

    # Build new patch_index_records.jsonl with temporal_order
    index_records = []
    for _, patch in kept:
        trigger = patch.get("trigger_turn", {})
        session_id = trigger.get("session_id") or 0
        turn_position = trigger.get("turn_position") or 0
        # Handle string session ids (Persona uses session_id=0 always)
        try:
            sid_int = int(session_id)
        except (ValueError, TypeError):
            sid_int = 0
        try:
            tp_int = int(turn_position)
        except (ValueError, TypeError):
            tp_int = 0

        patch_type = patch["patch_overall"].get("patch_type", "UPDATE_PREFERENCE")
        index_records.append({
            "patch_id": patch["patch_id"],
            "sample_id": patch["sample_id"],
            "session_id": session_id,
            "turn_position": turn_position,
            "temporal_order": f"{sid_int:06d}_{tp_int:08d}",
            "patch_type": patch_type,
            "index_document": patch.get("index_document", ""),
        })

    # Sort by temporal order before writing (ensures retriever index order is chronological)
    index_records.sort(key=lambda r: r["temporal_order"])

    with open(dst_sample / "patch_index_records.jsonl", "w") as f:
        for rec in index_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Copy global_graph/ and build_status.json unchanged
    for name in ["global_graph", "build_status.json"]:
        src_path = src_sample / name
        dst_path = dst_sample / name
        if src_path.is_dir():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        elif src_path.is_file():
            shutil.copy2(src_path, dst_path)

    # Do NOT copy patch_retriever.pkl / patch_retriever_embeddings.npy
    # The system will rebuild them from patch_index_records.jsonl on first QA run

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source patch cache directory")
    parser.add_argument("--dst", required=True, help="Destination (new) patch cache directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing anything")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")

    sample_dirs = sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    print(f"Found {len(sample_dirs)} samples in {src_root}")
    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {dst_root}")
    else:
        print("[DRY RUN — no files will be written]")

    totals = Counter()
    for sample_dir in sample_dirs:
        dst_sample = dst_root / sample_dir.name
        stats = postprocess_sample(sample_dir, dst_sample, dry_run=args.dry_run)
        totals.update(stats)

        kept_types = {k: v for k, v in stats.items() if k.startswith("kept_") and k != "kept_total"}
        print(
            f"  {sample_dir.name}: "
            f"{stats.get('total', 0)} patches → "
            f"{stats.get('kept_total', 0)} kept, "
            f"{stats.get('discarded_NOISE', 0)} discarded | "
            + " ".join(f"{k.replace('kept_', '')}={v}" for k, v in sorted(kept_types.items()))
        )

    print(f"\nTotals:")
    print(f"  Total patches:   {totals.get('total', 0)}")
    print(f"  Kept:            {totals.get('kept_total', 0)} "
          f"({totals.get('kept_total', 0) / max(totals.get('total', 1), 1) * 100:.1f}%)")
    print(f"  Discarded noise: {totals.get('discarded_NOISE', 0)}")
    print(f"  REVOKE patches:  {totals.get('kept_REVOKE_PREFERENCE', 0)}")
    print(f"  ADD patches:     {totals.get('kept_ADD_PREFERENCE', 0)}")
    print(f"  UPDATE patches:  {totals.get('kept_UPDATE_PREFERENCE', 0)}")


if __name__ == "__main__":
    main()
