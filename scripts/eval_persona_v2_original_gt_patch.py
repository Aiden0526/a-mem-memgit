from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from memory_layer_robust import RobustLLMController
from test_persona_robust import (
    check_mcq_correctness,
    create_mcq_options,
    extract_final_answer,
    normalize_openai_env,
    parse_incorrect_answers,
    parse_user_query,
    setup_logger,
    stable_int_seed,
)

csv.field_size_limit(1 << 30)


def sanitize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_gt_patch_block(row: Dict[str, Any], max_snippet_chars: int = 900) -> str:
    snippet = sanitize_text(row.get("related_conversation_snippet", ""))
    if len(snippet) > max_snippet_chars:
        snippet = snippet[:max_snippet_chars].rstrip() + " ..."
    return (
        "[Ground-Truth Preference Patch]\n"
        f"Earlier preference: {sanitize_text(row.get('prev_pref', ''))}\n"
        f"Later preference: {sanitize_text(row.get('preference', ''))}\n"
        f"Evidence snippet: {snippet}\n\n"
        "Interpretation rule: do not assume the later preference is automatically the answer. "
        "Treat condition and time phrases inside the preference itself as critical evidence. "
        "Phrases like when dining alone, on weekends, in the morning, during spring, on weekdays, or when at home define when that preference applies. "
        "Decide whether the question matches the earlier preference, the later preference, or the transition between them."
    )


def build_prompt(question: str, option_mapping: Dict[str, str], row: Dict[str, Any]) -> str:
    options_text = "\n".join(f"{k}. {v}" for k, v in option_mapping.items())
    patch_block = build_gt_patch_block(row)
    return f"""You are answering a PersonaMem-v2 benchmark question using a ground-truth preference patch.

Use the patch below as high-confidence evidence about the user's preference change.

Reasoning rules:
1. Do not assume the later preference automatically wins.
2. First identify whether the question is asking about a general preference, a condition-specific preference, or a time-specific preference.
3. Treat condition and time phrases inside the preference itself as critical evidence.
4. Keep two notions separate: change over time in the patch, and condition/time scope inside the preference text.
5. Choose the option best supported by the patch evidence, whether that support is explicit or implicit in the wording.

{patch_block}

Question:
{question}

Options:
{options_text}

Before choosing, briefly determine:
- what condition or time scope the question is asking about
- whether that matches the earlier preference, the later preference, or neither
- which option is best supported by that matched state

Then end with exactly one line in this format:
Final Answer: <LETTER>"""


def default_output_path(model: str, updated_only: bool) -> Path:
    tag = "updated_only" if updated_only else "all_rows"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return Path("patch_results") / f"persona_v2_original_gt_patch_{tag}_{model}_{timestamp}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PersonaMem-v2 original benchmark with benchmark-derived gold patches")
    parser.add_argument("--benchmark_file", type=str, default="data/Persona-V2-original/benchmark/text/benchmark.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "openrouter", "ollama", "sglang", "vllm"])
    parser.add_argument("--api_base", type=str, default=None,
                        help="Optional OpenAI-compatible API base URL. Useful for custom OpenRouter-compatible endpoints.")
    parser.add_argument("--updated_only", action="store_true", default=False)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--answer_temperature", type=float, default=0.0)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    args = parser.parse_args()

    normalize_openai_env()

    benchmark_file = Path(args.benchmark_file)
    output_path = Path(args.output) if args.output else default_output_path(args.model, args.updated_only)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / f"persona_v2_original_gt_patch_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"
    logger = setup_logger(log_path)

    df = pd.read_csv(benchmark_file)
    if args.updated_only:
        mask = df["updated"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[mask].copy()
    if args.max_items is not None:
        df = df.head(args.max_items).copy()
    df = df.reset_index().rename(columns={"index": "original_row_index"})

    llm = RobustLLMController(
        backend=args.backend,
        model=args.model,
        api_key=None,
        api_base=args.api_base,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    ).llm

    logger.info("Loaded %d rows from %s", len(df), benchmark_file)
    logger.info("updated_only=%s", args.updated_only)

    out_rows: List[Dict[str, Any]] = []
    correct = 0
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        user_query = parse_user_query(row_dict.get("user_query", ""))
        question = user_query.get("content", "")
        _, option_mapping, correct_letter = create_mcq_options(
            row_dict.get("correct_answer", ""),
            parse_incorrect_answers(row_dict.get("incorrect_answers", "")),
            seed=stable_int_seed(row_dict.get("persona_id", ""), row_dict.get("user_query", "")),
        )
        prompt = build_prompt(question, option_mapping, row_dict)
        response = llm.get_completion(prompt, temperature=args.answer_temperature, max_tokens=2048)
        prediction = extract_final_answer(str(response))
        is_correct = check_mcq_correctness(prediction, row_dict.get("correct_answer", ""), option_mapping)
        correct += int(is_correct)
        out = dict(row_dict)
        out["predicted_answer_mcq_gt_patch"] = prediction
        out["is_correct_mcq_gt_patch"] = str(is_correct)
        out["correct_mcq_option_gt_patch"] = correct_letter
        out["model_response_mcq_gt_patch"] = str(response)
        out["raw_input_prompt_mcq_gt_patch"] = prompt
        out_rows.append(out)

    pd.DataFrame(out_rows).to_csv(output_path, index=False)
    acc = correct / len(out_rows) if out_rows else 0.0
    logger.info("Accuracy: %.4f (%d/%d)", acc, correct, len(out_rows))
    logger.info("Results saved to %s", output_path)
    print(json.dumps({"rows": len(out_rows), "accuracy": acc, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
