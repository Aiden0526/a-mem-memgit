"""Evaluation harness for patch-augmented A-Mem."""

from memory_layer_robust import RobustLLMController, normalize_openai_compatible_base_url
from memory_layer_patch import PatchAugmentedMemorySystem, PatchConfig
from llm_text_parsers import parse_keywords_response, parse_plain_text_answer
import argparse
import json
import logging
import os
import random
import subprocess
import sys
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm
from load_dataset import load_locomo_dataset
from locomo_eval_utils import get_locomo_prompt_answer, get_locomo_reference_answer
from utils import calculate_locomo_official_metrics, aggregate_metrics

logger = logging.getLogger("amem_patch")

DEFAULT_PPAPI_CHAT_COMPLETIONS_URL = "https://app.ppapi.ai/v1/chat/completions"


def setup_logging(log_file: str | None, level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(resolved_level)
    formatter = logging.Formatter("%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def resolve_api_base(api_base: str | None) -> str | None:
    if api_base:
        return normalize_openai_compatible_base_url(api_base)
    return normalize_openai_compatible_base_url(
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("PPAPI_BASE_URL")
        or DEFAULT_PPAPI_CHAT_COMPLETIONS_URL
    )


def resolve_local_backend_endpoint(
    backend: str,
    sglang_host: str,
    sglang_port: int,
    vllm_host: str | None,
    vllm_port: int | None,
) -> tuple[str, int]:
    if backend == "vllm":
        return vllm_host or sglang_host, vllm_port or sglang_port
    return sglang_host, sglang_port


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def build_patch_result_key(sample_id: Any, qa_index: int, category: int, question: str) -> str:
    return json.dumps([str(sample_id), int(qa_index), int(category), question], ensure_ascii=False)


def load_existing_patch_result(inventory_output: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], defaultdict[int, int], set[str]]:
    if not inventory_output or not os.path.exists(inventory_output):
        return [], [], defaultdict(int), set()
    try:
        with open(inventory_output, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return [], [], defaultdict(int), set()
    qa_results = payload.get('qa_results', []) if isinstance(payload, dict) else []
    sample_patch_summaries = payload.get('sample_patch_summaries', []) if isinstance(payload, dict) else []
    category_counts = defaultdict(int)
    completed_result_keys: set[str] = set()
    for row in qa_results:
        category = int(row.get('category', 0))
        category_counts[category] += 1
        completed_result_keys.add(
            build_patch_result_key(
                row.get('sample_id'),
                int(row.get('qa_index', 0)),
                category,
                row.get('question', ''),
            )
        )
    return qa_results, sample_patch_summaries, category_counts, completed_result_keys


def persist_patch_result(
    inventory_output: str | None,
    model: str,
    dataset_path: str,
    patch_usage: str,
    samples: int,
    category_counts: defaultdict[int, int],
    qa_results: list[dict[str, Any]],
    sample_patch_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    all_metrics = [row['metrics'] for row in qa_results if row.get('metrics') is not None]
    all_categories = [row['category'] for row in qa_results if row.get('metrics') is not None and row.get('category') is not None]
    aggregate_results = aggregate_metrics(all_metrics, all_categories) if all_metrics else {}
    result = {
        'model': model,
        'dataset': dataset_path,
        'memory_layer': 'patch',
        'patch_usage': patch_usage,
        'samples': samples,
        'total_questions': len(qa_results),
        'category_counts': dict(category_counts),
        'aggregate_metrics': aggregate_results,
        'sample_patch_summaries': sample_patch_summaries,
        'qa_results': qa_results,
    }
    if inventory_output:
        atomic_write_json(inventory_output, result)
    return result


def flatten_locomo_sample_turns(sample) -> list[tuple[int, Any, int, Any]]:
    turn_entries = []
    for session_id, turns in sample.conversation.sessions.items():
        for turn_position, turn in enumerate(turns.turns):
            turn_entries.append((session_id, turns, turn_position, turn))
    return turn_entries


def resolve_patch_resume_index(memory_system: PatchAugmentedMemorySystem, turn_entries: list[tuple[int, Any, int, Any]]) -> int:
    status = memory_system.get_build_status()
    last_session_id = status.get('last_session_id')
    last_turn_position = status.get('last_turn_position')
    if last_session_id is not None and last_turn_position is not None:
        for idx, (session_id, _turns, turn_position, _turn) in enumerate(turn_entries):
            if session_id == last_session_id and turn_position == last_turn_position:
                return idx + 1
    return min(memory_system.get_resume_turn_index(), len(turn_entries))


class PatchAdvancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 patch_top_k=2, patch_usage="always", api_key=None, api_base=None,
                 min_patch_similarity=0.0, patch_node_rerank=False, patch_node_top_k=2,
                 patch_node_query_mode="expanded", patch_hybrid_retrieval=False,
                 patch_hybrid_alpha=0.7, patch_hybrid_node_rerank=False,
                 cache_root=None):
        self.memory_system = PatchAugmentedMemorySystem(
            sample_id="0",
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            store_root=cache_root,
            config=PatchConfig(
                patch_top_k=patch_top_k,
                retrieve_k_current=retrieve_k,
                patch_usage=patch_usage,
                min_patch_similarity=min_patch_similarity,
                patch_node_rerank=patch_node_rerank,
                patch_node_top_k=patch_node_top_k,
                patch_node_query_mode=patch_node_query_mode,
                patch_hybrid_retrieval=patch_hybrid_retrieval,
                patch_hybrid_alpha=patch_hybrid_alpha,
                patch_hybrid_node_rerank=patch_hybrid_node_rerank,
            ),
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def set_sample(self, sample_id: str):
        self.memory_system.set_sample(sample_id)

    def add_memory(self, content, time=None, **kwargs):
        self.memory_system.ingest_turn_with_patch_history(content, time=time, **kwargs)

    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_keywords_response(response)

    def answer_question(self, question: str, category: int, answer: str):
        if self.memory_system.config.patch_usage == "gated":
            return self.memory_system.answer_with_patch_history_gated(
                question,
                category,
                answer,
                temperature_c5=self.temperature_c5,
            )
        return self.memory_system.answer_with_patch_history(
            question,
            category,
            answer,
            temperature_c5=self.temperature_c5,
        )


def evaluate_dataset(dataset_path: str, model: str, ratio: float = 1.0,
                     backend: str = "sglang", temperature_c5: float = 0.5,
                     retrieve_k: int = 10, patch_top_k: int = 5,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     vllm_host: str | None = None, vllm_port: int | None = None,
                     max_samples: int | None = None,
                     start_sample: int = 0,
                     end_sample: int | None = None,
                     num_workers: int = 1,
                     worker_id: int = 0,
                     skip_qa: bool = False,
                     api_key: str | None = None,
                     api_base: str | None = None,
                     inventory_output: str | None = None,
                     patch_usage: str = "always",
                     min_patch_similarity: float = 0.0,
                     patch_node_rerank: bool = False,
                     patch_node_top_k: int = 2,
                     patch_node_query_mode: str = "expanded",
                     patch_hybrid_retrieval: bool = False,
                     patch_hybrid_alpha: float = 0.7,
                     patch_hybrid_node_rerank: bool = False,
                     cache_root: str | None = None,
                     log_file: str | None = None):
    if log_file:
        logger.info("logging to %s", log_file)
    samples = load_locomo_dataset(dataset_path)
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
    if start_sample < 0:
        raise ValueError("start_sample must be >= 0")
    if end_sample is not None and end_sample < start_sample:
        raise ValueError("end_sample must be >= start_sample")
    samples = samples[start_sample:end_sample]
    if max_samples is not None:
        samples = samples[:max_samples]
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")
    if num_workers > 1:
        samples = [sample for sample_idx, sample in enumerate(samples)
                   if sample_idx % num_workers == worker_id]

    local_host, local_port = resolve_local_backend_endpoint(
        backend,
        sglang_host,
        sglang_port,
        vllm_host,
        vllm_port,
    )

    agent = PatchAdvancedMemAgent(
        model, backend, retrieve_k, temperature_c5,
        local_host, local_port, patch_top_k, patch_usage, api_key, api_base,
        min_patch_similarity=min_patch_similarity,
        patch_node_rerank=patch_node_rerank,
        patch_node_top_k=patch_node_top_k,
        patch_node_query_mode=patch_node_query_mode,
        patch_hybrid_retrieval=patch_hybrid_retrieval,
        patch_hybrid_alpha=patch_hybrid_alpha,
        patch_hybrid_node_rerank=patch_hybrid_node_rerank,
        cache_root=cache_root,
    )
    qa_results, sample_patch_summaries, category_counts, completed_result_keys = load_existing_patch_result(inventory_output)
    total_questions = len(qa_results)
    total_qas_in_run = sum(len(sample.qa) for sample in samples) if not skip_qa else 0

    sample_progress = tqdm(samples, desc="Samples", unit="sample")
    for sample_idx, sample in enumerate(sample_progress):
        sample_progress.set_postfix_str(f"sample_id={sample.sample_id}")
        agent.set_sample(sample.sample_id)
        if agent.memory_system.has_complete_global_graph_cache():
            logger.info("sample=%s loaded complete global graph cache; skipping rebuild", sample.sample_id)
        else:
            turn_entries = flatten_locomo_sample_turns(sample)
            resume_turn_index = resolve_patch_resume_index(agent.memory_system, turn_entries)
            if resume_turn_index > 0:
                logger.info(
                    "sample=%s resuming patch build from turn %s/%s",
                    sample.sample_id,
                    resume_turn_index,
                    len(turn_entries),
                )
            turn_progress = tqdm(
                total=len(turn_entries),
                initial=resume_turn_index,
                desc=f"Build {sample.sample_id}",
                unit="turn",
                leave=False,
            )
            for global_turn_index, (session_id, turns, turn_position, turn) in enumerate(turn_entries[resume_turn_index:], start=resume_turn_index):
                conversation_tmp = "Speaker " + turn.speaker + " says : " + turn.text
                agent.add_memory(
                    conversation_tmp,
                    time=turns.date_time,
                    session_id=session_id,
                    session_date_time=turns.date_time,
                    session_summary=sample.session_summary.get(f"session_{session_id}_summary", ""),
                    turn_position=turn_position,
                    turn_number=global_turn_index + 1,
                    dia_id=turn.dia_id,
                    speaker=turn.speaker,
                )
                turn_progress.update(1)
            turn_progress.close()
            agent.memory_system.mark_sample_complete()

        patch_summary = agent.memory_system.summarize_patch_inventory()
        patch_dir = agent.memory_system.store.patches_dir(sample.sample_id)
        patch_files = sorted(str(p.name) for p in patch_dir.glob("patch_*.json"))
        patch_summary["patch_dir"] = str(patch_dir)
        patch_summary["patch_files"] = patch_files
        sample_patch_summaries = [s for s in sample_patch_summaries if str(s.get('sample_id')) != str(sample.sample_id)]
        sample_patch_summaries.append(patch_summary)
        logger.info(
            "patch_inventory sample=%s patches=%s types=%s avg_changed_nodes=%.2f max_changed_nodes=%s sessions=%s patch_dir=%s patch_files=%s",
            sample.sample_id,
            patch_summary["patch_count"],
            patch_summary["patch_type_counts"],
            patch_summary["avg_changed_nodes"],
            patch_summary["max_changed_nodes"],
            patch_summary["session_patch_counts"],
            patch_summary["patch_dir"],
            patch_summary["patch_files"],
        )
        persist_patch_result(
            inventory_output,
            model,
            dataset_path,
            patch_usage,
            len(samples),
            category_counts,
            qa_results,
            sample_patch_summaries,
        )

        if skip_qa:
            continue

        qa_progress = tqdm(sample.qa, desc=f"QA {sample.sample_id}", unit="qa", leave=False)
        for qa_idx, qa in enumerate(qa_progress):
            qa_key = build_patch_result_key(sample.sample_id, qa_idx, int(qa.category), qa.question)
            if qa_key in completed_result_keys:
                continue

            total_questions += 1
            qa_progress.set_postfix_str(f"global={total_questions}/{total_qas_in_run} cat={qa.category}")
            logger.info(
                "qa_progress sample=%s qa=%s/%s sample_qa=%s/%s category=%s question=%s",
                sample.sample_id,
                total_questions,
                total_qas_in_run,
                qa_idx + 1,
                len(sample.qa),
                qa.category,
                qa.question,
            )
            category_counts[qa.category] += 1
            prompt_answer = get_locomo_prompt_answer(qa.category, qa.answer, qa.adversarial_answer)
            reference_answer = get_locomo_reference_answer(qa.category, qa.answer)
            prediction, user_prompt, raw_context, answer_metadata = agent.answer_question(
                qa.question, qa.category, prompt_answer
            )
            prediction = parse_plain_text_answer(prediction)
            metrics = calculate_locomo_official_metrics(prediction, reference_answer, qa.category)
            qa_results.append({
                "sample_index": sample_idx,
                "sample_id": sample.sample_id,
                "qa_index": qa_idx,
                "category": qa.category,
                "question": qa.question,
                "prediction": prediction,
                "reference": reference_answer,
                "adversarial_answer": qa.adversarial_answer,
                "user_prompt": user_prompt,
                "raw_context": raw_context,
                "metrics": metrics,
                "answer_metadata": answer_metadata,
                "evaluation_protocol": "official_locomo",
            })
            completed_result_keys.add(qa_key)
            persist_patch_result(
                inventory_output,
                model,
                dataset_path,
                patch_usage,
                len(samples),
                category_counts,
                qa_results,
                sample_patch_summaries,
            )
            logger.info("sample=%s category=%s question=%s prediction=%s", sample_idx, qa.category, qa.question, prediction)
            logger.debug("prompt=%s", user_prompt)
            logger.debug("context=%s", raw_context)
        qa_progress.close()

    sample_progress.close()
    result = persist_patch_result(
        inventory_output,
        model,
        dataset_path,
        patch_usage,
        len(samples),
        category_counts,
        qa_results,
        sample_patch_summaries,
    )
    return result


def _worker_inventory_output_path(inventory_output: str | None, worker_id: int) -> str | None:
    if not inventory_output:
        return None
    root, ext = os.path.splitext(inventory_output)
    suffix = ext or ".json"
    return f"{root}.worker_{worker_id}{suffix}"


def _merge_inventory_results(worker_results: list[dict]) -> dict:
    category_counts = defaultdict(int)
    sample_patch_summaries = []
    qa_results = []
    all_metrics = []
    all_categories = []
    total_samples = 0
    total_questions = 0
    model = worker_results[0].get("model") if worker_results else None
    dataset = worker_results[0].get("dataset") if worker_results else None
    patch_usage = worker_results[0].get("patch_usage", "always") if worker_results else "always"
    for result in worker_results:
        total_samples += result.get("samples", 0)
        total_questions += result.get("total_questions", 0)
        for category, count in result.get("category_counts", {}).items():
            category_counts[int(category)] += count
        sample_patch_summaries.extend(result.get("sample_patch_summaries", []))
        worker_qa_results = result.get("qa_results", [])
        qa_results.extend(worker_qa_results)
        for qa_result in worker_qa_results:
            metrics = qa_result.get("metrics")
            category = qa_result.get("category")
            if metrics is not None and category is not None:
                all_metrics.append(metrics)
                all_categories.append(category)
    aggregate_results = aggregate_metrics(all_metrics, all_categories) if all_metrics else {}
    return {
        "model": model,
        "dataset": dataset,
        "memory_layer": "patch",
        "patch_usage": patch_usage,
        "samples": total_samples,
        "total_questions": total_questions,
        "category_counts": dict(category_counts),
        "aggregate_metrics": aggregate_results,
        "sample_patch_summaries": sample_patch_summaries,
        "qa_results": qa_results,
    }


def run_batch_workers(args) -> dict:
    if args.batch < 1:
        raise ValueError("batch must be at least 1")

    child_base = [
        sys.executable,
        os.path.abspath(__file__),
        "--dataset", args.dataset,
        "--model", args.model,
        "--backend", args.backend,
        "--ratio", str(args.ratio),
        "--retrieve_k", str(args.retrieve_k),
        "--patch_top_k", str(args.patch_top_k),
        "--patch_usage", args.patch_usage,
        "--temperature_c5", str(args.temperature_c5),
        "--min_patch_similarity", str(args.min_patch_similarity),
        "--patch_node_top_k", str(args.patch_node_top_k),
        "--patch_node_query_mode", args.patch_node_query_mode,
        "--patch_hybrid_alpha", str(args.patch_hybrid_alpha),
        "--sglang_host", args.sglang_host,
        "--sglang_port", str(args.sglang_port),
        "--vllm_host", args.vllm_host,
        "--vllm_port", str(args.vllm_port),
        "--start_sample", str(args.start_sample),
        "--num_workers", str(args.batch),
    ]

    if args.end_sample is not None:
        child_base.extend(["--end_sample", str(args.end_sample)])
    if args.max_samples is not None:
        child_base.extend(["--max_samples", str(args.max_samples)])
    if args.skip_qa:
        child_base.append("--skip_qa")
    if args.patch_node_rerank:
        child_base.append("--patch_node_rerank")
    if args.patch_hybrid_retrieval:
        child_base.append("--patch_hybrid_retrieval")
    if args.patch_hybrid_node_rerank:
        child_base.append("--patch_hybrid_node_rerank")
    if args.api_key is not None:
        child_base.extend(["--api_key", args.api_key])
    if args.api_base is not None:
        child_base.extend(["--api_base", args.api_base])

    processes = []
    worker_outputs = []
    for worker_id in range(args.batch):
        worker_output = _worker_inventory_output_path(args.inventory_output, worker_id)
        cmd = child_base + ["--worker_id", str(worker_id)]
        if worker_output is not None:
            cmd.extend(["--inventory_output", worker_output])
        if args.cache_root:
            cmd.extend(["--cache_root", args.cache_root])
        logger.info("launching worker_id=%s command=%s", worker_id, cmd)
        processes.append((worker_id, worker_output, subprocess.Popen(cmd)))
        worker_outputs.append(worker_output)

    failures = []
    for worker_id, worker_output, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((worker_id, return_code))
            logger.error("worker_id=%s failed return_code=%s", worker_id, return_code)
        else:
            logger.info("worker_id=%s completed inventory_output=%s", worker_id, worker_output)

    if failures:
        failed = ", ".join(f"worker {worker_id} (exit {return_code})" for worker_id, return_code in failures)
        raise RuntimeError(f"batch run failed: {failed}")

    worker_results = []
    for worker_output in worker_outputs:
        if worker_output is None:
            continue
        with open(worker_output, "r", encoding="utf-8") as f:
            worker_results.append(json.load(f))

    merged_result = _merge_inventory_results(worker_results)
    if args.inventory_output:
        with open(args.inventory_output, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=2)
    return merged_result


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Patch-augmented A-Mem evaluation")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "openrouter", "ollama", "sglang", "vllm"])
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--patch_top_k", type=int, default=2)
    parser.add_argument("--patch_usage", type=str, default="always", choices=["always", "gated"])
    parser.add_argument("--min_patch_similarity", type=float, default=0.0)
    parser.add_argument("--patch_node_rerank", action="store_true")
    parser.add_argument("--patch_node_top_k", type=int, default=2)
    parser.add_argument("--patch_node_query_mode", type=str, default="expanded",
                        choices=["question_only", "keywords_only", "expanded", "answer_style"])
    parser.add_argument("--patch_hybrid_retrieval", action="store_true")
    parser.add_argument("--patch_hybrid_alpha", type=float, default=0.7)
    parser.add_argument("--patch_hybrid_node_rerank", action="store_true")
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--vllm_host", type=str, default="http://localhost",
                        help="vLLM server host when --backend vllm")
    parser.add_argument("--vllm_port", type=int, default=8000,
                        help="vLLM server port when --backend vllm")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--end_sample", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--skip_qa", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None,
                        help="OpenAI-compatible API base or full /chat/completions endpoint. Defaults to OPENAI_BASE_URL, PPAPI_BASE_URL, or ppapi.")
    parser.add_argument("--inventory_output", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Alias for --inventory_output.")
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()
    if args.output and not args.inventory_output:
        args.inventory_output = args.output
    args.api_base = resolve_api_base(args.api_base)

    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        worker_suffix = f"_worker{args.worker_id}" if args.num_workers > 1 else ""
        args.log_file = str(Path("logs") / f"eval_patch_{args.model}_{args.backend}_ratio{args.ratio}{worker_suffix}_{timestamp}.log")
    setup_logging(args.log_file, args.log_level)
    if args.batch is not None:
        if args.batch == 1:
            args.num_workers = 1
            args.worker_id = 0
        else:
            run_batch_workers(args)
            sys.exit(0)

    evaluate_dataset(
        dataset_path=args.dataset,
        model=args.model,
        ratio=args.ratio,
        backend=args.backend,
        temperature_c5=args.temperature_c5,
        retrieve_k=args.retrieve_k,
        patch_top_k=args.patch_top_k,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        max_samples=args.max_samples,
        start_sample=args.start_sample,
        end_sample=args.end_sample,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        skip_qa=args.skip_qa,
        api_key=args.api_key,
        api_base=args.api_base,
        inventory_output=args.inventory_output,
        cache_root=args.cache_root,
        log_file=args.log_file,
        patch_usage=args.patch_usage,
        min_patch_similarity=args.min_patch_similarity,
        patch_node_rerank=args.patch_node_rerank,
        patch_node_top_k=args.patch_node_top_k,
        patch_node_query_mode=args.patch_node_query_mode,
        patch_hybrid_retrieval=args.patch_hybrid_retrieval,
        patch_hybrid_alpha=args.patch_hybrid_alpha,
        patch_hybrid_node_rerank=args.patch_hybrid_node_rerank,
    )
