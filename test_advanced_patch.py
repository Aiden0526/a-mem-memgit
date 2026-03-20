"""Evaluation harness for patch-augmented A-Mem."""

from memory_layer_robust import RobustLLMController
from memory_layer_patch import PatchAugmentedMemorySystem, PatchConfig
from llm_text_parsers import parse_keywords_response
import argparse
import json
import logging
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from load_dataset import load_locomo_dataset

logger = logging.getLogger("amem_patch")


class PatchAdvancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 patch_top_k=2, api_key=None, api_base=None):
        self.memory_system = PatchAugmentedMemorySystem(
            sample_id="0",
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            config=PatchConfig(
                patch_top_k=patch_top_k,
                retrieve_k_current=retrieve_k,
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
        return self.memory_system.answer_with_patch_history(question, category, answer)


def evaluate_dataset(dataset_path: str, model: str, ratio: float = 1.0,
                     backend: str = "sglang", temperature_c5: float = 0.5,
                     retrieve_k: int = 10, patch_top_k: int = 5,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     max_samples: int | None = None,
                     num_workers: int = 1,
                     worker_id: int = 0,
                     skip_qa: bool = False,
                     api_key: str | None = None,
                     api_base: str | None = None,
                     inventory_output: str | None = None):
    samples = load_locomo_dataset(dataset_path)
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
    if max_samples is not None:
        samples = samples[:max_samples]
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")
    if num_workers > 1:
        samples = [sample for sample_idx, sample in enumerate(samples)
                   if sample_idx % num_workers == worker_id]

    agent = PatchAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                  sglang_host, sglang_port, patch_top_k, api_key, api_base)
    category_counts = defaultdict(int)

    sample_patch_summaries = []
    qa_results = []

    for sample_idx, sample in enumerate(samples):
        agent.set_sample(sample.sample_id)
        if agent.memory_system.has_complete_global_graph_cache():
            logger.info("sample=%s loaded complete global graph cache; skipping rebuild", sample.sample_id)
        else:
            for session_id, turns in sample.conversation.sessions.items():
                for turn_position, turn in enumerate(turns.turns):
                    conversation_tmp = "Speaker " + turn.speaker + " says : " + turn.text
                    agent.add_memory(
                        conversation_tmp,
                        time=turns.date_time,
                        session_id=session_id,
                        session_date_time=turns.date_time,
                        session_summary=sample.session_summary.get(f"session_{session_id}_summary", ""),
                        turn_position=turn_position,
                        turn_number=turn_position + 1,
                        dia_id=turn.dia_id,
                        speaker=turn.speaker,
                    )
            agent.memory_system.mark_sample_complete()

        patch_summary = agent.memory_system.summarize_patch_inventory()
        patch_dir = agent.memory_system.store.patches_dir(sample.sample_id)
        patch_files = sorted(str(p.name) for p in patch_dir.glob("patch_*.json"))
        patch_summary["patch_dir"] = str(patch_dir)
        patch_summary["patch_files"] = patch_files
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

        if skip_qa:
            continue

        for qa_idx, qa in enumerate(sample.qa):
            category_counts[qa.category] += 1
            prediction, user_prompt, raw_context = agent.answer_question(
                qa.question, qa.category, qa.final_answer
            )
            qa_results.append({
                "sample_index": sample_idx,
                "sample_id": sample.sample_id,
                "qa_index": qa_idx,
                "category": qa.category,
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.final_answer,
                "user_prompt": user_prompt,
                "raw_context": raw_context,
            })
            logger.info("sample=%s category=%s question=%s prediction=%s", sample_idx, qa.category, qa.question, prediction)
            logger.debug("prompt=%s", user_prompt)
            logger.debug("context=%s", raw_context)

    result = {
        "samples": len(samples),
        "category_counts": dict(category_counts),
        "sample_patch_summaries": sample_patch_summaries,
        "qa_results": qa_results,
    }
    if inventory_output:
        with open(inventory_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
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
    total_samples = 0
    for result in worker_results:
        total_samples += result.get("samples", 0)
        for category, count in result.get("category_counts", {}).items():
            category_counts[int(category)] += count
        sample_patch_summaries.extend(result.get("sample_patch_summaries", []))
    return {
        "samples": total_samples,
        "category_counts": dict(category_counts),
        "sample_patch_summaries": sample_patch_summaries,
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
        "--temperature_c5", str(args.temperature_c5),
        "--sglang_host", args.sglang_host,
        "--sglang_port", str(args.sglang_port),
        "--num_workers", str(args.batch),
    ]

    if args.max_samples is not None:
        child_base.extend(["--max_samples", str(args.max_samples)])
    if args.skip_qa:
        child_base.append("--skip_qa")
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
    parser = argparse.ArgumentParser(description="Patch-augmented A-Mem evaluation")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--model", type=str, default="gemini3-flash-preview")
    parser.add_argument("--backend", type=str, default="openrouter")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--patch_top_k", type=int, default=2)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--skip_qa", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--inventory_output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
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
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        skip_qa=args.skip_qa,
        api_key=args.api_key,
        api_base=args.api_base,
        inventory_output=args.inventory_output,
    )
