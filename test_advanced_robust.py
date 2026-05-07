"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import json
import argparse
import logging
from typing import Any, List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import subprocess
import sys
from tqdm import tqdm
from utils import calculate_locomo_official_metrics, aggregate_metrics
from locomo_eval_utils import build_locomo_answer_prompt, locomo_answer_temperature, get_locomo_prompt_answer, get_locomo_reference_answer
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")


def resolve_api_base(api_base: Optional[str]) -> Optional[str]:
    from memory_layer_robust import normalize_openai_compatible_base_url

    if api_base:
        return normalize_openai_compatible_base_url(api_base)
    return normalize_openai_compatible_base_url(
        os.getenv("OPENAI_BASE_URL") or os.getenv("PPAPI_BASE_URL")
    )


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
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

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        user_prompt = build_locomo_answer_prompt(question, category, answer, context)
        temperature = locomo_answer_temperature(category, self.temperature_c5)

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.error("answer_question failed: %s", e)
            raise
        if response is None:
            raise RuntimeError("answer_question received None response from LLM backend")
        return response, user_prompt, raw_context


def setup_logger(log_file: Optional[str] = None, raw_llm_log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    eval_logger = logging.getLogger('locomo_eval_robust')
    backend_logger = logging.getLogger('amem_robust')
    openai_logger = logging.getLogger('openai')

    for named_logger in (eval_logger, backend_logger, openai_logger):
        named_logger.setLevel(logging.INFO)
        named_logger.propagate = False
        named_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)
    backend_logger.addHandler(console_handler)
    openai_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)
        backend_logger.addHandler(file_handler)
        openai_logger.addHandler(file_handler)

    if raw_llm_log_file:
        raw_logger = logging.getLogger('amem_robust_raw')
        raw_logger.setLevel(logging.INFO)
        raw_logger.propagate = False
        raw_logger.handlers.clear()
        raw_file_handler = logging.FileHandler(raw_llm_log_file)
        raw_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        raw_logger.addHandler(raw_file_handler)

    return eval_logger


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def build_locomo_result_key(sample_id: int, qa_index: int, category: int, question: str) -> str:
    return json.dumps([int(sample_id), int(qa_index), int(category), question], ensure_ascii=False)


def flatten_sample_turns(sample) -> List[tuple[str, Any]]:
    turn_entries = []
    for _, turns in sample.conversation.sessions.items():
        for turn in turns.turns:
            turn_entries.append((turns.date_time, turn))
    return turn_entries


def robust_build_status_path(memories_dir: str, sample_idx: int) -> str:
    return os.path.join(memories_dir, f"build_status_sample_{sample_idx}.json")


def load_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_robust_build_checkpoint(
    agent: RobustAdvancedMemAgent,
    memory_cache_file: str,
    retriever_cache_file: str,
    retriever_cache_embeddings_file: str,
    status_path: str,
    sample_idx: int,
    next_turn_index: int,
    total_turns: int,
    complete: bool,
) -> None:
    with open(memory_cache_file, 'wb') as f:
        pickle.dump(agent.memory_system.memories, f)
    agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
    atomic_write_json(
        status_path,
        {
            'sample_id': sample_idx,
            'next_turn_index': next_turn_index,
            'total_turns': total_turns,
            'complete': complete,
            'updated_at': datetime.utcnow().isoformat() + 'Z',
        },
    )


def restore_existing_results(output_path: Optional[str], eval_logger: logging.Logger):
    if not output_path or not os.path.exists(output_path):
        return [], [], [], defaultdict(int), set()
    payload = load_json_file(output_path)
    existing_results = payload.get('individual_results', []) if isinstance(payload, dict) else []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)
    completed_result_keys = set()
    for entry in existing_results:
        category = int(entry.get('category', 0))
        qa_index = int(entry.get('qa_index', 0))
        sample_id = int(entry.get('sample_id', 0))
        question = entry.get('question', '')
        completed_result_keys.add(build_locomo_result_key(sample_id, qa_index, category, question))
        category_counts[category] += 1
        if entry.get('metrics') is not None:
            all_metrics.append(entry['metrics'])
            all_categories.append(category)
    eval_logger.info('Resuming from %d existing QA results in %s', len(existing_results), output_path)
    return existing_results, all_metrics, all_categories, category_counts, completed_result_keys


def persist_locomo_results(
    output_path: Optional[str],
    model: str,
    dataset_path: str,
    category_counts: Dict[int, int],
    all_metrics: List[Dict[str, Any]],
    all_categories: List[int],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    aggregate_results = aggregate_metrics(all_metrics, all_categories) if all_metrics else {}
    final_results = {
        'model': model,
        'dataset': dataset_path,
        'memory_layer': 'robust',
        'total_questions': len(results),
        'category_distribution': {str(cat): count for cat, count in category_counts.items()},
        'aggregate_metrics': aggregate_results,
        'individual_results': results,
    }
    if output_path:
        atomic_write_json(output_path, final_results)
    return final_results


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     vllm_host: Optional[str] = None, vllm_port: Optional[int] = None,
                     api_key: Optional[str] = None, api_base: Optional[str] = None,
                     start_sample: int = 0, end_sample: Optional[int] = None,
                     num_workers: int = 1, worker_id: int = 0,
                     sample_ids: Optional[List[int]] = None,
                     raw_llm_log: Optional[str] = None):
    """Evaluate the robust agent on the LoComo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    worker_suffix = f"_worker{worker_id}" if num_workers > 1 else ""
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}{worker_suffix}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path, raw_llm_log)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = list(enumerate(load_locomo_dataset(dataset_path)))
    eval_logger.info(f"Loaded {len(samples)} samples")

    if sample_ids:
        sample_id_set = set(sample_ids)
        samples = [sample_item for sample_item in samples if sample_item[0] in sample_id_set]
        eval_logger.info(f"Filtered to sample ids: {sorted(sample_id_set)} ({len(samples)} samples)")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    if start_sample < 0:
        raise ValueError("start_sample must be >= 0")
    if end_sample is not None and end_sample < start_sample:
        raise ValueError("end_sample must be >= start_sample")
    samples = samples[start_sample:end_sample]
    eval_logger.info("Selected sample range [%s, %s) -> %d samples", start_sample, end_sample, len(samples))

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")
    if num_workers > 1:
        samples = [sample_item for position, sample_item in enumerate(samples)
                   if position % num_workers == worker_id]
        eval_logger.info(f"Worker {worker_id}/{num_workers} processing {len(samples)} samples")

    results, all_metrics, all_categories, category_counts, completed_result_keys = restore_existing_results(output_path, eval_logger)
    total_questions = len(results)

    i = 0
    error_num = 0
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(backend, model),
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]
    total_qas_in_run = sum(
        sum(1 for qa in sample.qa if int(qa.category) in allow_categories)
        for _, sample in samples
    )

    local_host = vllm_host or sglang_host if backend == "vllm" else sglang_host
    local_port = vllm_port or sglang_port if backend == "vllm" else sglang_port

    sample_progress = tqdm(samples, desc="Samples", unit="sample")
    for shard_idx, (sample_idx, sample) in enumerate(sample_progress):
        sample_progress.set_postfix_str(f"sample_id={sample_idx}")
        agent = RobustAdvancedMemAgent(
            model, backend, retrieve_k, temperature_c5,
            local_host, local_port, api_key, api_base,
        )

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        build_status_file = robust_build_status_path(memories_dir, sample_idx)
        build_status = load_json_file(build_status_file)
        turn_entries = flatten_sample_turns(sample)
        resume_turn_index = 0

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file) and os.path.exists(retriever_cache_embeddings_file):
                eval_logger.info(f"Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(f"No retriever cache found, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, 'all-MiniLM-L6-v2'
                )
            resume_turn_index = int(build_status.get('next_turn_index', 0) or 0)
            if build_status.get('complete') or not build_status:
                resume_turn_index = len(turn_entries)
            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")

        if resume_turn_index < len(turn_entries):
            total_turns = len(turn_entries)
            if resume_turn_index > 0:
                eval_logger.info(
                    "Resuming memory build for sample %s from turn %s/%s",
                    sample_idx,
                    resume_turn_index,
                    total_turns,
                )
            build_progress = tqdm(
                total=total_turns,
                initial=resume_turn_index,
                desc=f"Build {sample_idx}",
                unit="turn",
                leave=False,
            )

            for turn_idx, (turn_datetime, turn) in enumerate(turn_entries[resume_turn_index:], start=resume_turn_index):
                conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                agent.add_memory(conversation_tmp, time=turn_datetime)
                build_progress.update(1)
                save_robust_build_checkpoint(
                    agent,
                    memory_cache_file,
                    retriever_cache_file,
                    retriever_cache_embeddings_file,
                    build_status_file,
                    sample_idx,
                    turn_idx + 1,
                    total_turns,
                    complete=False,
                )

            build_progress.close()
            save_robust_build_checkpoint(
                agent,
                memory_cache_file,
                retriever_cache_file,
                retriever_cache_embeddings_file,
                build_status_file,
                sample_idx,
                len(turn_entries),
                len(turn_entries),
                complete=True,
            )
            eval_logger.info(f"Successfully cached {len(agent.memory_system.memories)} memories")

        eval_logger.info(f"Processing sample {shard_idx + 1}/{len(samples)} (dataset sample {sample_idx})")

        qa_progress = tqdm(sample.qa, desc=f"QA {sample_idx}", unit="qa", leave=False)
        for qa_idx, qa in enumerate(qa_progress):
            if int(qa.category) not in allow_categories:
                continue

            qa_key = build_locomo_result_key(sample_idx, qa_idx, int(qa.category), qa.question)
            if qa_key in completed_result_keys:
                continue

            total_questions += 1
            qa_progress.set_postfix_str(f"global={total_questions}/{total_qas_in_run} cat={qa.category}")
            category_counts[qa.category] += 1

            prompt_answer = get_locomo_prompt_answer(qa.category, qa.answer, qa.adversarial_answer)
            reference_answer = get_locomo_reference_answer(qa.category, qa.answer)
            prediction, user_prompt, raw_context = agent.answer_question(
                qa.question, qa.category, prompt_answer
            )

            prediction = parse_plain_text_answer(prediction)

            eval_logger.info(f"Question {total_questions}: {qa.question}")
            eval_logger.info(f"Prediction: {prediction}")
            eval_logger.info(f"Reference: {reference_answer}")
            eval_logger.info(f"Adversarial Answer: {qa.adversarial_answer}")
            eval_logger.info(f"User Prompt: {user_prompt}")
            eval_logger.info(f"Category: {qa.category}")
            eval_logger.info(f"Raw Context: {raw_context}")

            metrics = calculate_locomo_official_metrics(prediction, reference_answer, qa.category)

            all_metrics.append(metrics)
            all_categories.append(qa.category)

            result = {
                "sample_id": sample_idx,
                "qa_index": qa_idx,
                "question": qa.question,
                "prediction": prediction,
                "reference": reference_answer,
                "adversarial_answer": qa.adversarial_answer,
                "category": qa.category,
                "user_prompt": user_prompt,
                "raw_context": raw_context,
                "metrics": metrics,
                "evaluation_protocol": "official_locomo",
            }
            results.append(result)
            completed_result_keys.add(qa_key)
            persist_locomo_results(
                output_path,
                model,
                dataset_path,
                category_counts,
                all_metrics,
                all_categories,
                results,
            )

            if total_questions % 10 == 0:
                eval_logger.info(f"Processed {total_questions} questions")

    final_results = persist_locomo_results(
        output_path,
        model,
        dataset_path,
        category_counts,
        all_metrics,
        all_categories,
        results,
    )
    eval_logger.info(f"Error number: {error_num}")

    if output_path:
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in final_results.get("aggregate_metrics", {}).items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def _worker_output_path(output_path: Optional[str], worker_id: int) -> Optional[str]:
    if not output_path:
        return None
    output = Path(output_path)
    suffix = output.suffix or ".json"
    stem = output.stem if output.suffix else output.name
    return str(output.with_name(f"{stem}.worker_{worker_id}{suffix}"))


def _merge_batch_results(worker_results: List[Dict]) -> Dict:
    category_counts = defaultdict(int)
    all_metrics = []
    all_categories = []
    individual_results = []
    total_questions = 0
    model = worker_results[0].get("model") if worker_results else None
    dataset = worker_results[0].get("dataset") if worker_results else None
    memory_layer = worker_results[0].get("memory_layer", "robust") if worker_results else "robust"

    for result in worker_results:
        total_questions += result.get("total_questions", 0)
        for category, count in result.get("category_distribution", {}).items():
            category_counts[int(category)] += count
        worker_items = result.get("individual_results", [])
        individual_results.extend(worker_items)
        for item in worker_items:
            metrics = item.get("metrics")
            category = item.get("category")
            if metrics is not None and category is not None:
                all_metrics.append(metrics)
                all_categories.append(category)

    aggregate_results = aggregate_metrics(all_metrics, all_categories) if all_metrics else {}
    return {
        "model": model,
        "dataset": dataset,
        "memory_layer": memory_layer,
        "total_questions": total_questions,
        "category_distribution": {str(cat): count for cat, count in sorted(category_counts.items())},
        "aggregate_metrics": aggregate_results,
        "individual_results": individual_results,
    }


def run_batch_workers(args, dataset_path: str, output_path: Optional[str]) -> Dict:
    if args.batch < 1:
        raise ValueError("batch must be at least 1")

    child_base = [
        sys.executable,
        os.path.abspath(__file__),
        "--dataset", args.dataset,
        "--model", args.model,
        "--ratio", str(args.ratio),
        "--backend", args.backend,
        "--temperature_c5", str(args.temperature_c5),
        "--retrieve_k", str(args.retrieve_k),
        "--sglang_host", args.sglang_host,
        "--sglang_port", str(args.sglang_port),
        "--vllm_host", args.vllm_host,
        "--vllm_port", str(args.vllm_port),
        "--start_sample", str(args.start_sample),
        "--num_workers", str(args.batch),
    ]
    if args.end_sample is not None:
        child_base.extend(["--end_sample", str(args.end_sample)])
    if args.sample_ids:
        child_base.extend(["--sample-ids", args.sample_ids])
    if args.api_key is not None:
        child_base.extend(["--api_key", args.api_key])
    if args.api_base is not None:
        child_base.extend(["--api_base", args.api_base])

    processes = []
    worker_outputs = []
    for worker_id in range(args.batch):
        worker_output = _worker_output_path(output_path, worker_id)
        cmd = child_base + ["--worker_id", str(worker_id)]
        if worker_output is not None:
            rel_worker_output = os.path.relpath(worker_output, os.path.dirname(__file__))
            cmd.extend(["--output", rel_worker_output])
        logger.info("launching worker_id=%s command=%s", worker_id, cmd)
        processes.append((worker_id, worker_output, subprocess.Popen(cmd, cwd=os.path.dirname(__file__))))
        worker_outputs.append(worker_output)

    failures = []
    for worker_id, worker_output, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((worker_id, return_code))
        else:
            logger.info("worker_id=%s completed output=%s", worker_id, worker_output)

    if failures:
        failed = ", ".join(f"worker {worker_id} (exit {return_code})" for worker_id, return_code in failures)
        raise RuntimeError(f"batch run failed: {failed}")

    worker_results = []
    for worker_output in worker_outputs:
        if worker_output is None:
            continue
        with open(worker_output, "r", encoding="utf-8") as f:
            worker_results.append(json.load(f))

    merged_result = _merge_batch_results(worker_results)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_result, f, indent=2)
    return merged_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--vllm_host", type=str, default="http://localhost",
                        help="vLLM server host (for vllm backend)")
    parser.add_argument("--vllm_port", type=int, default=8000,
                        help="vLLM server port (for vllm backend)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI-compatible API key. Defaults to environment variables.")
    parser.add_argument("--api_base", type=str, default=None,
                        help="OpenAI-compatible API base or full /chat/completions endpoint.")
    parser.add_argument("--start_sample", type=int, default=0,
                        help="Inclusive start sample index after applying ratio")
    parser.add_argument("--end_sample", type=int, default=None,
                        help="Exclusive end sample index after applying ratio")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total number of workers for manual sharding")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker id for manual sharding")
    parser.add_argument("--batch", type=int, default=None,
                        help="Launch this many worker processes and merge outputs")
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="Comma-separated dataset sample ids to evaluate")
    parser.add_argument("--raw_llm_log", type=str, default=None,
                        help="Optional file path for full raw LLM prompts/responses.")
    args = parser.parse_args()
    load_dotenv(override=True)
    args.api_base = resolve_api_base(args.api_base)

    raw_llm_log = args.raw_llm_log

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(part.strip()) for part in args.sample_ids.split(",") if part.strip()]

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    worker_suffix = f"_worker{args.worker_id}" if args.num_workers > 1 else ""
    default_log_filename = f"eval_robust_{args.model}_{args.backend}_ratio{args.ratio}{worker_suffix}_{timestamp}.log"
    default_log_path = os.path.join(os.path.dirname(__file__), "logs", default_log_filename)
    os.makedirs(os.path.dirname(default_log_path), exist_ok=True)
    if raw_llm_log is None:
        raw_llm_log = default_log_path.replace('.log', '.raw_llm.log')
    setup_logger(default_log_path, raw_llm_log)

    if args.batch is not None:
        if args.batch == 1:
            args.num_workers = 1
            args.worker_id = 0
        else:
            run_batch_workers(args, dataset_path, output_path)
            return

    logger = logging.getLogger('locomo_eval_robust')
    logger.info('Raw LLM log file: %s', raw_llm_log)

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port,
        args.vllm_host, args.vllm_port,
        args.api_key, args.api_base,
        args.start_sample, args.end_sample,
        args.num_workers, args.worker_id, sample_ids,
        raw_llm_log,
    )


if __name__ == "__main__":
    main()
