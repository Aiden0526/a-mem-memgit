"""Patch-augmented wrapper around the robust A-Mem memory layer."""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

from memory_layer import SimpleEmbeddingRetriever
from memory_layer_robust import RobustAgenticMemorySystem, RobustLLMController
from patch_prompts import PATCH_CONTEXT_INSTRUCTION, PATCH_SUMMARIZATION_PROMPT
from patch_store import PatchStore

logger = logging.getLogger("amem_patch_layer")


@dataclass
class PatchConfig:
    patch_top_k: int = 2
    retrieve_k_current: int = 10


class PatchAugmentedMemorySystem:
    """Wrapper that adds patch history on top of the robust global memory graph."""

    def __init__(
        self,
        sample_id: str,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "sglang",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        store_root: Optional[str] = None,
        config: Optional[PatchConfig] = None,
    ):
        self.sample_id = sample_id
        self.config = config or PatchConfig()
        self.base_system = RobustAgenticMemorySystem(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.patch_llm = RobustLLMController(
            backend=llm_backend,
            model=llm_model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.store = PatchStore(
            store_root or os.path.join(os.path.dirname(__file__), f"cached_memories_patch_{llm_backend}_{llm_model}")
        )
        self.patch_retriever = SimpleEmbeddingRetriever(model_name)
        self._patch_counter = 0
        self._load_or_build_patch_retriever()

    def _load_or_build_patch_retriever(self) -> None:
        index_records = self.store.load_patch_index_records(self.sample_id)
        if not index_records:
            return
        cache_file, embeddings_file = self.store.patch_retriever_paths(self.sample_id)
        if os.path.exists(cache_file) and os.path.exists(embeddings_file):
            self.patch_retriever = self.patch_retriever.load(cache_file, embeddings_file)
            return
        self.patch_retriever.add_documents([r["index_document"] for r in index_records])

    def _save_patch_retriever(self) -> None:
        cache_file, embeddings_file = self.store.patch_retriever_paths(self.sample_id)
        self.patch_retriever.save(cache_file, embeddings_file)

    def ingest_turn_with_patch_history(
        self,
        content: str,
        time: Optional[str] = None,
        session_id: Optional[int] = None,
        session_date_time: Optional[str] = None,
        session_summary: Optional[str] = None,
        turn_position: Optional[int] = None,
        dia_id: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> str:
        before_memories = copy.deepcopy(self.base_system.memories)
        note_id, evolve_trace = self.base_system.add_note_with_trace(content, time=time)
        after_memories = self.base_system.memories

        diff_result = self.detect_patchable_change(before_memories, after_memories)
        if diff_result["patch_type"] == "additive_only":
            return note_id

        detail_blocks = self.build_patch_detail_blocks(before_memories, after_memories, diff_result)
        patch_overall = self.summarize_patch_with_llm(
            trigger_turn={
                "session_id": session_id,
                "session_date_time": session_date_time,
                "session_summary": session_summary,
                "turn_position": turn_position,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": content,
            },
            evolve_trace=evolve_trace,
            detail_blocks=detail_blocks,
        )
        patch_record = self.build_patch_record(
            trigger_turn={
                "session_id": session_id,
                "session_date_time": session_date_time,
                "session_summary": session_summary,
                "turn_position": turn_position,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": content,
            },
            patch_type=diff_result["patch_type"],
            patch_overall=patch_overall,
            diff_result=diff_result,
            detail_blocks=detail_blocks,
            evolve_trace=evolve_trace,
        )
        self.commit_patch_if_needed(patch_record)
        return note_id

    def detect_patchable_change(self, before_memories: Dict[str, Any], after_memories: Dict[str, Any]) -> Dict[str, Any]:
        created_note_ids = []
        updated_note_ids = []
        updated_fields: Dict[str, List[str]] = {}
        patch_type = "additive_only"

        for note_id, after_note in after_memories.items():
            if note_id not in before_memories:
                created_note_ids.append(note_id)
                continue
            before_note = before_memories[note_id]
            fields = []
            for field in ("content", "context", "keywords", "tags"):
                if getattr(before_note, field) != getattr(after_note, field):
                    fields.append(field)
            if list(getattr(before_note, "links", [])) != list(getattr(after_note, "links", [])):
                fields.append("links")
            if fields:
                updated_note_ids.append(note_id)
                updated_fields[note_id] = fields

        for note_id in updated_note_ids:
            fields = updated_fields[note_id]
            if any(field in ("content", "context", "keywords", "tags") for field in fields):
                patch_type = "overwrite_update"
                break
            if "links" in fields:
                before_links = list(getattr(before_memories[note_id], "links", []))
                after_links = list(getattr(after_memories[note_id], "links", []))
                if before_links and before_links != after_links:
                    patch_type = "link_rewrite_update"
                    break

        return {
            "patch_type": patch_type,
            "created_note_ids": created_note_ids,
            "updated_note_ids": updated_note_ids,
            "updated_fields": updated_fields,
        }

    def build_patch_detail_blocks(self, before_memories: Dict[str, Any], after_memories: Dict[str, Any], diff_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = []
        for note_id in diff_result["updated_note_ids"]:
            before_note = before_memories[note_id]
            after_note = after_memories[note_id]
            before_payload = self._note_payload(before_note)
            after_payload = self._note_payload(after_note)
            blocks.append({
                "note_id": note_id,
                "changed_fields": diff_result["updated_fields"][note_id],
                "before": before_payload,
                "after": after_payload,
                "link_change_summary": self._link_change_summary(
                    list(getattr(before_note, "links", [])),
                    list(getattr(after_note, "links", [])),
                ),
            })
        return blocks

    def summarize_patch_with_llm(self, trigger_turn: Dict[str, Any], evolve_trace: Dict[str, Any], detail_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        session_metadata = {
            "session_id": trigger_turn.get("session_id"),
            "session_date_time": trigger_turn.get("session_date_time"),
            "session_summary": trigger_turn.get("session_summary"),
            "turn_position": trigger_turn.get("turn_position"),
            "dia_id": trigger_turn.get("dia_id"),
            "speaker": trigger_turn.get("speaker"),
        }
        prompt = PATCH_SUMMARIZATION_PROMPT.format(
            trigger_turn=json.dumps(trigger_turn, ensure_ascii=False, indent=2),
            session_metadata=json.dumps(session_metadata, ensure_ascii=False, indent=2),
            evolve_trace=json.dumps(self._compact_trace(evolve_trace), ensure_ascii=False, indent=2),
            detail_blocks=json.dumps(self._compact_detail_blocks_for_summary(detail_blocks), ensure_ascii=False, indent=2),
        )
        response = self.patch_llm.llm.get_completion(prompt)
        return self._parse_patch_summary_response(response)

    def build_patch_record(self, trigger_turn: Dict[str, Any], patch_type: str, patch_overall: Dict[str, Any], diff_result: Dict[str, Any], detail_blocks: List[Dict[str, Any]], evolve_trace: Dict[str, Any]) -> Dict[str, Any]:
        self._patch_counter += 1
        patch_id = f"patch_{self._patch_counter:06d}"
        patch_record = {
            "patch_id": patch_id,
            "sample_id": self.sample_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "patch_type": patch_type,
            "trigger_turn": trigger_turn,
            "patch_overall": patch_overall,
            "affected_notes": diff_result["updated_note_ids"],
            "difference_to_previous_state": self._difference_summaries(detail_blocks),
            "changed_nodes": detail_blocks,
            "changed_links": [
                {
                    "note_id": block["note_id"],
                    "before_links": block["before"]["links"],
                    "after_links": block["after"]["links"],
                    "link_change_summary": block["link_change_summary"],
                }
                for block in detail_blocks if "links" in block["changed_fields"]
            ],
            "evolve_trace": self._compact_trace(evolve_trace),
        }
        patch_record["index_document"] = self.build_patch_index_document(patch_record)
        return patch_record

    def build_patch_index_document(self, patch_record: Dict[str, Any]) -> str:
        overall = patch_record["patch_overall"]
        changed_bits = []
        for block in patch_record["changed_nodes"][:2]:
            changed_bits.append(
                f"Node {block['note_id']} before context: {block['before']['context']} after context: {block['after']['context']}"
            )
            if block.get("link_change_summary"):
                changed_bits.append(block["link_change_summary"])
        return " ".join([
            f"Trigger: {patch_record['trigger_turn'].get('text', '')}",
            f"Session summary: {patch_record['trigger_turn'].get('session_summary', '')}",
            f"Decision: {overall.get('decision', '')}",
            f"Overall summary: {overall.get('overall_summary', '')}",
            f"Reasoning: {overall.get('update_reasoning', '')}",
            f"Change pattern: {overall.get('change_pattern', '')}",
            f"Signals: {', '.join(overall.get('selection_signals', []))}",
            f"Task pattern: {overall.get('task_pattern_summary', '')}",
            " ".join(changed_bits),
        ]).strip()

    def commit_patch_if_needed(self, patch_record: Dict[str, Any]) -> None:
        if not patch_record["patch_overall"].get("should_commit_patch", False):
            logger.debug(
                "skip_patch sample=%s session=%s turn=%s type=%s decision=%s",
                self.sample_id,
                patch_record["trigger_turn"].get("session_id"),
                patch_record["trigger_turn"].get("turn_position"),
                patch_record["patch_type"],
                patch_record["patch_overall"].get("decision"),
            )
            return
        self.store.save_patch(self.sample_id, patch_record)
        self.store.append_patch_index_record(
            self.sample_id,
            {
                "patch_id": patch_record["patch_id"],
                "sample_id": self.sample_id,
                "session_id": patch_record["trigger_turn"].get("session_id"),
                "index_document": patch_record["index_document"],
            },
        )
        self.patch_retriever.add_documents([patch_record["index_document"]])
        self._save_patch_retriever()
        logger.info(
            "commit_patch sample=%s patch=%s type=%s session=%s turn=%s affected=%s pattern=%s signals=%s",
            self.sample_id,
            patch_record["patch_id"],
            patch_record["patch_type"],
            patch_record["trigger_turn"].get("session_id"),
            patch_record["trigger_turn"].get("turn_position"),
            len(patch_record.get("changed_nodes", [])),
            patch_record["patch_overall"].get("change_pattern", ""),
            ",".join(patch_record["patch_overall"].get("selection_signals", [])),
        )

    def retrieve_current_context(self, query: str, k: Optional[int] = None) -> str:
        return self.base_system.find_related_memories_raw(query, k=k or self.config.retrieve_k_current)

    def summarize_patch_inventory(self) -> Dict[str, Any]:
        patches = self.store.load_all_patches(self.sample_id)
        patch_type_counts: Dict[str, int] = {}
        session_counts: Dict[str, int] = {}
        affected_note_counts: List[int] = []
        for patch in patches:
            patch_type = patch.get("patch_type", "unknown")
            patch_type_counts[patch_type] = patch_type_counts.get(patch_type, 0) + 1
            session_id = str(patch.get("trigger_turn", {}).get("session_id", "unknown"))
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
            affected_note_counts.append(len(patch.get("changed_nodes", [])))
        return {
            "sample_id": self.sample_id,
            "patch_count": len(patches),
            "patch_type_counts": patch_type_counts,
            "session_patch_counts": session_counts,
            "avg_changed_nodes": (sum(affected_note_counts) / len(affected_note_counts)) if affected_note_counts else 0.0,
            "max_changed_nodes": max(affected_note_counts) if affected_note_counts else 0,
        }

    def retrieve_relevant_patches(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        index_records = self.store.load_patch_index_records(self.sample_id)
        if not index_records:
            return []
        indices = self.patch_retriever.search(query, top_k or self.config.patch_top_k)
        patches = []
        for idx in indices:
            if idx >= len(index_records):
                continue
            patch_id = index_records[idx]["patch_id"]
            patch_record = self.store.load_patch(self.sample_id, patch_id)
            if patch_record:
                patches.append(patch_record)
        return patches

    def format_patch_for_context(self, patch_record: Dict[str, Any]) -> str:
        overall = patch_record["patch_overall"]
        changed_lines = []
        for block in patch_record["changed_nodes"][:2]:
            changed_lines.append(f"- Note {block['note_id']} before context: {block['before']['context']}")
            changed_lines.append(f"  after context: {block['after']['context']}")
            if "tags" in block["changed_fields"]:
                changed_lines.append(
                    f"  tags: {', '.join(block['before']['tags'])} -> {', '.join(block['after']['tags'])}"
                )
            if block.get("link_change_summary"):
                changed_lines.append(f"  links: {block['link_change_summary']}")
        return (
            f"Session {patch_record['trigger_turn'].get('session_id')} "
            f"Turn {patch_record['trigger_turn'].get('turn_position')} "
            f"({patch_record['trigger_turn'].get('dia_id')}) [historical change]:\n"
            f"Trigger: {patch_record['trigger_turn'].get('text', '')}\n"
            f"Overall summary: {overall.get('overall_summary', '')}\n"
            f"Reasoning: {overall.get('update_reasoning', '')}\n"
            f"Pattern: {overall.get('change_pattern', '')}\n"
            f"Signals: {', '.join(overall.get('selection_signals', []))}\n"
            f"Key changes:\n" + "\n".join(changed_lines)
        )

    def build_augmented_context(self, current_context: str, patches: List[Dict[str, Any]]) -> str:
        if not patches:
            return current_context
        patch_blocks = "\n\n".join(
            f"[Historical Patch {idx + 1}]\n{self.format_patch_for_context(patch)}"
            for idx, patch in enumerate(patches)
        )
        return (
            f"[Current Global Memory Evidence]\n{current_context}\n\n"
            f"[Relevant Historical Patches]\n{patch_blocks}\n\n"
            f"{PATCH_CONTEXT_INSTRUCTION}"
        )

    def answer_with_patch_history(self, question: str, category: int, answer: str) -> tuple:
        query = self.generate_query_llm(question)
        current_context = self.retrieve_current_context(query, self.config.retrieve_k_current)
        patches = self.retrieve_relevant_patches(query, self.config.patch_top_k)
        context = self.build_augmented_context(current_context, patches)
        if category == 5:
            prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer} or Not mentioned in the conversation  Short answer:"""
        elif category == 2:
            prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
        else:
            prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
        response = self.base_system.llm_controller.llm.get_completion(prompt, temperature=0.7)
        return response, prompt, context

    def generate_query_llm(self, question: str) -> str:
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
        return self.base_system.llm_controller.llm.get_completion(prompt, temperature=0.7).strip()

    @staticmethod
    def _note_payload(note_obj: Any) -> Dict[str, Any]:
        return {
            "content": note_obj.content,
            "context": note_obj.context,
            "keywords": list(note_obj.keywords),
            "tags": list(note_obj.tags),
            "links": list(note_obj.links),
            "retrieval_document": (
                "content:" + note_obj.content +
                " context:" + note_obj.context +
                " keywords: " + ", ".join(note_obj.keywords) +
                " tags: " + ", ".join(note_obj.tags)
            ),
        }

    @staticmethod
    def _link_change_summary(before_links: List[int], after_links: List[int]) -> str:
        if before_links == after_links:
            return ""
        return f"Links changed from {before_links} to {after_links}."

    @staticmethod
    def _difference_summaries(detail_blocks: List[Dict[str, Any]]) -> List[str]:
        summaries = []
        for block in detail_blocks:
            if "context" in block["changed_fields"]:
                summaries.append(
                    f"Updated {block['note_id']} context from '{block['before']['context']}' to '{block['after']['context']}'."
                )
            if "tags" in block["changed_fields"]:
                summaries.append(
                    f"Updated {block['note_id']} tags from {block['before']['tags']} to {block['after']['tags']}."
                )
            if "links" in block["changed_fields"]:
                summaries.append(
                    f"Rewrote {block['note_id']} links from {block['before']['links']} to {block['after']['links']}."
                )
        return summaries

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if np is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        if isinstance(value, dict):
            return {str(k): PatchAugmentedMemorySystem._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [PatchAugmentedMemorySystem._to_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [PatchAugmentedMemorySystem._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _compact_detail_blocks_for_summary(detail_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for block in detail_blocks:
            compact.append({
                "note_id": block["note_id"],
                "changed_fields": block["changed_fields"],
                "before_context": block["before"]["context"],
                "after_context": block["after"]["context"],
                "before_tags": block["before"]["tags"],
                "after_tags": block["after"]["tags"],
                "link_change_summary": block.get("link_change_summary", ""),
            })
        return compact

    @staticmethod
    def _compact_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            "neighbor_indices": trace.get("neighbor_indices", []),
            "decision_parsed": trace.get("decision_parsed"),
            "strengthen_parsed": trace.get("strengthen_parsed"),
            "neighbor_updates_parsed": trace.get("neighbor_updates_parsed"),
            "decision_response": trace.get("decision_response"),
            "strengthen_response": trace.get("strengthen_response"),
            "update_response": trace.get("update_response"),
        }
        return PatchAugmentedMemorySystem._to_jsonable(compact)

    @staticmethod
    def _parse_patch_summary_response(response: str) -> Dict[str, Any]:
        result = {
            "should_commit_patch": False,
            "decision": "UNKNOWN",
            "overall_summary": "",
            "update_reasoning": "",
            "change_pattern": "",
            "selection_signals": [],
            "task_pattern_summary": "",
        }
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "SHOULD_COMMIT_PATCH":
                result["should_commit_patch"] = value.upper().startswith("Y")
            elif key == "DECISION":
                result["decision"] = value
            elif key == "OVERALL_SUMMARY":
                result["overall_summary"] = value
            elif key == "UPDATE_REASONING":
                result["update_reasoning"] = value
            elif key == "CHANGE_PATTERN":
                result["change_pattern"] = value
            elif key == "SELECTION_SIGNALS":
                result["selection_signals"] = [item.strip() for item in value.split(",") if item.strip()][:5]
            elif key == "TASK_PATTERN_SUMMARY":
                result["task_pattern_summary"] = value
        return result
