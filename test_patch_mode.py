from pathlib import Path

from memory_layer_patch import PatchAugmentedMemorySystem


class DummyNote:
    def __init__(self, content, context, keywords, tags, links):
        self.content = content
        self.context = context
        self.keywords = keywords
        self.tags = tags
        self.links = links


def test_detect_patchable_change_additive_only(tmp_path: Path):
    system = PatchAugmentedMemorySystem(sample_id="0", store_root=str(tmp_path))
    before = {}
    after = {"n1": DummyNote("a", "ctx", ["k"], ["t"], [])}
    diff = system.detect_patchable_change(before, after)
    assert diff["patch_type"] == "additive_only"


def test_detect_patchable_change_overwrite_update(tmp_path: Path):
    system = PatchAugmentedMemorySystem(sample_id="0", store_root=str(tmp_path))
    before = {"n1": DummyNote("a", "old", ["k"], ["t"], [1])}
    after = {"n1": DummyNote("a", "new", ["k"], ["t"], [1])}
    diff = system.detect_patchable_change(before, after)
    assert diff["patch_type"] == "overwrite_update"


def test_detect_patchable_change_link_rewrite_update(tmp_path: Path):
    system = PatchAugmentedMemorySystem(sample_id="0", store_root=str(tmp_path))
    before = {"n1": DummyNote("a", "ctx", ["k"], ["t"], [1])}
    after = {"n1": DummyNote("a", "ctx", ["k"], ["t"], [2])}
    diff = system.detect_patchable_change(before, after)
    assert diff["patch_type"] == "link_rewrite_update"


def test_build_patch_record_contains_session_metadata(tmp_path: Path):
    system = PatchAugmentedMemorySystem(sample_id="0", store_root=str(tmp_path))
    patch = system.build_patch_record(
        trigger_turn={
            "session_id": 2,
            "session_date_time": "time",
            "session_summary": "summary",
            "turn_position": 3,
            "dia_id": "D2:3",
            "speaker": "Alice",
            "text": "hello",
        },
        patch_type="overwrite_update",
        patch_overall={
            "should_commit_patch": True,
            "decision": "STRENGTHEN_AND_UPDATE",
            "overall_summary": "summary",
            "update_reasoning": "reason",
            "change_pattern": "pattern",
            "selection_signals": ["a", "b"],
            "task_pattern_summary": "task",
        },
        diff_result={
            "updated_note_ids": ["n1"],
            "updated_fields": {"n1": ["context", "links"]},
        },
        detail_blocks=[
            {
                "note_id": "n1",
                "changed_fields": ["context", "links"],
                "before": {
                    "content": "before",
                    "context": "before context",
                    "keywords": ["x"],
                    "tags": ["old"],
                    "links": [1],
                    "retrieval_document": "before doc",
                },
                "after": {
                    "content": "after",
                    "context": "after context",
                    "keywords": ["x"],
                    "tags": ["new"],
                    "links": [2],
                    "retrieval_document": "after doc",
                },
                "link_change_summary": "Links changed from [1] to [2].",
            }
        ],
        evolve_trace={"decision_response": "resp"},
    )
    assert patch["trigger_turn"]["session_id"] == 2
    assert patch["patch_overall"]["overall_summary"] == "summary"
    assert "Trigger:" in patch["index_document"]
