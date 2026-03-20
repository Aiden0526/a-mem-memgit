"""Prompts for patch-centered memory history augmentation."""

PATCH_SUMMARIZATION_PROMPT = """You are summarizing one memory evolution event into a historical patch record.

Your job is to decide whether this event should be committed as a historical patch and, if yes, produce a concise high-level summary.
Only commit a patch if this event meaningfully rewrote existing memory interpretation, updated existing memory context/tags, or rewrote existing links in a way that may matter later.
Do not commit a patch for purely additive changes.

Trigger turn:
{trigger_turn}

Session metadata:
{session_metadata}

Condensed evolution trace:
{evolve_trace}

Condensed changed node/link summaries:
{detail_blocks}

Formatting rules:
- Write exactly one line per field.
- Do not use bullets, numbering, or markdown.
- Keep DECISION and CHANGE_PATTERN short.
- Keep SELECTION_SIGNALS to at most 5 comma-separated items.
- If SHOULD_COMMIT_PATCH is NO, still fill the remaining fields briefly based on the event.

Respond in EXACTLY this format:

SHOULD_COMMIT_PATCH: <YES or NO>
DECISION: <short label>
OVERALL_SUMMARY: <1 sentence high-level summary>
UPDATE_REASONING: <1 sentence explaining why the update happened>
CHANGE_PATTERN: <short snake_case pattern label>
SELECTION_SIGNALS: <comma-separated signals>
TASK_PATTERN_SUMMARY: <1 sentence>
"""


PATCH_CONTEXT_INSTRUCTION = """The historical patch blocks below describe important memory rewrites that happened earlier in the conversation timeline.
Use the current global memory evidence as the primary source.
Use relevant historical patches when they clarify, refine, or recover details that may have been overwritten by later memory evolution.
When comparing patch blocks, session index and turn index indicate temporal order.
If current evidence and historical patch evidence conflict, prefer the evidence that is more directly relevant to the question."""
