"""Load and validate Llama 3.2 SFT messages JSONL.

Schema (see DATA_FORMAT.md):
  {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset

ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


class MessagesSchemaError(ValueError):
    pass


def validate_messages(messages: Any, *, line_no: int | None = None) -> list[dict[str, str]]:
    loc = f"line {line_no}: " if line_no is not None else ""
    if not isinstance(messages, list) or not messages:
        raise MessagesSchemaError(f"{loc}`messages` must be a non-empty list")

    out: list[dict[str, str]] = []
    roles_seen: set[str] = set()
    for i, turn in enumerate(messages):
        if not isinstance(turn, dict):
            raise MessagesSchemaError(f"{loc}messages[{i}] must be an object")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ALLOWED_ROLES:
            raise MessagesSchemaError(
                f"{loc}messages[{i}].role must be one of {sorted(ALLOWED_ROLES)}, got {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise MessagesSchemaError(f"{loc}messages[{i}].content must be a non-empty string")
        roles_seen.add(role)
        out.append({"role": role, "content": content})

    if "user" not in roles_seen or "assistant" not in roles_seen:
        raise MessagesSchemaError(f"{loc}need at least one user and one assistant turn")
    return out


def iter_messages_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise MessagesSchemaError(f"line {line_no}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise MessagesSchemaError(f"line {line_no}: root must be a JSON object")
            if "messages" not in obj:
                raise MessagesSchemaError(f"line {line_no}: missing `messages` field")
            messages = validate_messages(obj["messages"], line_no=line_no)
            yield {"messages": messages}


def load_messages_dataset(path: str | Path) -> Dataset:
    rows = list(iter_messages_jsonl(Path(path)))
    if not rows:
        raise MessagesSchemaError(f"no valid samples in {path}")
    return Dataset.from_list(rows)


def summarize_jsonl(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    n = 0
    n_system = 0
    multi_turn = 0
    for row in iter_messages_jsonl(path):
        n += 1
        msgs = row["messages"]
        if any(m["role"] == "system" for m in msgs):
            n_system += 1
        if sum(1 for m in msgs if m["role"] == "user") > 1:
            multi_turn += 1
    return {
        "path": str(path),
        "num_samples": n,
        "with_system": n_system,
        "multi_turn_user": multi_turn,
    }
