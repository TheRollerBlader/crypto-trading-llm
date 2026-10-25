"""Remove credential-shaped text before it reaches logs or stored errors."""

from __future__ import annotations

import re

_ASSIGNED = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?secret|secret|token|bearer|password|authorization)\b\s*[:=]\s*\S+"
)
_ANTHROPIC = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def redact(text: str) -> str:
    cleaned = _ANTHROPIC.sub("[REDACTED]", text)
    cleaned = _ASSIGNED.sub(lambda match: f"{match.group(1)}=[REDACTED]", cleaned)
    return cleaned
