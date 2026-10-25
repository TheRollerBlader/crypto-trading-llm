"""JSON logs with a redaction filter."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.redact import redact


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {key: redact(value) if isinstance(value, str) else value for key, value in record.args.items()}
            else:
                record.args = tuple(redact(value) if isinstance(value, str) else value for value in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        data = getattr(record, "data", None)
        if isinstance(data, dict):
            payload["data"] = data
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    redact_filter = RedactFilter()
    formatter = JsonFormatter()
    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        stream.addFilter(redact_filter)
        root.addHandler(stream)
    if not any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == str(log_path) for handler in root.handlers):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact_filter)
        root.addHandler(file_handler)
