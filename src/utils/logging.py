"""Logging and artifact helper utilities."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path


def generate_run_id(existing_value: object | None) -> str:
    """Return explicit run ID or generate a timestamp-based one.

    Args:
        existing_value: Optional configured run ID.

    Returns:
        Existing non-empty ID or generated timestamp ID.
    """
    if isinstance(existing_value, str) and existing_value.strip():
        return existing_value.strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_stage_logger(name: str, log_file: Path) -> logging.Logger:
    """Build a stage-specific logger with console and file handlers.

    Args:
        name: Logger name.
        log_file: Stage log file path.

    Returns:
        Configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved_log_file = str(log_file.resolve())
    if logger.handlers:
        has_expected_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and str(Path(handler.baseFilename).resolve()) == resolved_log_file
            for handler in logger.handlers
        )
        if has_expected_file_handler:
            return logger
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def log_stage_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """Emit a structured stage event line.

    Args:
        logger: Destination logger.
        event: Event name.
        **fields: Optional key-value fields to append.
    """
    logger.info(format_stage_event(event, **fields))


def log_stage_event_to_file(logger: logging.Logger, event: str, **fields: object) -> None:
    """Emit one structured stage event only to file handlers.

    Args:
        logger: Destination logger.
        event: Event name.
        **fields: Optional key-value fields to append.
    """
    message = format_stage_event(event, **fields)
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def format_stage_event(event: str, **fields: object) -> str:
    """Return the rendered message body for one structured stage event.

    Args:
        event: Event name.
        **fields: Optional key-value fields to append.

    Returns:
        Formatted log message body.
    """
    event_label = _format_label(event)
    if not fields:
        return event_label
    formatted_fields = " | ".join(
        f"{_format_label(key)}: {_format_field_value(fields[key])}" for key in sorted(fields)
    )
    return f"{event_label} | {formatted_fields}"


def _format_field_value(value: object) -> str:
    """Format event field values in a stable human-readable form."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_label(token: str) -> str:
    """Convert machine-style tokens to concise human-readable labels."""
    acronyms = {"auc", "auprc", "csv", "ddp", "lr"}
    words = token.replace("-", "_").split("_")
    formatted_words = []
    for word in words:
        if not word:
            continue
        if word.lower() in acronyms:
            formatted_words.append(word.upper())
            continue
        formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def prepare_stage_directories(model_name: str, stage: str, run_id: str) -> tuple[Path, Path]:
    """Create and return log/model directories for a stage.

    Args:
        model_name: Model name (e.g. ``v3``).
        stage: Stage name (train/evaluate).
        run_id: Unique stage run ID.

    Returns:
        Tuple of ``(log_dir, model_dir)``.
    """
    log_dir = Path("logs") / model_name / stage / run_id
    model_dir = Path("models") / model_name / stage / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, model_dir


def append_csv_row(
    csv_path: Path,
    row: dict[str, float | int | str],
    fieldnames: list[str] | None = None,
) -> None:
    """Append a row to a CSV file, creating headers when needed.

    Args:
        csv_path: CSV file path.
        row: Row payload keyed by column names.
        fieldnames: Optional explicit column order.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_keys = fieldnames if fieldnames is not None else list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_keys)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
