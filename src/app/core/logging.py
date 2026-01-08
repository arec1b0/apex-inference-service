"""Structured JSON logging with trace correlation."""
import sys
import json
import logging
from contextvars import ContextVar
from loguru import logger as loguru_logger
from opentelemetry import trace

from src.app.core.config import settings

# Context variable for trace_id (thread-safe)
trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Get trace ID from OpenTelemetry context or fallback to manual trace_id."""
    # Try to get from OpenTelemetry span
    span = trace.get_current_span()
    if span and span.is_recording():
        span_context = span.get_span_context()
        if span_context.is_valid:
            return format(span_context.trace_id, "032x")
    
    # Fallback to manual trace_id
    return trace_id.get() or "no-trace"


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""
    
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def json_formatter(record):
    """Format loguru record as JSON with OpenTelemetry trace context."""
    # Get trace_id from OpenTelemetry or fallback
    current_trace_id = get_trace_id()
    
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": record["level"].name,
        "message": record["message"],
        "trace_id": current_trace_id,
        "service": "inference-api",
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add exception if present
    if record["exception"]:
        exc = record["exception"]
        log_entry["exception"] = {
            "type": exc.type.__name__ if exc.type else "Unknown",
            "value": str(exc.value) if exc.value else "",
        }
    
    # Add extra fields
    if record["extra"]:
        log_entry.update(record["extra"])
    
    return json.dumps(log_entry) + "\n"


def setup_logging():
    """Setup structured JSON logging for production."""
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    # Remove default handler
    loguru_logger.remove()
    
    # Production: JSON logs to stderr
    if settings.ENV == "production":
        loguru_logger.add(
            sys.stderr,
            format=json_formatter,
            level=log_level,
            serialize=False,
        )
    else:
        # Development: human-readable colored logs
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )
    
    # Intercept standard logging (uvicorn, fastapi, etc.)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]


# Export logger
logger = loguru_logger
