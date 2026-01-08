"""Structured JSON logging with correlation."""
import sys
import json
import logging
from contextvars import ContextVar
from loguru import logger as loguru_logger
from src.app.core.config import settings

# Context variable for trace_id
trace_id: ContextVar[str] = ContextVar("trace_id", default="")

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def json_formatter(record):
    """Format log record as JSON."""
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": record["level"].name,
        "message": record["message"],
        "trace_id": trace_id.get() or "no-trace",
        "service": "inference-api",
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add exception if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": str(record["exception"].type.__name__),
            "value": str(record["exception"].value),
        }
    
    # Add extra fields
    if record["extra"]:
        log_entry.update(record["extra"])
    
    return json.dumps(log_entry) + "\n"

def setup_logging():
    """Setup structured JSON logging."""
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    # Remove default handler
    loguru_logger.remove()
    
    # Add JSON handler for production
    if settings.ENV == "production":
        loguru_logger.add(
            sys.stderr,
            format=json_formatter,
            level=log_level,
            serialize=False,
        )
    else:
        # Human-readable for dev
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]

logger = loguru_logger