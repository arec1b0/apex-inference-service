import sys
import logging
import json
from loguru import logger
from src.app.core.config import settings

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def serialize(record):
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "context": record["extra"],  # This captures correlation_id
        "exception": record["exception"],
    }
    return json.dumps(subset)

def setup_logging():
    # Remove default handlers
    logging.getLogger().handlers = []
    
    # Configuration
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    # Configure Loguru with JSON sink for Production
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout, 
                "level": log_level,
                "serialize": True, # Native JSON serialization
                # Or use custom: "format": "{message}", "sink": lambda msg: print(serialize(msg.record))
            }
        ]
    )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]