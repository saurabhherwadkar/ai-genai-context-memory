"""Structured logging configuration using structlog."""

import logging  # Standard library logging module
import sys  # System-specific parameters for stdout access
from logging.handlers import RotatingFileHandler  # File handler with size-based rotation
from pathlib import Path  # Object-oriented filesystem paths

import structlog  # Structured logging library for consistent log output

from memory_graph.config.settings import LoggingSettings  # Logging settings model


def configure_logging(settings: LoggingSettings) -> None:
    """Configure structured logging for the entire application.

    Sets up structlog processors, output format, file rotation, and log level.
    """
    # Map string log level to logging module constant
    log_level = getattr(logging, settings.level.upper(), logging.INFO)  # Convert string to logging constant

    # Configure the standard library root logger
    root_logger = logging.getLogger()  # Get the root logger instance
    root_logger.setLevel(log_level)  # Set the minimum log level threshold

    # Remove any existing handlers to prevent duplicate output
    root_logger.handlers.clear()  # Clear pre-existing handlers

    # Create console handler for stdout output
    console_handler = logging.StreamHandler(sys.stdout)  # Direct logs to standard output
    console_handler.setLevel(log_level)  # Apply level filter to console handler
    root_logger.addHandler(console_handler)  # Attach console handler to root logger

    # Create file handler with rotation if a file path is configured
    if settings.file:  # Only create file handler when path is specified
        log_file_path = Path(settings.file)  # Convert string path to Path object
        log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
        file_handler = RotatingFileHandler(  # Create rotating file handler
            filename=str(log_file_path),  # Log file path as string
            maxBytes=settings.max_file_size_mb * 1024 * 1024,  # Convert MB to bytes for rotation threshold
            backupCount=settings.backup_count,  # Number of rotated files to keep
            encoding="utf-8",  # Explicit UTF-8 encoding for log files
        )
        file_handler.setLevel(log_level)  # Apply level filter to file handler
        root_logger.addHandler(file_handler)  # Attach file handler to root logger

    # Build the structlog processor chain based on output format
    shared_processors: list[structlog.types.Processor] = [  # Processors applied to all log entries
        structlog.contextvars.merge_contextvars,  # Merge context variables from async context
        structlog.stdlib.add_logger_name,  # Add the logger name to each event
        structlog.stdlib.add_log_level,  # Add the log level string to each event
        structlog.processors.TimeStamper(fmt="iso"),  # Add ISO 8601 timestamp
        structlog.processors.StackInfoRenderer(),  # Render stack info if present
        structlog.processors.UnicodeDecoder(),  # Decode byte strings to unicode
    ]

    # Select the renderer based on configured format
    if settings.format == "json":  # JSON format for structured log aggregation
        renderer = structlog.processors.JSONRenderer()  # Render as JSON objects
    else:  # Text format for human-readable console output
        renderer = structlog.dev.ConsoleRenderer()  # Render with colors and alignment

    # Configure structlog with the assembled processor chain
    structlog.configure(
        processors=[  # Full processor pipeline
            *shared_processors,  # Include all shared processors first
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Wrap for stdlib integration
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),  # Use stdlib logger factory
        wrapper_class=structlog.stdlib.BoundLogger,  # Bound logger with method-based API
        cache_logger_on_first_use=True,  # Cache logger instances for performance
    )

    # Apply structlog formatter to all handlers
    formatter = structlog.stdlib.ProcessorFormatter(  # Create stdlib-compatible formatter
        processors=[  # Processors run during formatting
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # Clean internal metadata
            renderer,  # Apply the selected renderer as final step
        ],
    )
    for handler in root_logger.handlers:  # Iterate over all configured handlers
        handler.setFormatter(formatter)  # Apply structlog formatter to each handler


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger instance.

    Creates a bound logger that includes the module name in all log entries.
    """
    return structlog.get_logger(name)  # Return a bound logger with the given name
