"""Logging utilities for Together Open Deep Research."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import litellm


class AgentLogger:
    """Logger class for agent operations."""

    def __init__(
        self,
        name: str = "root",
        level: Union[int, str, None] = logging.INFO,
        log_file: Optional[Path] = None,
        configure_root: bool = False,
    ):
        self.logger = logging.getLogger(name)

        # Prevent propagation to parent loggers to avoid duplicate messages
        self.logger.propagate = False

        # Handle level parameter correctly regardless of type
        if isinstance(level, str):
            level_value = getattr(logging, level.upper(), logging.INFO)
        else:
            level_value = level if level is not None else logging.INFO

        self.logger.setLevel(level_value)

        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create formatters
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="%",
            validate=True,
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="%",
            validate=True,
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(str(log_file))  # Convert Path to str
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # If configure_root is True, also configure the root logger and suppress noisy loggers
        if configure_root:
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(level_value)

            # Clear any existing handlers on root logger
            if root_logger.handlers:
                root_logger.handlers.clear()

            # Add a handler to root logger
            root_handler = logging.StreamHandler(sys.stdout)
            root_handler.setFormatter(console_formatter)
            root_logger.addHandler(root_handler)

        # Suppress noisy third-party loggers
        logging.getLogger("httpx").setLevel(logging.ERROR)

        litellm.suppress_debug_info = True

        # Disable specific litellm loggers
        litellm_loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM"]
        for logger_name in litellm_loggers:
            logger = logging.getLogger(logger_name)
            # Set higher than any standard level
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False  # Also prevent propagation to parent loggers

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.logger.critical(msg)
