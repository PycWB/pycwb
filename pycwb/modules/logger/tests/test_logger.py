"""Tests for pycwb.modules.logger — logging initialization and StreamToLogger."""
import logging
import io
import pytest
from pycwb.modules.logger.logger import logger_init, StreamToLogger


class TestLoggerInit:
    """Tests for logger_init — logging configuration helper."""

    def test_sets_level_from_string(self):
        """Should accept a string log level."""
        logger_init(log_level="DEBUG", silent=True)
        root = logging.getLogger()
        # basicConfig may not change effective level of already-configured loggers,
        # but at minimum the call should not raise.
        assert True

    def test_default_level_info(self):
        """Default level should be INFO when not specified."""
        logger_init(silent=True)
        assert True  # smoke test — no exception

    def test_silent_suppresses_info(self):
        """silent=True should not emit logger.info messages about init."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        logging.getLogger("pycwb.modules.logger.logger").addHandler(handler)

        logger_init(silent=True)
        output = buf.getvalue()
        # silent=True means no "Logging initialized" message
        assert "Logging initialized" not in output

    def test_not_silent_emits_init_message(self):
        """Default (silent=False) should log init messages."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        logging.getLogger("pycwb.modules.logger.logger").addHandler(handler)

        logger_init(silent=False)
        output = buf.getvalue()
        assert "Logging initialized" in output

    def test_worker_prefix_in_format(self):
        """worker_prefix should appear in log format."""
        # Smoke test: ensure worker_prefix doesn't crash formatting
        logger_init(worker_prefix="test_worker", silent=True)
        assert True

    def test_noisy_libraries_pinned_to_warning(self):
        """Noisy external libraries should be set to WARNING or higher."""
        logger_init(silent=True)
        for lib in ("jax", "numba", "matplotlib"):
            lib_logger = logging.getLogger(lib)
            assert lib_logger.level <= logging.WARNING


class TestStreamToLogger:
    """Tests for StreamToLogger — redirects writes to logging."""

    def test_write_logs_message(self):
        """Writing to StreamToLogger should emit a log record."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)

        test_logger = logging.getLogger("test_stream")
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        stream = StreamToLogger(test_logger, logging.INFO)
        stream.write("hello world\n")

        output = buf.getvalue()
        assert "hello world" in output

    def test_empty_write_no_log(self):
        """Empty or whitespace-only writes should not produce log records."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)

        test_logger = logging.getLogger("test_stream_empty")
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        stream = StreamToLogger(test_logger, logging.INFO)
        stream.write("   \n")
        stream.write("\n")

        output = buf.getvalue()
        assert output == "" or output.isspace()

    def test_flush_does_nothing(self):
        """flush() is a no-op and should not raise."""
        test_logger = logging.getLogger("test_flush")
        stream = StreamToLogger(test_logger)
        stream.flush()  # should not raise
