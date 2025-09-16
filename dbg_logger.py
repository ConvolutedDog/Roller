import logging
import os
from datetime import datetime
import inspect
import glob


class DBGLogger:
    """A singleton global logger with both file and console output capabilities.

    This logger provides a centralized logging solution with the following features:
    - Singleton pattern ensures single instance across the application
    - Simultaneous logging to both file (debug level and above) and console (warning level and above)
    - Automatic log file rotation with configurable retention policy
    - Detailed log format including timestamps, log levels, and source locations
    - Thread-safe operation through Python's built-in logging module

    Typical usage:
    >>> from dbg_logger import dbglogger
    >>> dbglogger.debug("Detailed debug information")
    >>> dbglogger.info("System status update")
    >>> dbglogger.warning("Potential issue detected")
    >>> dbglogger.error("Operation failed")
    >>> dbglogger.critical("System in critical state")

    Configuration:
    - Log directory defaults to 'logs/roller-debug' relative to the module location
    - Maximum retained log files defaults to 10 (oldest are automatically deleted)
    - Log format includes timestamp, level, filename, line number and message

    Note: This class should be accessed through the pre-initialized global logger
    instance 'dbglogger' instance rather than instantiated directly.
    """

    _instance = None  # Singleton instance holder

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self, log_dir="logs/roller-debug"):
        """Initialize the logger with console and file handlers"""
        self.logger = logging.getLogger("singleton_global_logger")
        self.logger.setLevel(logging.DEBUG)  # Capture all levels of logs

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)8s [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Log on console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)

        # Log to files
        current_file_path = inspect.getfile(self.__class__)
        current_dir = os.path.dirname(os.path.abspath(current_file_path))
        full_log_dir = os.path.join(current_dir, log_dir)
        os.makedirs(full_log_dir, exist_ok=True)

        # Clean up old log files
        clean_msg = self._clean_old_logs(full_log_dir, max_files=10)
        if clean_msg:
            self.info(clean_msg)

        log_filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(full_log_dir, log_filename)

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _clean_old_logs(self, log_dir, max_files=10):
        """Clean up old log files, keeping only the most recent ones"""
        try:
            log_files = glob.glob(os.path.join(log_dir, "*.log"))

            if len(log_files) <= max_files - 1:
                return

            # Sort files by modification time (newest first)
            log_files.sort(key=os.path.getmtime, reverse=True)

            for old_file in log_files[max_files - 1 :]:
                try:
                    os.remove(old_file)
                    return f"Removed old log file: {old_file}"
                except Exception as e:
                    print(f"Error removing {old_file}: {e}")

        except Exception as e:
            print(f"Error cleaning log files: {e}")

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)


# Global logger instance
dbglogger = DBGLogger()


def _test_global_logger():
    """Test function to demonstrate logger functionality"""

    # Messages with different formatting styles
    dbglogger.debug(f"This is a debug message with a timestamp: {datetime.now()}")
    dbglogger.debug("This is a debug message {} {}".format("with", "formatting"))
    dbglogger.info("This is an info message with a variable: %s" % "variable_value")
    dbglogger.warning("This is a warning message with a number: %d" % 42)
    dbglogger.error("This is an error message with a list: %s" % [1, 2, 3])
    dbglogger.critical(
        "This is a critical message with an exception: %s" % Exception("Critical error")
    )
