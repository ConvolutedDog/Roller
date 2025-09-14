import logging
import os
from datetime import datetime
import inspect
import glob


class DBGLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self, log_dir="logs/roller-debug"):
        self.logger = logging.getLogger("singleton_global_logger")
        self.logger.setLevel(logging.DEBUG)

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
        clean_msg = self._clean_old_logs(full_log_dir, max_files=10)

        log_filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(full_log_dir, log_filename)

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        if clean_msg:
            self.info(clean_msg)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _clean_old_logs(self, log_dir, max_files=10):
        try:
            log_files = glob.glob(os.path.join(log_dir, "*.log"))

            if len(log_files) <= max_files - 1:
                return

            log_files.sort(key=os.path.getmtime, reverse=True)

            for old_file in log_files[max_files - 1 :]:
                try:
                    os.remove(old_file)
                    return f"Removed old log file: {old_file}"
                except Exception as e:
                    print(f"Error removing {old_file}: {e}")

        except Exception as e:
            print(f"Error cleaning log files: {e}")

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


dbglogger = DBGLogger()


def _test_global_logger():
    dbglogger.debug("This is a debug message")
    dbglogger.info("This is an info message")
    dbglogger.warning("This is a warning message")
    dbglogger.error("This is an error message")
    dbglogger.critical("This is a critical message")

    dbglogger.debug(f"This is a debug message with a timestamp: {datetime.now()}")
    dbglogger.info("This is an info message with a variable: %s" % "variable_value")
    dbglogger.warning("This is a warning message with a number: %d" % 42)
    dbglogger.error("This is an error message with a list: %s" % [1, 2, 3])
    dbglogger.critical(
        "This is a critical message with an exception: %s" % Exception("Critical error")
    )

    dbglogger.debug("This is a debug message {} {}".format("with", "formatting"))
