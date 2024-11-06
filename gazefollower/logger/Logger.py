import logging
import threading


class Log:
    _instance = None
    _lock = threading.Lock()

    # def __new__(cls):
    #     if not cls._instance:
    #         with cls._lock:
    #             if not cls._instance:
    #                 cls._instance = super().__new__(cls)
    #                 cls._instance._logger = None
    #     return cls._instance

    @classmethod
    def init(cls, log_file_path):
        cls._instance = cls()
        cls._instance._logger = cls._create_logger(log_file_path)

    @staticmethod
    def _create_logger(log_file_path):
        logger = logging.getLogger('gaze_follower_logger')
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    @classmethod
    def i(cls, message):
        cls._check_logger()
        cls._instance._logger.info(message)

    @classmethod
    def d(cls, message):
        cls._check_logger()
        cls._instance._logger.debug(message)

    @classmethod
    def w(cls, message):
        cls._check_logger()
        cls._instance._logger.warning(message)

    @classmethod
    def e(cls, message):
        cls._check_logger()
        cls._instance._logger.error(message)

    @classmethod
    def _check_logger(cls):
        if cls._instance is None or cls._instance._logger is None:
            raise Exception("Logger has not been initialized. Please call Log.init() first.")
