from logging import INFO, Logger, StreamHandler, getLogger


def set_default_logger(logger_name: str) -> Logger:
    logger = getLogger(logger_name)
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setLevel(INFO)
    logger.addHandler(handler)

    return logger
