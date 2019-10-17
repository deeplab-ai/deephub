import logging
from typing import Callable
from time import monotonic
from contextlib import contextmanager

import click


def tf_cpp_log_level(python_log_level: int) -> int:
    """
    Get the TF CPP log level that performs the same type of filtering as python log level.

    Note: Tensorflow has 2 different logging mechanisms. The first is inside the cpp modules and the other
    is in the tensorflow python core modules. To control the CPP log level you need to set the
    environment variable TF_CPP_MIN_LOG_LEVEL

    :param python_log_level: The log level as defined in standard library logging module.
    :return: The log level to perform the desired filtering on CPP modules
    """

    return {
        logging.DEBUG: 0,
        logging.INFO: 0,
        logging.WARNING: 1,
        logging.ERROR: 2
    }.get(python_log_level, 0)


@contextmanager
def context_log(task_title: str, logging_level: int = logging.DEBUG, logger: logging.Logger = None):
    """
    A context manager that will wrap a task and log its start and end

    :param task_title: The title of the task to be emitted in logs.
    :param logging_level: The logging level to use
    :param logger: The logger object, otherwise it will use the root level
    """
    if logger is None:
        logger = logging

    logger.log(logging_level, f"Starting task \"{task_title}\".")
    total_time = -monotonic()
    try:
        yield
    except Exception as e:
        logger.exception(f"Task \"{task_title}\" failed with error: {e!s}")
        raise
    total_time += monotonic()
    logger.log(logging_level, f"Finished task \"{task_title}\" after {total_time:0.2f} secs.")


class ClickStreamHandler(logging.StreamHandler):

    def __init__(self, stream=None, color_formatter: Callable = None):

        self.color_formatter = self._default_color_formatter
        if color_formatter is not None:
            self.color_formatter = color_formatter
        super().__init__(stream)

    @staticmethod
    def _default_color_formatter(record: logging.LogRecord, msg: str) -> str:
        if record.levelno in {logging.ERROR, logging.CRITICAL}:
            return click.style(msg, fg='red', bold=True)
        elif record.levelno == logging.WARNING:
            return click.style(msg, fg='yellow', bold=True)
        elif record.levelno == logging.INFO:
            return click.style(msg, fg='white', bold=True)
        elif record.levelno == logging.DEBUG:
            return click.style(msg, fg='white')
        return msg

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            msg = self.color_formatter(record, msg)
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
