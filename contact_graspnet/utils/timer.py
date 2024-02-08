import functools
from contextlib import AbstractContextManager
from time import perf_counter

from real_robot.utils.logger import get_logger

_logger_fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"


class RuntimeTimer(AbstractContextManager):
    def __init__(self, description, enabled=True):
        self.description = description
        self.enabled = enabled
        self.elapsed_time = 0.0

        self.logger = get_logger("Timer", fmt=_logger_fmt)

    def __enter__(self):
        if self.enabled:
            self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self.elapsed_time = perf_counter() - self.start_time
            self.logger.info(
                f"{self.description}: Took {self.elapsed_time:.3f} seconds"
            )


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        ret = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        get_logger("Timer", fmt=_logger_fmt).info(
            f"{func.__qualname__}: Took {elapsed_time:0.4f} seconds"
        )
        return ret

    return wrapper_timer
