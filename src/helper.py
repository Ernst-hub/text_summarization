import logging
import time

import memory_profiler


def log_method_call(func):
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            logging.info(f"Calling {func.__name__}")
            start_time = time.time()
            start_memory = memory_profiler.memory_usage()[0]
            res = func(self, *args, **kwargs)
            end_time = time.time()
            end_memory = memory_profiler.memory_usage()[0]

            elapsed_time = end_time - start_time
            used_memory = end_memory - start_memory

            logging.info(
                f"Finished {func.__name__} in {elapsed_time:.4f} seconds.\nMemory used {used_memory:.4f} MB."
            )
            return res
        else:
            return func(self, *args, **kwargs)

    return wrapper
