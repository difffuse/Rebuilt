"""Portable process CPU capacity for native workers."""
import os


def available_cpu_count():
    for query in (
        getattr(os, 'process_cpu_count', lambda: None),
        lambda: len(os.sched_getaffinity(0)),
        os.cpu_count,
    ):
        try:
            count = query()
        except (AttributeError, OSError, NotImplementedError):
            continue
        if count is not None and count > 0:
            return int(count)
    return 1
