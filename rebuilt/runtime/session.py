"""Job-local runtime expectations; each verification still reads installed files."""
from contextlib import contextmanager
from contextvars import ContextVar

_expected = ContextVar('rebuilt_expected_runtime', default=None)


@contextmanager
def runtime_scope(expected):
    """Bind explicitly in each worker and reset even when native work raises."""
    token = _expected.set(expected)
    try:
        yield
    finally:
        _expected.reset(token)


def check_manifest(current):
    from .manifest import require
    expected = _expected.get()
    if expected is not None:
        require(current.identity == expected.identity,
                'Runtime changed during the job; restart the job.')


def verify_current_runtime():
    """Verify at launch/transaction boundaries; standalone callers stay unscoped."""
    from .manifest import load_runtime_manifest
    expected = _expected.get()
    if expected is not None:
        check_manifest(load_runtime_manifest(expected.root))
