"""Bounded byte transport and lifetime ownership for native CLI processes.

The caller supplies the verified executable and interprets its protocol, including
nonzero result exits. No shell, engine fallback, output-file deletion or Blender
operation is performed here. Ordinary descendants belong to this invocation.
"""
from dataclasses import dataclass
import math
import os
import queue
import signal
import subprocess
import threading
import time


class ProcessStartError(OSError):
    """The native child could not be created; callbacks have not run in it."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: bytes = b''
    stderr: bytes = b''
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    cancelled: bool = False
    timed_out: bool = False


class _PosixProcess:
    def __init__(self, argv, cwd, env):
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, cwd=cwd, env=env,
                                        start_new_session=True)
        self.stdin, self.stdout, self.stderr = self.process.stdin, self.process.stdout, self.process.stderr

    def poll(self):
        return self.process.poll()

    def stop(self):
        # Also terminate ordinary descendants after a successful parent exit;
        # otherwise inherited pipe handles can strand the reader/writer threads.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return self.process.wait()

    def close(self):
        pass


def run_process(argv, *, input=b'', timeout, cancelled=None, max_stdout=64*1024*1024,
                max_stderr=64*1024, cwd=None, env=None, on_output=None):
    """Run a CLI, retaining a bounded stdout prefix and stderr tail.

    Cancellation and deadlines return explicit flags, even when an engine exits
    zero concurrently. Exceptions from the cancellation callback propagate only
    after process-tree and pipe cleanup. Truncation is explicit: callers must not
    accept a truncated structured response. ``timeout`` includes process startup.

    Optional ``on_output(stream, bytes)`` runs on the caller's thread as output
    arrives, before child exit. It must return promptly. Streams preserve their
    own byte order; chunks need not be whole text lines or UTF-8 characters.
    Capture limits do not limit callback delivery. A bounded queue backpressures
    readers; cancellation or callback failure stops delivery and cleans up the
    entire process tree before returning or propagating the error.
    """
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout < 0):
        raise ValueError('timeout must be finite and nonnegative')
    if any(type(v) is not int or v < 0 for v in (max_stdout, max_stderr)):
        raise ValueError('output limits must be nonnegative integers')
    if on_output is not None and not callable(on_output):
        raise TypeError('on_output must be callable')
    if not isinstance(input, bytes):
        raise TypeError('process input must be bytes')
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError('an argument vector is required')
    argv = [os.fsdecode(arg) for arg in argv]
    if any('\0' in arg for arg in argv):
        raise ValueError('process arguments cannot contain NUL')
    started = time.monotonic()
    if cancelled is not None and cancelled():
        return ProcessResult(None, cancelled=True)
    if timeout == 0:
        return ProcessResult(None, timed_out=True)
    try:
        if os.name == 'nt':
            from ._process_windows import WindowsProcess
            process = WindowsProcess(argv, cwd, env)
        else:
            process = _PosixProcess(argv, cwd, env)
    except OSError as error:
        raise ProcessStartError(str(error)) from error
    buffers = [bytearray(), bytearray()]
    truncated = [False, False]
    failures = []
    wake = threading.Event()
    workers = []
    pending = queue.Queue(maxsize=16) if on_output is not None else None
    closing = threading.Event()

    def protect(work):
        try:
            work()
        except BaseException as error:
            failures.append(error)
            wake.set()

    def read(stream, index, limit):
        try:
            while data := stream.read(8192):
                target = buffers[index]
                truncated[index] |= len(target) + len(data) > limit
                if index == 0:
                    target.extend(data[:max(0, limit-len(target))])
                elif limit:
                    target.extend(data)
                    del target[:-limit]
                if pending is not None:
                    while not closing.is_set():
                        try:
                            pending.put((('stdout', 'stderr')[index], data), timeout=.01)
                            wake.set()
                            break
                        except queue.Full:
                            pass
        finally:
            stream.close()

    def write():
        try:
            view = memoryview(input)
            while view:
                count = process.stdin.write(view[:8192])
                if not count:
                    break
                view = view[count:]
        except BrokenPipeError:
            # Engines may reject input early and still return structured output.
            pass
        finally:
            process.stdin.close()

    was_cancelled = timed_out = stopped = False
    try:
        for work in (lambda: read(process.stdout, 0, max_stdout),
                     lambda: read(process.stderr, 1, max_stderr), write):
            thread = threading.Thread(target=protect, args=(work,), name='rebuilt-native-io')
            thread.start()
            workers.append(thread)
        while True:
            was_cancelled = bool(cancelled is not None and cancelled())
            timed_out = not was_cancelled and time.monotonic()-started >= timeout
            if was_cancelled or timed_out or failures:
                break
            if not stopped and process.poll() is not None:
                # End ordinary descendants immediately, even if they hold pipes
                # open. Keep draining/delivering the bytes already in the pipes.
                returncode = process.stop()
                stopped = True
            if pending is not None:
                try:
                    stream, data = pending.get_nowait()
                except queue.Empty:
                    pass
                else:
                    on_output(stream, data)
                    continue
            if stopped and all(not worker.is_alive() for worker in workers):
                # Recheck after observing reader completion: a reader may have
                # queued its last chunk between get_nowait() and is_alive().
                if pending is None or pending.empty():
                    break
                continue
            wake.wait(.01)
            wake.clear()
    finally:
        closing.set()  # Release a reader blocked on callback-queue backpressure.
        try:
            if not stopped:
                returncode = process.stop()
        finally:
            for thread in workers:
                thread.join()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            process.close()
    if failures:
        raise failures[0]
    return ProcessResult(returncode, bytes(buffers[0]), bytes(buffers[1]),
                         *truncated, was_cancelled, timed_out)
