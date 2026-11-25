"""Windows 10+ process ownership with an atomic inherited Job Object.

A Job Object is attached during CreateProcessW, before any child code can run.
Closing it kills ordinary descendants even after the direct child has exited.
This is deliberately separate from engine/library policy and imports only on NT.
"""
import ctypes as c
from ctypes import wintypes as w
import msvcrt
import os
import subprocess

k = c.WinDLL('kernel32', use_last_error=True)
SIZE_T = c.c_size_t
HANDLE = w.HANDLE


class _BasicLimits(c.Structure):
    _fields_ = [('user_time', c.c_longlong), ('job_time', c.c_longlong),
                ('flags', w.DWORD), ('min_working', SIZE_T), ('max_working', SIZE_T),
                ('active', w.DWORD), ('affinity', SIZE_T), ('priority', w.DWORD),
                ('scheduling', w.DWORD)]


class _Limits(c.Structure):
    _fields_ = [('basic', _BasicLimits), ('io', c.c_ulonglong * 6),
                ('process_memory', SIZE_T), ('job_memory', SIZE_T),
                ('peak_process', SIZE_T), ('peak_job', SIZE_T)]


class _Startup(c.Structure):
    _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR),
                ('title', w.LPWSTR), ('x', w.DWORD), ('y', w.DWORD),
                ('width', w.DWORD), ('height', w.DWORD), ('chars_x', w.DWORD),
                ('chars_y', w.DWORD), ('fill', w.DWORD), ('flags', w.DWORD),
                ('show', w.WORD), ('reserved_size', w.WORD), ('reserved_bytes', c.c_void_p),
                ('stdin', HANDLE), ('stdout', HANDLE), ('stderr', HANDLE)]


class _StartupEx(c.Structure):
    _fields_ = [('startup', _Startup), ('attributes', c.c_void_p)]


class _ProcessInfo(c.Structure):
    _fields_ = [('process', HANDLE), ('thread', HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]


def _api(name, restype, *args):
    fn = getattr(k, name)
    fn.restype = restype
    fn.argtypes = args
    return fn


_close = _api('CloseHandle', w.BOOL, HANDLE)
_create_job = _api('CreateJobObjectW', HANDLE, c.c_void_p, w.LPCWSTR)
_set_job = _api('SetInformationJobObject', w.BOOL, HANDLE, c.c_int, c.c_void_p, w.DWORD)
_init = _api('InitializeProcThreadAttributeList', w.BOOL, c.c_void_p, w.DWORD, w.DWORD, c.POINTER(SIZE_T))
_update = _api('UpdateProcThreadAttribute', w.BOOL, c.c_void_p, w.DWORD, SIZE_T,
               c.c_void_p, SIZE_T, c.c_void_p, c.c_void_p)
_delete = _api('DeleteProcThreadAttributeList', None, c.c_void_p)
_create = _api('CreateProcessW', w.BOOL, w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
               w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(_StartupEx), c.POINTER(_ProcessInfo))
_wait = _api('WaitForSingleObject', w.DWORD, HANDLE, w.DWORD)
_exit_code = _api('GetExitCodeProcess', w.BOOL, HANDLE, c.POINTER(w.DWORD))


def _check(value):
    if not value:
        raise c.WinError(c.get_last_error())
    return value


class WindowsProcess:
    def __init__(self, argv, cwd, env):
        self.job = self.handle = None
        self.stdin = self.stdout = self.stderr = None
        descriptors = []
        attributes = None
        info = _ProcessInfo()
        try:
            self.job = _check(_create_job(None, None))
            limits = _Limits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _check(_set_job(self.job, 9, c.byref(limits), c.sizeof(limits)))
            for _ in range(3):
                descriptors.extend(os.pipe())
            in_read, in_write, out_read, out_write, err_read, err_write = descriptors
            child_fds = (in_read, out_write, err_write)
            for fd in child_fds:
                os.set_inheritable(fd, True)
            handles = (HANDLE * 3)(*(msvcrt.get_osfhandle(fd) for fd in child_fds))
            size = SIZE_T()
            _init(None, 2, 0, c.byref(size))
            if not size.value:
                raise c.WinError(c.get_last_error())
            storage = c.create_string_buffer(size.value)
            _check(_init(storage, 2, 0, c.byref(size)))
            attributes = storage
            _check(_update(attributes, 0, 0x20002, handles, c.sizeof(handles), None, None))
            job_handle = HANDLE(self.job)
            _check(_update(attributes, 0, 0x2000D, c.byref(job_handle), c.sizeof(job_handle), None, None))
            startup = _StartupEx()
            startup.startup.cb = c.sizeof(startup)
            startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
            startup.attributes = c.cast(attributes, c.c_void_p)
            command = c.create_unicode_buffer(subprocess.list2cmdline(argv))
            environment = None
            if env is not None:
                values = [str(key)+'='+str(value) for key, value in sorted(env.items(), key=lambda kv: kv[0].upper())]
                if any('\0' in value for value in values):
                    raise ValueError('process environment cannot contain NUL')
                environment = c.create_unicode_buffer('\0'.join(values)+'\0')
            # CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT
            _check(_create(argv[0], command, None, None, True, 0x08080400,
                           environment, None if cwd is None else os.fsdecode(cwd),
                           c.byref(startup), c.byref(info)))
            self.handle = info.process
            for fd in child_fds:
                os.close(fd)
                descriptors.remove(fd)
            for name, fd, mode in (('stdin', in_write, 'wb'), ('stdout', out_read, 'rb'),
                                   ('stderr', err_read, 'rb')):
                setattr(self, name, os.fdopen(fd, mode, buffering=0))
                descriptors.remove(fd)
        except BaseException:
            self.stop()
            for stream in (self.stdin, self.stdout, self.stderr):
                if stream is not None:
                    stream.close()
            self.close()
            raise
        finally:
            if info.thread:
                _close(info.thread)
            if attributes is not None:
                _delete(attributes)
            for fd in descriptors:
                os.close(fd)

    def poll(self):
        status = _wait(self.handle, 0)
        if status == 258:  # WAIT_TIMEOUT
            return None
        if status != 0:
            raise c.WinError(c.get_last_error())
        result = w.DWORD()
        _check(_exit_code(self.handle, c.byref(result)))
        return result.value

    def stop(self):
        if self.job:
            _check(_close(self.job))
            self.job = None
        if self.handle:
            if _wait(self.handle, 0xFFFFFFFF) != 0:
                raise c.WinError(c.get_last_error())
            return self.poll()

    def close(self):
        if self.handle:
            _check(_close(self.handle))
            self.handle = None
