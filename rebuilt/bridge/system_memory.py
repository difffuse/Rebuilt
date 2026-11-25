"""Installed RAM capacity, without importing Blender or inspecting user data."""
import ctypes
import os
import sys


def installed_memory_bytes():
    if sys.platform == 'darwin':
        size = ctypes.c_size_t(ctypes.sizeof(ctypes.c_uint64))
        capacity = ctypes.c_uint64()
        library = ctypes.CDLL(None, use_errno=True)
        call = library.sysctlbyname
        call.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                         ctypes.c_void_p, ctypes.c_size_t]
        call.restype = ctypes.c_int
        if call(b'hw.memsize', ctypes.byref(capacity), ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot read installed RAM')
        value = capacity.value
    elif sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    'total_physical', 'available_physical', 'total_pagefile',
                    'available_pagefile', 'total_virtual', 'available_virtual',
                    'available_extended_virtual')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        library = ctypes.WinDLL('kernel32', use_last_error=True)
        call = library.GlobalMemoryStatusEx
        call.argtypes = [ctypes.POINTER(MemoryStatus)]
        call.restype = ctypes.c_int
        if not call(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), 'Cannot read installed RAM')
        value = status.total_physical
    else:
        value = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    if value <= 0:
        raise ValueError('Installed RAM capacity is unavailable')
    return int(value)
