"""Filesystem names derived from display names, without lossy substitutions.

Escape Windows path syntax on every platform so an export remains portable.
The escape marker is itself escaped: ``A?B``, ``A_B`` and ``A~3FB`` must not
silently share output files. Callers pass the original asset name, not a stem.
"""

import hashlib
import unicodedata


_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} | {
    prefix + digit for prefix in ('COM', 'LPT') for digit in '123456789¹²³'
}
# Leave 55 bytes in a 255-byte component for engine prefixes, map/resolution
# suffixes, extensions and metadata. UTF-8 also bounds Windows UTF-16 units.
_MAX_STEM_BYTES = 200


def asset_stem(name: str) -> str:
    """Return a portable stem of at most 200 UTF-8 bytes from a raw asset name.

    NFC names that fit retain their escaped spelling. Longer or non-NFC names
    retain a whole token prefix and a stable digest of the original name, keeping
    distinct inputs separate on Unicode-normalizing filesystems too.
    """
    if not name:
        return '~'
    reserved = name.split('.', 1)[0].rstrip(' ').upper() in _RESERVED
    end = len(name.rstrip(' .'))
    tokens = [
        f'~{ord(char):02X}' if (
            char in '<>:"/\\|?*~' or ord(char) < 32 or index >= end
            or (index == 0 and reserved)
        ) else char
        for index, char in enumerate(name)
    ]
    stem = ''.join(tokens)
    if (len(stem.encode('utf-8')) <= _MAX_STEM_BYTES
            and unicodedata.normalize('NFC', name) == name):
        return stem

    # Hash the raw name, not its normalized form: distinct Blender names can
    # otherwise share one filesystem entry. Never adopt that ambiguous old path.
    # Normal escapes reserve '~H'; literal '~H' cannot claim a generated stem.
    suffix = '~H' + hashlib.sha256(name.encode('utf-8')).hexdigest()[:24]
    budget = _MAX_STEM_BYTES - len(suffix)
    prefix = []
    for token in tokens:
        size = len(token.encode('utf-8'))
        if size > budget:
            break
        prefix.append(token)
        budget -= size
    return ''.join(prefix) + suffix
