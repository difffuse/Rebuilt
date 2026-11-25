"""Stable per-engine cache dependencies from an already verified runtime snapshot."""
import json

from . import manifest


def engine_identity(runtime: manifest.RuntimeManifest, name: str) -> str:
    """Identify code and declared semantics without installation/publication details."""
    metadata = runtime.resolve_engine(name).metadata
    entrypoints = metadata['entrypoints']
    notices = set(metadata['license']['files']) - set(entrypoints.values())
    files = sorted((item['path'], item['sha256']) for item in metadata['files']
                   if not (item['path'].startswith('licenses/') and item['path'] in notices))
    identity = {
        'schema': 1, 'engine': name, 'platform': runtime.platform, 'python_abi': runtime.python_abi,
        'commit': metadata['commit'], 'version': metadata['version'], 'protocol': metadata['protocol'],
        'policies': sorted(set(metadata['policies'])),
        'capabilities': sorted(set(metadata['capabilities'])),
        'entrypoints': entrypoints, 'files': files,
    }
    return manifest.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode('utf-8'))


def dependency_key(settings_key: str, names: tuple[str, ...],
                   runtime: manifest.RuntimeManifest | None = None) -> str:
    """Append only the engines this stage uses; never cache a filesystem observation."""
    if not names:
        return settings_key
    runtime = runtime if runtime is not None else manifest.load_runtime_manifest()
    identities = [(name, engine_identity(runtime, name)) for name in sorted(set(names))]
    digest = manifest.sha256(json.dumps(identities, separators=(',', ':')).encode('utf-8'))
    return f'{settings_key}:runtime-1:{digest}'
