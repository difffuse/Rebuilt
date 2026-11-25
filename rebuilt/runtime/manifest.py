"""Offline runtime identity and compatibility checks, independent of Blender."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys

SUPPORTED = frozenset({'macos-arm64', 'windows-x64', 'linux-x64'})
ENGINES = frozenset({'autoremesher', 'cleanseams', 'uvpack'})


class RuntimeManifestError(RuntimeError):
    """A runtime cannot be trusted or used on the requested host."""


def require(condition, message):
    if not condition:
        raise RuntimeManifestError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value,
            f'invalid runtime path: {value!r}')
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(v not in {'', '.', '..'} for v in value.split('/')),
            f'invalid runtime path: {value!r}')
    return value


def _hash(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), 'invalid SHA-256 hash')


def _strings(value, name):
    require(isinstance(value, list) and all(isinstance(v, str) and v for v in value)
            and len(set(value)) == len(value), f'invalid {name}')


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as error:
        raise RuntimeManifestError(f'invalid runtime JSON: {error}') from error


def current_target():
    machine = platform.machine().lower()
    arch = {'arm64': 'arm64', 'aarch64': 'arm64', 'x86_64': 'x64', 'amd64': 'x64'}.get(machine)
    system = {'darwin': 'macos', 'win32': 'windows', 'linux': 'linux'}.get(sys.platform)
    target = f'{system}-{arch}'
    require(target in SUPPORTED, f'unsupported native platform: {sys.platform}/{machine}')
    return target


def current_python_abi():
    require(sys.implementation.name == 'cpython', 'native runtime requires the CPython ABI')
    return f'cp{sys.version_info.major}{sys.version_info.minor}'


def validate_platform(record, target, status):
    require(target in SUPPORTED, f'unsupported native platform: {target}')
    require(isinstance(record, dict), 'invalid platform record')
    require(isinstance(record.get('python_abi'), str)
            and re.fullmatch(r'cp[0-9]{2,3}', record['python_abi']), 'invalid Python ABI')
    engines = record.get('engines')
    require(isinstance(engines, dict) and set(engines) == ENGINES, 'runtime needs exactly three engines')
    destinations = set()
    for name, engine in engines.items():
        require(isinstance(engine, dict), f'invalid engine: {name}')
        require(isinstance(engine.get('commit'), str) and re.fullmatch(r'[0-9a-f]{40}', engine['commit']),
                f'{name}: invalid source commit')
        require(isinstance(engine.get('version'), str) and engine['version'], f'{name}: missing version')
        require(type(engine.get('protocol')) is int and engine['protocol'] >= 0, f'{name}: invalid protocol')
        for key in ('policies', 'capabilities'):
            _strings(engine.get(key), f'{name} {key}')
        for key in ('asset', 'source'):
            value = engine.get(key)
            require(isinstance(value, dict) and 'url' in value, f'{name}: missing {key} provenance')
            _hash(value.get('sha256'))
            require(value['url'] is None or isinstance(value['url'], str), f'{name}: invalid {key} URL')
        require(all(key in engine and (engine[key] is None or isinstance(engine[key], str))
                    for key in ('repository', 'tag')), f'{name}: invalid publication provenance')
        license_info = engine.get('license')
        require(isinstance(license_info, dict) and 'expression' in license_info
                and isinstance(license_info.get('permission'), str)
                and license_info['permission'] in {'pending', 'approved'}, f'{name}: invalid license record')
        _strings(license_info.get('files'), f'{name} license files')
        require(license_info['files'], f'{name}: license files are required')
        require(isinstance(engine.get('loader'), dict), f'{name}: missing loader requirements')
        _strings(engine['loader'].get('requirements'), f'{name} loader requirements')
        if status == 'release':
            urls = [engine['repository'], engine['asset']['url'], engine['source']['url']]
            require(all(isinstance(v, str) and v.startswith('https://') for v in urls)
                    and isinstance(engine['tag'], str) and engine['tag']
                    and isinstance(license_info['expression'], str) and license_info['expression']
                    and license_info['permission'] == 'approved',
                    f'{name}: release publication provenance/permission is incomplete')
        files = engine.get('files')
        require(isinstance(files, list) and files, f'{name}: missing locked files')
        own_paths = set()
        for item in files:
            require(isinstance(item, dict), f'{name}: invalid file record')
            relative_path(item.get('member'))
            path = relative_path(item.get('path'))
            # Case-insensitive collision rejection makes the same lock safe on Windows/macOS.
            require(path.casefold() not in destinations, f'duplicate runtime path: {path}')
            prefixes = [f'bin/{name}/{target}/', f'licenses/{name}/']
            if name == 'uvpack':
                prefixes.append('_vendor/uvpack_geometry/')
            require(any(path.startswith(prefix) for prefix in prefixes),
                    f'{name}: file path is outside its runtime namespace: {path}')
            destinations.add(path.casefold());own_paths.add(path)
            _hash(item.get('sha256'))
            require(type(item.get('mode')) is int and item['mode'] in {0o644, 0o755}, f'{path}: invalid mode')
        require(set(license_info['files']) <= own_paths, f'{name}: license files are not locked')
        entries = engine.get('entrypoints')
        require(isinstance(entries, dict) and entries and all(isinstance(k, str) and k for k in entries),
                f'{name}: missing entrypoints')
        require(all(isinstance(v, str) and v in own_paths for v in entries.values()), f'{name}: entrypoint path is not locked')
        require(all(v.startswith(f'bin/{name}/{target}/') for v in entries.values()),
                f'{name}: entrypoint path does not match engine/platform')


def read_lock(path, target):
    data = Path(path).read_bytes();lock = _json(data)
    require(isinstance(lock, dict) and type(lock.get('schema')) is int and lock['schema'] == 1,
            'unsupported runtime lock schema')
    require(isinstance(lock.get('status'), str) and lock['status'] in {'candidate', 'release'}, 'invalid runtime lock status')
    platforms = lock.get('platforms')
    require(isinstance(platforms, dict) and platforms, 'runtime lock has no platforms')
    for name, record in platforms.items():
        validate_platform(record, name, lock['status'])
    require(target in SUPPORTED and target in platforms, f'no locked runtime for platform: {target}')
    return lock, sha256(data)


@dataclass(frozen=True)
class EngineArtifact:
    name: str
    entrypoints: dict
    metadata: dict


@dataclass(frozen=True)
class RuntimeManifest:
    root: Path
    platform: str
    python_abi: str
    status: str
    identity: str
    engines: dict

    def resolve_engine(self, engine):
        require(engine in self.engines, f'unknown runtime engine: {engine}')
        return self.engines[engine]


def _load_runtime_manifest(root=None, *, target=None, python_abi=None, check_executable=True):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    manifest = root / 'runtime-manifest.json'
    require(manifest.is_file() and not manifest.is_symlink(), 'missing installed runtime manifest')
    data = manifest.read_bytes();record = _json(data)
    require(isinstance(record, dict) and type(record.get('schema')) is int and record['schema'] == 1,
            'unsupported installed runtime schema')
    require(isinstance(record.get('status'), str) and record['status'] in {'candidate', 'release'}, 'invalid installed runtime status')
    target = target or current_target();python_abi = python_abi or current_python_abi()
    require(record.get('platform') == target, f'runtime platform mismatch: expected {target}')
    require(record.get('python_abi') == python_abi, f'runtime Python ABI mismatch: expected {python_abi}')
    _hash(record.get('lock_sha256'))
    validate_platform(record, target, record['status'])
    expected = set()
    for engine in record['engines'].values():
        for item in engine['files']:
            path = root / item['path'];expected.add(item['path'])
            relative = Path(item['path'])
            require(not any((root / Path(*relative.parts[:i])).is_symlink()
                            for i in range(1, len(relative.parts)+1)), f'runtime symlink: {item["path"]}')
            require(path.is_file(), f'missing runtime file: {item["path"]}')
            require(sha256(path.read_bytes()) == item['sha256'], f'runtime hash mismatch: {item["path"]}')
            if check_executable and os.name != 'nt' and item['mode'] == 0o755:
                require(path.stat().st_mode & 0o111, f'runtime is not executable: {item["path"]}')
    if (root / 'bin').exists():
        actual = {p.relative_to(root).as_posix() for p in (root / 'bin').rglob('*') if p.is_file() or p.is_symlink()}
        require(actual <= expected, f'unlocked native files: {sorted(actual - expected)}')
    engines = {name: EngineArtifact(name, {k: root / v for k, v in value['entrypoints'].items()}, value)
               for name, value in record['engines'].items()}
    return RuntimeManifest(root, target, python_abi, record['status'], sha256(data), engines)



def load_runtime_manifest(root=None, *, target=None, python_abi=None):
    from .session import check_manifest
    verified = _load_runtime_manifest(root, target=target, python_abi=python_abi)
    check_manifest(verified)
    return verified


def prepare_runtime(root=None, *, target=None, python_abi=None):
    """Restore ZIP-lost execute bits only after the entire installed tree verifies."""
    verified = _load_runtime_manifest(root, target=target, python_abi=python_abi,
                                      check_executable=False)
    if os.name != 'nt':
        for engine in verified.engines.values():
            for item in engine.metadata['files']:
                if item['mode'] != 0o755:
                    continue
                path = verified.root/item['path']
                try:
                    mode = path.stat().st_mode
                    if mode & 0o111 != 0o111:
                        path.chmod(mode | 0o111)
                except OSError as error:
                    raise RuntimeManifestError(f'cannot prepare executable {path}: {error}') from error
    return load_runtime_manifest(verified.root, target=target, python_abi=python_abi)


def resolve_engine(engine, root=None):
    return load_runtime_manifest(root).resolve_engine(engine)


def verify_compatibility(artifact, capabilities):
    """Compare an engine's observed handshake with its required locked semantics."""
    require(isinstance(capabilities, dict), f'{artifact.name}: missing native handshake')
    required = artifact.metadata
    require(type(capabilities.get('protocol')) is int and capabilities['protocol'] == required['protocol'],
            f'{artifact.name}: incompatible native protocol')
    for key in ('policies', 'capabilities'):
        actual = capabilities.get(key)
        _strings(actual, f'{artifact.name} observed {key}')
        require(set(required[key]) <= set(actual), f'{artifact.name}: missing required {key}')
