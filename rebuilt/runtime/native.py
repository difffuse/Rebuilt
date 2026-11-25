"""Package-owned loading of an integrity-checked CleanSeams CPython binding."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

from .manifest import RuntimeManifestError, load_runtime_manifest, require
from .preflight import _observe, preflight

_lock = threading.RLock()
_loaded = {}


def _import_binding(fullname, path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    require(spec is not None and spec.loader is not None,
            'cleanseams: invalid native Python entrypoint')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cleanseams(root=None, *, timeout=30.0, cancelled=None):
    """Verify CLI and binding identity; reuse only this package's same artifact.

    The upstream binding must embed its private core and own its Session types
    and exception translator. Python namespacing alone cannot supply isolation.
    A runtime changed after import requires restarting Blender; native images
    cannot safely be unloaded or replaced in a running interpreter.
    """
    with _lock:
        checked = preflight(root, timeout=timeout, cancelled=cancelled)
        artifact = checked.manifest.engines['cleanseams']
        require('python' in artifact.entrypoints, 'cleanseams: missing locked Python binding')
        path = artifact.entrypoints['python'].resolve()
        relative = artifact.entrypoints['python'].relative_to(checked.manifest.root).as_posix()
        digest = next(item['sha256'] for item in artifact.metadata['files'] if item['path'] == relative)
        fullname = __package__ + '.cleanseams_native'
        existing = sys.modules.get(fullname)
        owned = _loaded.get(fullname)
        if existing is not None:
            require(owned is not None and owned[0] is existing and owned[1:] == (path, digest),
                    'cleanseams: native module is already loaded or changed; restart Blender')
            require(Path(getattr(existing, '__file__', '')).resolve() == path,
                    'cleanseams: native import belongs to another path; restart Blender')
            module = existing
        else:
            require(owned is None, 'cleanseams: loaded module was removed; restart Blender')
            try:
                module = _import_binding(fullname, path)
            except (ImportError, OSError) as error:
                raise RuntimeManifestError(f'cleanseams: cannot load locked binding: {error}') from error
        require(callable(getattr(module, 'build_info', None)),
                'cleanseams: binding has no native build metadata')
        try:
            metadata = json.dumps(module.build_info(), allow_nan=False).encode('utf-8')
        except (TypeError, ValueError) as error:
            raise RuntimeManifestError('cleanseams: invalid binding metadata') from error
        observed = _observe(artifact, metadata)
        require(observed == checked.observed['cleanseams'],
                'cleanseams: binding and CLI identities differ; restart Blender')
        # Import may have taken time. Refuse a runtime swapped since preflight.
        verified = load_runtime_manifest(checked.manifest.root)
        require(verified.identity == checked.manifest.identity,
                'cleanseams: runtime changed while loading binding; restart Blender')
        sys.modules[fullname] = module
        _loaded[fullname] = (module, path, digest)
        return module


_uvpack_modules = {}


def load_uvpack(root=None, *, timeout=30.0, cancelled=None):
    """Load the locked upstream companion in this addon's private namespace."""
    from types import SimpleNamespace
    with _lock:
        checked = preflight(root, timeout=timeout, cancelled=cancelled)
        artifact = checked.manifest.engines['uvpack']
        required = {'packing_transforms_v1', 'input_policy_v1', 'complete_result_v1',
                    'original_area_density_v1', 'uniform_layout_density_v1', 'hull_policy_v1',
                    'float32_packing_v1', 'emitted_geometry_validation_v1',
                    'validate_operation_v1', 'float32_validation_v1', 'request_cancellation_v1',
                    'normalize_operation_v1', 'fixed_chart_density_v1'}
        require(required <= set(checked.observed['uvpack']['capabilities']),
                'uvpack: locked engine lacks required packing contracts')
        relative = '_vendor/uvpack_geometry/__init__.py'
        files = {item['path']: item['sha256'] for item in artifact.metadata['files']
                 if item['path'].startswith('_vendor/uvpack_geometry/')}
        require(relative in files, 'uvpack: missing locked geometry companion')
        path = (checked.manifest.root / relative).resolve()
        fullname = __package__ + '.uvpack_geometry'
        owned = _uvpack_modules.get(fullname)
        if owned is not None:
            old_path, old_files, modules = owned
            require((path, files) == (old_path, old_files) and all(
                sys.modules.get(name) is module for name, module in modules.items()),
                'uvpack: loaded companion changed; restart Blender')
            module = modules[fullname]
        else:
            require(not any(name == fullname or name.startswith(fullname + '.') for name in sys.modules),
                    'uvpack: companion namespace already occupied; restart Blender')
            spec = importlib.util.spec_from_file_location(fullname, path,
                                                         submodule_search_locations=[str(path.parent)])
            require(spec is not None and spec.loader is not None, 'uvpack: invalid companion entrypoint')
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module  # Relative imports and dataclasses need the package registered.
            try:
                spec.loader.exec_module(module)
                observed = _observe(artifact, json.dumps(module.build_info(), allow_nan=False).encode('utf-8'))
                require(observed == checked.observed['uvpack'], 'uvpack: companion and CLI identities differ')
                require(load_runtime_manifest(checked.manifest.root).identity == checked.manifest.identity,
                        'uvpack: runtime changed during companion import')
            except BaseException:
                for name in list(sys.modules):
                    if name == fullname or name.startswith(fullname + '.'):
                        del sys.modules[name]
                raise
            modules = {name: value for name, value in sys.modules.items()
                       if name == fullname or name.startswith(fullname + '.')}
            _uvpack_modules[fullname] = (path, files, modules)
        observed = _observe(artifact, json.dumps(module.build_info(), allow_nan=False).encode('utf-8'))
        require(observed == checked.observed['uvpack'], 'uvpack: companion and CLI identities differ')
        return SimpleNamespace(companion=module, checked=checked)
