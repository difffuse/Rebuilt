"""Observe locked CLI semantics before work, without importing native bindings."""
from dataclasses import dataclass
import json
import math
import os
import re
import time

from .manifest import (RuntimeManifest, RuntimeManifestError, _json, _strings,
                       load_runtime_manifest, require, sha256, verify_compatibility)
from .process import run_process


@dataclass(frozen=True)
class RuntimePreflight:
    manifest: RuntimeManifest
    observed: dict
    identity: str

    def diagnostic(self):
        """Return a serializable, installation-path-independent identity report."""
        return {'manifest_identity': self.manifest.identity, 'identity': self.identity,
                'platform': self.manifest.platform, 'python_abi': self.manifest.python_abi,
                'status': self.manifest.status,
                'engines': json.loads(json.dumps(self.observed))}


def _observe(artifact, payload):
    name = artifact.name
    try:
        text = payload.decode('utf-8', errors='strict')
        if name == 'autoremesher':
            match = re.fullmatch(
                r'autoremesher-fine-details (unknown|[0-9a-f]{7,40}) rebuilt-cli-([0-9]+) '
                r'([a-zA-Z0-9_.-]+)\s*', text)
            require(match is not None, 'invalid AutoRemesher version response')
            revision, protocol, patch = match.groups()
            require(revision == 'unknown' or artifact.metadata['commit'].startswith(revision),
                    'native source revision differs from lock')
            # This legacy CLI has no semantic version field. The lock's version
            # is a provenance label, not an observed version; retain unknown.
            observed = {'source_revision': revision, 'protocol': int(protocol),
                        'policies': [], 'capabilities': [patch]}
        else:
            data = _json(text)
            require(isinstance(data, dict), 'native version response must be an object')
            require(data.get('version') == artifact.metadata['version'],
                    'native version differs from lock')
            if name == 'cleanseams':
                require(data.get('engine') == name, 'wrong native engine')
                require(data.get('source_revision') == artifact.metadata['commit'],
                        'native source revision differs from lock')
                digest = data.get('source_tree_sha256')
                require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                        'invalid native source content identity')
                schemas = data.get('schemas')
                features = data.get('features')
                require(isinstance(schemas, dict), 'missing native schemas')
                require(isinstance(features, dict), 'missing native features')
                observed = {'version': data['version'], 'source_revision': data['source_revision'],
                            'source_tree_sha256': digest, 'protocol': schemas.get('run_input'),
                            'policies': data.get('behaviors'),
                            'capabilities': [k for k, v in features.items() if v is True]}
            else:
                require(data.get('dialect') == name, 'wrong native dialect')
                require(type(data.get('metadata_schema')) is int and data['metadata_schema'] == 1,
                        'unsupported native metadata schema')
                require(type(data.get('companion_api')) is int and data['companion_api'] >= 1,
                        'invalid native companion API')
                observed = {'version': data['version'], 'protocol': data.get('protocol_api'),
                            'companion_api': data['companion_api'], 'policies': [],
                            'capabilities': data.get('capabilities')}
        for field in ('policies', 'capabilities'):
            _strings(observed[field], f'observed {field}')
            observed[field] = sorted(observed[field])
        verify_compatibility(artifact, observed)
        return observed
    except (RuntimeManifestError, UnicodeError, ValueError) as error:
        raise RuntimeManifestError(f'{name}: {error}') from error


def preflight(root=None, *, timeout=30.0, cancelled=None):
    """Verify all files, query all engines and reverify before returning identity.

    The single deadline includes file verification and all three bounded child
    processes. This is a snapshot, not a filesystem lease: callers must not
    replace the runtime during a job. No mesh or Blender state is modified.
    """
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout >= 0,
            'invalid runtime preflight timeout')
    deadline = time.monotonic() + timeout
    if cancelled is not None and cancelled():
        raise RuntimeManifestError('runtime preflight cancelled')
    manifest = load_runtime_manifest(root)
    # Loader overrides belong to the host application, never to locked children.
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(('LD_', 'DYLD_'))}
    observed = {}
    for name in sorted(manifest.engines):
        remaining = max(0.0, deadline - time.monotonic())
        require(remaining > 0, f'{name}: runtime preflight timed out')
        artifact = manifest.engines[name]
        require('cli' in artifact.entrypoints, f'{name}: missing locked CLI entrypoint')
        argv = [str(artifact.entrypoints['cli'].resolve()), '--version']
        if name == 'cleanseams':
            argv.append('--json')
        try:
            result = run_process(argv, timeout=remaining, cancelled=cancelled,
                                 max_stdout=256*1024, max_stderr=8192, env=env)
        except OSError as error:
            raise RuntimeManifestError(f'{name}: cannot start native handshake: {error}') from error
        require(not result.cancelled, f'{name}: runtime preflight cancelled')
        require(not result.timed_out, f'{name}: runtime preflight timed out')
        require(not result.stdout_truncated, f'{name}: native handshake exceeds output limit')
        require(result.returncode == 0,
                f'{name}: native handshake exited with status {result.returncode}')
        observed[name] = _observe(artifact, result.stdout)
    verified = load_runtime_manifest(manifest.root)
    require(verified.identity == manifest.identity, 'runtime changed during native preflight')
    require(time.monotonic() < deadline, 'runtime preflight timed out')
    if cancelled is not None and cancelled():
        raise RuntimeManifestError('runtime preflight cancelled')
    identity = sha256(json.dumps({'schema': 1, 'manifest': manifest.identity, 'observed': observed},
                                 sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return RuntimePreflight(manifest, observed, identity)
