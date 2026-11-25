"""Durable, conflict-checked texture writes and cancellation recovery (no bpy)."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid


class FileRecoveryError(RuntimeError):
    pass


class FileCommitUncertain(FileRecoveryError):
    """Neither rollback nor cleanup is safe until the persisted decision can be read."""


def _sync_directory(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _regular(path):
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise FileRecoveryError(f'Recovery path contains a symbolic link: {path}')
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise FileRecoveryError(f'Recovery path is not a regular file: {path}')
    return info


def _identity(info):
    return {key: getattr(info, 'st_' + key) for key in ('dev', 'ino', 'size', 'mtime_ns')}


def _fingerprint(path):
    before = _regular(path)
    if before is None:
        return None
    digest = hashlib.sha256()
    with path.open('rb') as file:
        if _identity(os.fstat(file.fileno())) != _identity(before):
            raise FileRecoveryError(f'File changed while being read: {path}')
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(file.fileno())
    current = _regular(path)
    if current is None or _identity(before) != _identity(after) or _identity(before) != _identity(current):
        raise FileRecoveryError(f'File changed while being read: {path}')
    return dict(_identity(before), sha256=digest.hexdigest())


def _same_content(left, right):
    return left is not None and right is not None and (left['size'], left['sha256']) == (right['size'], right['sha256'])


def _sync_file(path):
    # Windows flushes require write access; r+b neither creates nor truncates.
    # Keep POSIX support for flushing files without write permission.
    with path.open('r+b' if os.name == 'nt' else 'rb') as file:
        os.fsync(file.fileno())


def _require(condition):
    if not condition:
        raise ValueError('Invalid recovery journal field')


def _valid_fingerprint(value):
    return value is None or (
        isinstance(value, dict) and set(value) == {'dev', 'ino', 'size', 'mtime_ns', 'sha256'}
        and all(type(value[key]) is int and value[key] >= 0 for key in ('dev', 'ino', 'size', 'mtime_ns'))
        and isinstance(value['sha256'], str) and len(value['sha256']) == 64
        and all(c in '0123456789abcdef' for c in value['sha256']))


def validate_descriptor(descriptor):
    if not hasattr(descriptor, 'get'):
        raise FileRecoveryError('Invalid texture recovery descriptor.')
    directory, token = descriptor.get('directory'), descriptor.get('token')
    if (not isinstance(directory, str) or not isinstance(token, str)
            or len(token) != 32 or any(c not in '0123456789abcdef' for c in token)
            or not Path(directory).is_absolute() or Path(directory).name != token):
        raise FileRecoveryError('Invalid texture recovery descriptor.')
    return Path(directory)


class CancelFiles:
    def __init__(self, root, *, before_first_write=None, origin=None):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / uuid.uuid4().hex
        self.directory.mkdir()
        self._before_first_write = before_first_write
        self._lease = None
        self._state = {'version': 2, 'token': self.directory.name, 'phase': 'active',
                       'files': {}, 'folders': {}, 'staging': {}}
        if origin is not None:
            _require(isinstance(origin, dict) and all(isinstance(key, str) and
                     type(value) in (str, int) for key, value in origin.items()))
            self._state['origin'] = dict(origin)
        self._save()
        _sync_directory(root)

    @property
    def descriptor(self):
        return {'directory': str(self.directory), 'token': self.directory.name}

    @property
    def origin(self):
        """Discovery metadata; it never replaces the lease or content checks."""
        with self._locked():
            return dict(self._state.get('origin', {}))

    @classmethod
    def reopen(cls, descriptor):
        self = cls.__new__(cls)
        self.directory = validate_descriptor(descriptor)
        self._before_first_write = None
        self._lease = None
        self._read()
        return self

    @contextmanager
    def _locked(self):
        if self._lease is not None:
            self._read()
            yield
        else:
            with self._file_lock():
                yield

    def hold(self):
        """Keep recovery exclusive while a live Blender job owns this journal."""
        if self._lease is None:
            lease = self._file_lock()
            lease.__enter__()
            self._lease = lease

    def release(self):
        """Release the live owner without discarding persistent recovery data."""
        if self._lease is not None:
            lease, self._lease = self._lease, None
            lease.__exit__(None, None, None)

    @contextmanager
    def _file_lock(self):
        # Never unlink the lock inode: every process must lock the same file.
        path = self.directory / '.lock'
        _regular(path)
        with path.open('a+b') as lock:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b'0')
                    lock.flush()
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise FileRecoveryError('Texture recovery is already in use.') from exc
            else:
                import fcntl
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise FileRecoveryError('Texture recovery is already in use.') from exc
            try:
                self._read()
                yield
            finally:
                if os.name == 'nt':
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self):
        path = self.directory / 'journal.json'
        if _regular(path) is None:
            raise FileRecoveryError(f'Texture recovery journal is missing: {path}')
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            _require(type(state['version']) is int and state['version'] in (1, 2))
            _require(state['token'] == self.directory.name)
            _require(state['phase'] in ('active', 'rolling_back', 'rolled_back', 'committed'))
            if 'origin' in state:
                _require(isinstance(state['origin'], dict) and all(isinstance(key, str) and
                         type(value) in (str, int) for key, value in state['origin'].items()))
            _require(isinstance(state['files'], dict) and isinstance(state['folders'], dict))
            for name, row in state['files'].items():
                _require(isinstance(name, str) and Path(name).is_absolute() and isinstance(row, dict))
                _require(set(row) == {'original', 'expected', 'pending', 'restored', 'removing', 'backup'})
                _require(all(_valid_fingerprint(row[key]) for key in ('original', 'expected', 'pending', 'restored')))
                _require(type(row['removing']) is bool)
                _require(row['backup'] is None or (isinstance(row['backup'], str) and Path(row['backup']).name == row['backup'] and row['backup'].endswith('.backup')))
                _require((row['original'] is None) == (row['backup'] is None))
            for folder, identity in state['folders'].items():
                _require(isinstance(folder, str) and Path(folder).is_absolute())
                _require(any(Path(folder) in Path(name).parents for name in state['files']))
                _require(isinstance(identity, dict) and set(identity) == {'dev', 'ino'})
                _require(all(type(value) is int and value >= 0 for value in identity.values()))
            if state['version'] == 1:
                state['staging'] = {}
                state['version'] = 2
            _require(isinstance(state['staging'], dict))
            for name, row in state['staging'].items():
                _require(isinstance(name, str) and Path(name).is_absolute())
                _require(isinstance(row, dict) and set(row) == {'target', 'dev', 'ino'})
                _require(row['target'] in state['files'])
                _require(Path(name).parent == Path(row['target']).parent)
                _require(Path(name).name.startswith('.rebuilt-' + state['token'] + '-'))
                _require(all(type(row[key]) is int and row[key] >= 0 for key in ('dev', 'ino')))
        except (ValueError, TypeError, KeyError, AssertionError) as exc:
            raise FileRecoveryError(f'Invalid texture recovery journal; backups retained: {path}') from exc
        self._state = state

    def _save(self):
        fd, name = tempfile.mkstemp(prefix='.journal-', dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file:
                json.dump(self._state, file, sort_keys=True, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.directory / 'journal.json')
            _sync_directory(self.directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _check_current(self, path, row):
        current = _fingerprint(path)
        if self._state['phase'] == 'rolled_back':
            allowed = [row['restored']]
        else:
            allowed = [row['expected']]
            allowed.extend(row[key] for key in ('pending', 'restored') if row[key] is not None)
            if row['removing']:
                allowed.append(None)
        if current not in allowed:
            raise FileRecoveryError(f'Texture changed outside this build; recovery retained: {path}')
        return current

    def _validate_staging(self):
        for name, row in self._state['staging'].items():
            info = _regular(Path(name))
            if info is not None and (info.st_dev, info.st_ino) != (row['dev'], row['ino']):
                raise FileRecoveryError(f'Texture staging file was replaced; recovery retained: {name}')

    def _remove_staging(self, path):
        row = self._state['staging'][str(path)]
        info = _regular(path)
        if info is not None:
            if (info.st_dev, info.st_ino) != (row['dev'], row['ino']):
                raise FileRecoveryError(f'Texture staging file was replaced; recovery retained: {path}')
            path.unlink()
            _sync_directory(path.parent)
        del self._state['staging'][str(path)]
        self._save()

    def _cleanup_staging(self):
        self._validate_staging()
        for name in list(self._state['staging']):
            self._remove_staging(Path(name))

    @contextmanager
    def _staging(self, path, operation):
        fd, name = tempfile.mkstemp(prefix=f'.rebuilt-{self.directory.name}-{operation}-',
                                    suffix=path.suffix, dir=path.parent)
        info = os.fstat(fd)
        os.close(fd)
        staging = Path(name)
        self._state['staging'][name] = {'target': str(path), 'dev': info.st_dev, 'ino': info.st_ino}
        try:
            # Persist ownership before an encoder/copy can put bytes in this file.
            self._save()
            yield staging
        finally:
            self._remove_staging(staging)

    def _destination(self, path):
        """Reuse an existing entry's spelling without merging distinct hardlinks."""
        files = self._state['files']
        if str(path) in files:
            return path
        current = _regular(path)
        matches = []
        entries = None
        for name in files:
            recorded = Path(name)
            try:
                same_parent = path.parent.samefile(recorded.parent)
            except FileNotFoundError:
                same_parent = str(path.parent) == str(recorded.parent)
            if not same_parent:
                continue
            if path.name == recorded.name:
                matches.append(recorded)
                continue
            previous = _regular(recorded)
            if current is None or previous is None:
                # With no directory entry, case folding alone cannot establish
                # identity. Do not create an alias row or bypass a missing file
                # conflict; recovery of the earlier attempt remains available.
                if (current is None and previous is None
                        and path.name.casefold() == recorded.name.casefold()):
                    raise FileRecoveryError(
                        f'Texture destination identity is ambiguous; recovery retained: {path}')
                continue
            if (current.st_dev, current.st_ino) != (previous.st_dev, previous.st_ino):
                continue
            if entries is None:
                with os.scandir(path.parent) as listing:
                    # DirEntry.stat() reports zero inode/device on Windows.
                    entries = [(entry.name, os.stat(entry.path, follow_symlinks=False))
                               for entry in listing]
            names = [entry for entry, info in entries
                     if (info.st_dev, info.st_ino) == (current.st_dev, current.st_ino)]
            # Exact names are separate directory entries even with one inode.
            # If only one name owns the inode, both lookups are aliases of it.
            if path.name in names and recorded.name in names:
                continue
            if len(names) != 1:
                raise FileRecoveryError(
                    f'Texture destination identity is ambiguous; recovery retained: {path}')
            matches.append(recorded)
        if len(matches) > 1:
            raise FileRecoveryError(
                f'Texture destination has multiple recovery records; backups retained: {path}')
        return matches[0] if matches else path

    def write(self, path, writer):
        path = Path(path).absolute()
        if path.is_symlink():
            raise FileRecoveryError(f'Texture destination is a symbolic link: {path}')
        path = path.parent.resolve() / path.name
        with self._locked():
            if self._state['phase'] != 'active':
                raise FileRecoveryError('This texture transaction is no longer writable.')
            path = self._destination(path)
            self._cleanup_staging()
            files = self._state['files']
            row = files.get(str(path))
            if row is None:
                original = _fingerprint(path)
                backup = f'{len(files)}.backup' if original is not None else None
                if backup is not None:
                    target = self.directory / backup
                    shutil.copy2(path, target)
                    _sync_file(target)
                    if not _same_content(_fingerprint(target), original) or _fingerprint(path) != original:
                        raise FileRecoveryError(f'Texture changed while backing it up: {path}')
                row = {'original': original, 'expected': original, 'pending': None,
                       'restored': None, 'removing': False, 'backup': backup}
                files[str(path)] = row
                self._save()
            before = self._check_current(path, row)
            if self._before_first_write is not None:
                self._before_first_write()
                self._before_first_write = None
            for parent in reversed([parent for parent in path.parents if not parent.exists()]):
                try:
                    parent.mkdir()
                except FileExistsError:
                    continue
                info = parent.stat()
                self._state['folders'][str(parent)] = {'dev': info.st_dev, 'ino': info.st_ino}
                self._save()
            with self._staging(path, 'write') as staging:
                writer(staging)
                self._validate_staging()
                _sync_file(staging)
                prepared = _fingerprint(staging)
                if prepared is None:
                    raise FileRecoveryError('Texture encoder produced no file.')
                if _fingerprint(path) != before:
                    raise FileRecoveryError(f'Texture changed during encoding: {path}')
                row['expected'], row['pending'] = before, prepared
                self._save()
                if _fingerprint(path) != before:
                    raise FileRecoveryError(f'Texture changed before replacement: {path}')
                self._validate_staging()
                os.replace(staging, path)
                _sync_directory(path.parent)
                row['expected'], row['pending'] = prepared, None
                self._save()

    def _validate_rollback(self):
        if self._state['phase'] == 'committed':
            raise FileRecoveryError('This texture build already completed. Do not recover an older pending scene over it.')
        self._validate_staging()
        for name, row in self._state['files'].items():
            self._check_current(Path(name), row)
            if self._state['phase'] != 'rolled_back' and row['original'] is not None:
                backup = self.directory / row['backup']
                if not _same_content(_fingerprint(backup), row['original']):
                    raise FileRecoveryError(f'Texture backup is missing or damaged; all files retained: {backup}')

    def validate_rollback(self):
        with self._locked():
            self._validate_rollback()

    def review(self):
        """Describe validated recovery effects without writing any output file."""
        with self._locked():
            self._validate_rollback()
            done = self._state['phase'] == 'rolled_back'
            return {
                'signature': json.dumps(self._state, sort_keys=True),
                'files': [(('Already restored' if done else 'Restore') if row['original'] is not None
                           else ('Already removed' if done else 'Remove'), name)
                          for name, row in sorted(self._state['files'].items())],
            }

    def rollback(self):
        with self._locked():
            self._validate_rollback()
            self._cleanup_staging()
            if self._state['phase'] == 'rolled_back':
                return
            self._state['phase'] = 'rolling_back'
            self._save()
            for name, row in self._state['files'].items():
                path = Path(name)
                current = self._check_current(path, row)
                if row['original'] is None:
                    row['removing'] = True
                    self._save()
                    self._check_current(path, row)
                    path.unlink(missing_ok=True)
                    if current is not None:
                        _sync_directory(path.parent)
                    row['restored'] = None
                elif _same_content(current, row['original']):
                    row['restored'] = current
                else:
                    with self._staging(path, 'restore') as staging:
                        shutil.copy2(self.directory / row['backup'], staging)
                        self._validate_staging()
                        _sync_file(staging)
                        restored = _fingerprint(staging)
                        if not _same_content(restored, row['original']):
                            raise FileRecoveryError(f'Texture backup changed during restoration: {path}')
                        row['restored'] = restored
                        self._save()
                        self._check_current(path, row)
                        self._validate_staging()
                        os.replace(staging, path)
                        _sync_directory(path.parent)
                self._save()
            self._state['phase'] = 'rolled_back'
            self._save()
            self._discard_backups()
            for name in sorted(self._state['folders'], key=lambda p: len(Path(p).parts), reverse=True):
                folder = Path(name)
                if folder.is_dir() and not folder.is_symlink() and not any(folder.iterdir()):
                    info = folder.stat()
                    if {'dev': info.st_dev, 'ino': info.st_ino} == self._state['folders'][name]:
                        folder.rmdir()

    def _discard_backups(self):
        for row in self._state['files'].values():
            if row['backup'] is not None:
                (self.directory / row['backup']).unlink(missing_ok=True)
        _sync_directory(self.directory)

    def validate_commit(self):
        with self._locked():
            if self._state['phase'] != 'active':
                raise FileRecoveryError('This texture transaction is no longer writable.')
            self._validate_staging()
            for name, row in self._state['files'].items():
                self._check_current(Path(name), row)

    @property
    def phase(self):
        """Read the persisted phase, never infer commitment from an attempted save."""
        with self._locked():
            return self._state['phase']

    def _confirm_terminal_durability(self):
        try:
            _sync_file(self.directory / 'journal.json')
            _sync_directory(self.directory)
        except OSError as error:
            raise FileCommitUncertain(
                f'Terminal texture journal durability cannot be confirmed. Files '
                f'and backups were retained for review: {self.directory}') from error

    def _commit_locked(self):
        if self._state['phase'] == 'committed':
            self._confirm_terminal_durability()
            return
        if self._state['phase'] != 'active':
            raise FileRecoveryError('This texture transaction is no longer writable.')
        self._cleanup_staging()
        for name, row in self._state['files'].items():
            self._check_current(Path(name), row)
        self._state['phase'] = 'committed'
        try:
            self._save()
        except Exception as error:
            # A write can fail before or after the atomic journal replacement.
            # Read the actual decision while still holding the journal lease.
            try:
                self._read()
            except Exception as read_error:
                raise FileCommitUncertain(
                    f'Texture commit outcome cannot be determined. Files and backups '
                    f'were retained for review: {self.directory}') from read_error
            if self._state['phase'] != 'committed':
                raise error
            # Replacement can be visible even when its directory sync failed.
            # Confirm the durability barrier before acknowledging the decision.
            self._confirm_terminal_durability()

    def commit(self):
        """Persist success while retaining backups for postcommit retirement."""
        with self._locked():
            self._commit_locked()

    def _cleanup_locked(self):
        if self._state['phase'] not in ('committed', 'rolled_back'):
            raise FileRecoveryError('Texture transaction is not complete; backups retained.')
        self._cleanup_staging()
        self._confirm_terminal_durability()
        self._discard_backups()

    def cleanup(self):
        """Retire owned backups after a known terminal decision; safe to retry."""
        with self._locked():
            self._cleanup_locked()

    def close(self):
        """Complete and clean up; retain the marker needed by older saved scenes."""
        with self._locked():
            if self._state['phase'] == 'rolling_back':
                raise FileRecoveryError('Texture rollback is incomplete; backups retained.')
            if self._state['phase'] == 'active':
                self._commit_locked()
            self._cleanup_locked()
