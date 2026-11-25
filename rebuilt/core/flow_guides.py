"""Isolated guided remesher process. This module never accesses Blender data."""

import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

from .hardware import available_cpu_count
from .types import MeshArrays

REVISION = "fef338d-reference-steering-3"


def binary_path():
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return None
    path = Path(__file__).resolve().parents[1] / "bin/flow_guides/macos-arm64/FlowRetopo"
    return path if path.is_file() else None


def run(payload, target, strength, progress, *, timeout=300, binary=None):
    """Return validated mesh arrays and native diagnostics, or preserve the prior result by failing."""
    progress.check_cancelled()
    binary = binary or binary_path()
    if binary is None:
        raise RuntimeError("Flow Guides requires the Apple Silicon build on this platform.")
    if not 100 <= target <= 100000:
        raise ValueError("Flow Guides supports 100–100,000 quads. Adjust Quads before generating.")
    if not payload["samples"]:
        raise ValueError("Draw at least one guide, or turn off Flow Guides.")
    if not os.access(binary, os.X_OK):
        binary.chmod(binary.stat().st_mode | 0o111)
    with tempfile.TemporaryDirectory(prefix="rebuilt_flow_") as folder:
        folder = Path(folder)
        source, output, guides = (folder / name for name in ("source.obj", "result.obj", "guides.txt"))
        with source.open("w") as stream:
            for co in payload["vertices"]:
                stream.write("v " + " ".join(format(float(v), ".17g") for v in co) + "\n")
            for face in payload["triangles"]:
                stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")
        with guides.open("w") as stream:
            for row in payload["samples"]:
                stream.write(" ".join(format(float(v), ".17g") for v in row[:-1]) + f" {int(row[-1])}\n")
        threads = min(64, available_cpu_count())
        command = [str(binary), str(source), str(output), str(guides), str(target), str(threads), str(strength)]
        options = payload.get("engine_options", {})
        command += [str(options.get("sharp", 90)), str(int(options.get("cloth_folds", True))),
                    str(options.get("adaptivity", 1)), str(options.get("anisotropy", 1)),
                    options.get("model", "organic")]
        started = time.monotonic()
        with (folder / "engine.log").open("w+") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    progress.update(min(.85, .15 + (time.monotonic() - started) / timeout * .7), "Building guided quads")
                    if time.monotonic() - started > timeout:
                        raise RuntimeError("Flow Guides timed out. The previous mesh is kept; try fewer guides.")
                    time.sleep(.05)
                progress.check_cancelled()
                if process.returncode != 0 or not output.is_file():
                    log.seek(0)
                    errors = [line.strip() for line in log if line.startswith("FLOW_ERROR")]
                    raise RuntimeError(errors[-1] if errors else "Flow Guides could not build a mesh. The previous result is kept.")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        vertices, faces = [], []
        for line in output.read_text().splitlines():
            row = line.split()
            if row and row[0] == "v":
                vertices.append([float(v) for v in row[1:4]])
            elif row and row[0] == "f":
                faces.append([int(v.split("/")[0]) - 1 for v in row[1:]])
        if not vertices or not faces:
            raise ValueError("Flow Guides returned an empty mesh.")
        positions = np.asarray(vertices, dtype=np.float64)
        if positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("Flow Guides returned invalid coordinates.")
        positions = positions * payload["scale"] + np.asarray(payload["center"])
        inverse = np.asarray(payload["inverse_matrix"])
        positions = positions @ inverse[:3, :3].T + inverse[:3, 3]
        if np.linalg.det(inverse[:3, :3]) < 0:
            faces = [list(reversed(f)) for f in faces]
        # Extraction can emit the same vertex set twice, even with a different
        # winding/order. Match Blender's duplicate-face rule, retaining the first.
        unique, seen = [], set()
        for face in faces:
            key = tuple(sorted(face))
            if key not in seen:
                seen.add(key)
                unique.append(face)
        duplicate_faces = len(faces) - len(unique)
        faces = unique
        sizes = np.asarray([len(f) for f in faces], dtype=np.int32)
        arrays = MeshArrays(positions, np.asarray([v for f in faces for v in f], dtype=np.int32),
                            np.concatenate(([0], np.cumsum(sizes)[:-1])), sizes)
        diagnostics = json.loads(Path(str(output) + ".json").read_text())
        diagnostics.update(native_faces=diagnostics.get("faces"), faces=arrays.num_faces,
                           duplicate_faces_removed=duplicate_faces)
        diagnostics.update(revision=REVISION, threads=threads, guides=payload["guide_count"],
                           elapsed_seconds=time.monotonic() - started)
        progress.update(.9, "Checking guided mesh")
        return arrays, diagnostics
