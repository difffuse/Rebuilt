"""Keep native QuadriFlow failures outside the user's Blender process."""

from pathlib import Path
import json
import tempfile

import numpy as np

from ..core.jobs import Cancelled
from ..core.types import MeshArrays
from ..runtime.process import ProcessStartError, run_process


def remesh(arrays, recipe, target, *, passes, tolerance, timeout=900.0,
           binary_path=None, cancelled=None):
    if binary_path is None:
        # Direct script callers run on Blender's main thread. Job workers receive
        # the executable path with their prepared arrays and never access bpy.
        import bpy
        binary_path = bpy.app.binary_path
    with tempfile.TemporaryDirectory(prefix="rebuilt_qf_") as folder:
        root = Path(folder)
        np.savez(
            root / "input.npz",
            positions=arrays.positions,
            corner_verts=arrays.corner_verts,
            face_starts=arrays.face_starts,
            face_sizes=arrays.face_sizes,
        )
        options = dict(
            target=target,
            passes=passes,
            tolerance=tolerance,
            preserve_sharp=recipe.preserve_sharp,
            preserve_boundary=recipe.preserve_boundary,
            seed=recipe.seed,
        )
        (root / "options.json").write_text(json.dumps(options))
        command = [
            binary_path,
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).with_name("quadriflow_worker.py")),
            "--",
            str(root),
        ]
        try:
            result = run_process(command, timeout=timeout, cancelled=cancelled,
                                 max_stdout=64 * 1024, max_stderr=64 * 1024)
        except ProcessStartError as exc:
            print(f"Rebuilt: isolated QuadriFlow did not complete: {exc}", flush=True)
            return None
        if result.cancelled:
            raise Cancelled()
        if result.timed_out:
            print(f"Rebuilt: isolated QuadriFlow timed out after {timeout:g} seconds", flush=True)
            return None
        output = root / "output.npz"
        repair_path = root / "input_repair.json"
        if repair_path.exists():
            repair = json.loads(repair_path.read_text())
            if repair["repaired"]:
                print(
                    f"Rebuilt: QuadriFlow input repaired ({repair['removed_faces']} invalid face(s) removed)",
                    flush=True,
                )
        if result.returncode or not output.exists():
            print(
                f"Rebuilt: isolated QuadriFlow declined the mesh (exit {result.returncode})",
                flush=True,
            )
            return None
        with np.load(output, allow_pickle=False) as data:
            return MeshArrays(
                **{
                    key: data[key].copy()
                    for key in ("positions", "corner_verts", "face_starts", "face_sizes")
                }
            )
