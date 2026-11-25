"""Bleed baked colors into empty texels without full-frame RGBA temporaries."""

import numpy as np


def repair_projection(values, written, labels, size, neutral, *, normal=False, passes=64):
    """Fill failed surface texels inside their UV island; retain measured pixels exactly."""
    filled = values.copy()
    known = written & np.isfinite(values).all(axis=1)
    missing = (labels >= 0) & ~known
    selected = np.flatnonzero(missing)
    remaining = selected.copy()
    for _ in range(passes):
        if not remaining.size:
            break
        y, x = np.divmod(remaining, size)
        total = np.zeros((len(remaining), values.shape[1]), dtype=values.dtype)
        count = np.zeros(len(remaining), dtype=np.int32)
        for delta, inside in ((-size, y > 0), (size, y < size - 1),
                              (-1, x > 0), (1, x < size - 1)):
            ids = np.clip(remaining + delta, 0, len(values) - 1)
            usable = inside & known[ids] & (labels[ids] == labels[remaining])
            total[usable] += filled[ids[usable]]
            count += usable
        usable = count > 0
        if not usable.any():
            break
        filled[remaining[usable]] = total[usable] / count[usable, None]
        known[remaining[usable]] = True
        remaining = remaining[~usable]
    filled[remaining] = neutral
    if normal and selected.size:
        vectors = filled[selected, :3] * 2 - 1
        lengths = np.linalg.norm(vectors, axis=1)
        vectors[lengths < 1e-8] = (0, 0, 1)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1), 1e-8)[:, None]
        filled[selected, :3] = vectors * 0.5 + 0.5
    return filled, missing, {"approximated_texels": int(selected.size),
                             "neutral_texels": int(remaining.size)}


def fill_from_neighbours(values, missing, size, passes=8):
    """Average four known neighbours at each successive one-texel front."""
    if not missing.any() or missing.all():
        return values
    filled = values.copy()
    hole = missing.reshape(size, size).copy()
    for _ in range(passes):
        if not hole.any():
            break

        known = ~hole
        adjacent = np.roll(known, 1, 0) | np.roll(known, -1, 0)
        adjacent |= np.roll(known, 1, 1) | np.roll(known, -1, 1)
        target = np.flatnonzero(hole & adjacent)
        if not len(target):
            break
        y, x = np.divmod(target, size)
        total = np.zeros((len(target), values.shape[-1]), dtype=values.dtype)
        count = np.zeros(len(target), dtype=np.float32)
        for ids in (
            ((y - 1) % size) * size + x,
            ((y + 1) % size) * size + x,
            y * size + (x - 1) % size,
            y * size + (x + 1) % size,
        ):
            has = known.reshape(-1)[ids]
            total += filled[ids] * has[:, None]
            count += has
        filled[target] = total / count[:, None]
        hole.reshape(-1)[target] = False
    return filled
