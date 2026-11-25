"""Conservative bake planning estimates; these are not allocator-enforced caps."""
from dataclasses import dataclass
from operator import index

GIB = 1024**3


def _integer(value, name, *, minimum=0):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return result


@dataclass(frozen=True)
class BakeMemoryEstimate:
    resolution: int
    map_count: int
    raster_bytes: int
    map_bytes: int
    working_bytes: int
    geometry_bytes: int
    host_bytes: int

    @property
    def texture_bytes(self):
        return self.raster_bytes + self.map_bytes + self.working_bytes

    @property
    def total_bytes(self):
        return self.texture_bytes + self.geometry_bytes + self.host_bytes + 32 * 1024**2


def estimate_bake_memory(resolution, map_count, *, geometry_bytes, host_bytes):
    """Account for retained maps/raster, working arrays, mesh work and host allowance.

    Working storage includes raster/probe/retry/replay/fill temporaries. Arbitrary
    source shaders, modifiers and driver allocations can exceed this estimate.
    """
    size = _integer(resolution, 'resolution', minimum=1)
    count = _integer(map_count, 'map count')
    mesh = _integer(geometry_bytes, 'geometry bytes')
    host = _integer(host_bytes, 'host bytes')
    texels = size * size
    return BakeMemoryEstimate(size, count, 33 * texels, 16 * count * texels,
                              384 * texels, 8 * mesh, host)


def automatic_budget(installed_bytes):
    """Leave a fifth of installed capacity outside the planned job budget."""
    return _integer(installed_bytes, 'installed memory', minimum=1) * 4 // 5


def require_budget(estimate, budget_bytes):
    limit = _integer(budget_bytes, 'memory limit', minimum=1)
    if estimate.total_bytes > limit:
        raise ValueError(
            f'Baking {estimate.resolution} × {estimate.resolution} with '
            f'{estimate.map_count} maps is estimated to need '
            f'{estimate.total_bytes / GIB:.2f} GiB; the bake memory limit is '
            f'{limit / GIB:.2f} GiB. Choose a lower resolution or fewer maps, '
            'or set a suitable Bake memory limit in Rebuilt addon preferences '
            'if your machine has enough capacity.'
        )
