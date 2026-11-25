"""Stage dependency graph and cache invalidation."""

from __future__ import annotations

from dataclasses import dataclass


class StageGraphError(ValueError):
    """The declared graph is malformed."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    depends_on: tuple[str, ...] = ()


PIPELINE: tuple[StageSpec, ...] = (
    StageSpec("ingest"),
    StageSpec("cleanup", ("ingest",)),
    StageSpec("analyze", ("cleanup",)),
    StageSpec("retopo", ("analyze",)),
    StageSpec("seams", ("retopo",)),
    StageSpec("uv", ("seams",)),
    StageSpec("bake", ("uv",)),
    StageSpec("qa", ("bake",)),
    StageSpec("output", ("qa",)),
)


class StageGraph:
    """A validated, topologically ordered set of stages."""

    def __init__(self, specs: tuple[StageSpec, ...] = PIPELINE) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise StageGraphError("duplicate stage name")

        for spec in specs:
            for dependency in spec.depends_on:
                if dependency not in self._specs:
                    raise StageGraphError(
                        f"stage {spec.name!r} depends on unknown stage {dependency!r}"
                    )

        self._order = self._topological_order()

    def _topological_order(self) -> tuple[str, ...]:
        """Kahn's algorithm; leftover nodes mean a cycle."""
        remaining = {name: set(spec.depends_on) for name, spec in self._specs.items()}
        ordered: list[str] = []

        while remaining:
            ready = sorted(name for name, deps in remaining.items() if not deps)
            if not ready:
                raise StageGraphError(f"dependency cycle among {sorted(remaining)}")
            for name in ready:
                ordered.append(name)
                del remaining[name]
            for deps in remaining.values():
                deps.difference_update(ready)

        return tuple(ordered)

    @property
    def order(self) -> tuple[str, ...]:
        """Stage names in an order where dependencies always come first."""
        return self._order

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._specs)

    def spec(self, name: str) -> StageSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise StageGraphError(f"unknown stage {name!r}") from None

    def dependencies(self, name: str) -> tuple[str, ...]:
        return self.spec(name).depends_on

    def downstream(self, name: str) -> frozenset[str]:
        """Every stage that transitively depends on `name`, excluding itself."""
        self.spec(name)
        found: set[str] = set()
        frontier = {name}
        while frontier:
            frontier = {
                other
                for other, spec in self._specs.items()
                if other not in found and frontier & set(spec.depends_on)
            }
            found |= frontier
        return frozenset(found)

    def stages_to_run(
        self,
        current_keys: dict[str, str],
        cached_keys: dict[str, str],
        force: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        """Which stages need running, in dependency order."""
        unknown = set(current_keys) - self.names
        if unknown:
            raise StageGraphError(f"current_keys mentions unknown stages: {sorted(unknown)}")
        missing = self.names - set(current_keys)
        if missing:
            raise StageGraphError(f"current_keys is missing stages: {sorted(missing)}")
        unknown_force = force - self.names
        if unknown_force:
            raise StageGraphError(f"force mentions unknown stages: {sorted(unknown_force)}")

        dirty: set[str] = set()
        for name in self._order:
            settings_changed = cached_keys.get(name) != current_keys[name]
            upstream_rerun = bool(dirty & set(self._specs[name].depends_on))
            if name in force or settings_changed or upstream_rerun:
                dirty.add(name)

        return tuple(name for name in self._order if name in dirty)
