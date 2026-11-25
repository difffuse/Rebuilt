"""Pipeline stages, in PRD order: ingest, cleanup, analyze, retopo, uv, bake, qa, output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.context import AssetContext
    from .base import Stage

from ..core.graph import StageGraph
from .analyze import AnalyzeStage
from .bake import BakeStage
from .cleanup import CleanupStage
from .ingest import IngestStage
from .output import OutputStage
from .qa import QAStage
from .retopo import RetopoStage
from .seams import SeamsStage
from .uvs import UVStage


REGISTRY: tuple[type, ...] = (
    IngestStage,
    CleanupStage,
    AnalyzeStage,
    RetopoStage,
    SeamsStage,
    UVStage,
    BakeStage,
    QAStage,
    OutputStage,
)

_BY_NAME = {cls.name: cls for cls in REGISTRY}


def all_stages() -> list[Stage]:
    """One fresh instance of every stage, in order."""
    return [cls() for cls in REGISTRY]


def cache_keys(recipe, stages=None, runtime=None, *, ctx=None) -> dict[str, str]:
    """Share verified runtime and optional live scene inputs across stage keys."""
    selected = all_stages() if stages is None else list(stages)
    if runtime is None and any(stage.runtime_engines(recipe) for stage in selected):
        from ..runtime import manifest
        runtime = manifest.load_runtime_manifest()
    keys = {stage.name: (stage.cache_key(recipe, runtime=runtime)
                        if stage.runtime_engines(recipe) else stage.cache_key(recipe))
            for stage in selected}
    if ctx is not None and {'uv', 'bake'} & keys.keys():
        from ..bridge.cache_inputs import uv_cache_inputs
        inputs = uv_cache_inputs(ctx.artifact('low'))
        for name in inputs.keys() & keys.keys():
            keys[name] += ':' + inputs[name]
    return keys


def build_pipeline(
    ctx: AssetContext,
    force: frozenset[str] = frozenset(),
    graph: StageGraph | None = None,
    *, recipe=None, cached_keys=None,
) -> list[Stage]:
    """The stages this asset actually needs to run."""
    stages = all_stages()
    graph = graph or StageGraph()

    recipe = ctx.recipe if recipe is None else recipe
    current = cache_keys(recipe, stages, ctx=ctx)

    cached = ctx.cached_keys if cached_keys is None else cached_keys

    for name in ("seams", "uv"):
        if name in cached and (ctx.report(name) or {}).get("method") == "external":
            current[name] = cached[name]
    for name in graph.names - set(current):
        current[name] = cached.get(name, "")

    due = set(graph.stages_to_run(current, cached, force=force & graph.names))
    return [stage for stage in stages if stage.name in due]


def stage_named(name: str) -> Stage:
    """A single stage by name, for Expert mode."""
    try:
        return _BY_NAME[name]()
    except KeyError:
        raise KeyError(f"unknown stage {name!r}; have {sorted(_BY_NAME)}") from None
