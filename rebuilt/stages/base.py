"""The contract every pipeline stage implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..core.jobs import MainThreadCalls, ProgressReporter
from ..core.recipe import AssetRecipe
from ..pipeline.context import AssetContext


class Stage(ABC):
    """One step of the pipeline."""

    name: ClassVar[str]

    slot: ClassVar[str] = "working"

    role: ClassVar[str] = ""

    label: ClassVar[str] = ""

    main_thread_calls: MainThreadCalls | None = None

    # Possible native dependencies, including recipe-dependent execution paths.
    native_engines: ClassVar[tuple[str, ...]] = ()

    def compute_on_main_thread(self, payload: dict[str, Any]) -> Any:
        """Service an isolated native solver request, when this stage needs one."""
        raise NotImplementedError(f"{self.name} has no native compute handler")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", ""):
            raise TypeError(f"{cls.__name__} must define a stage name")
        if not cls.role:
            cls.role = cls.name
        if not cls.label:
            cls.label = cls.name.replace("_", " ").title()

    def runtime_engines(self, recipe: AssetRecipe) -> tuple[str, ...]:
        return self.native_engines

    def _with_runtime(self, key: str, recipe: AssetRecipe, runtime=None) -> str:
        from ..runtime.cache import dependency_key
        return dependency_key(key, self.runtime_engines(recipe), runtime)

    def cache_key(self, recipe: AssetRecipe, *, runtime=None) -> str:
        """Fingerprint of the settings this stage actually depends on."""
        return self._with_runtime(recipe.fingerprint(), recipe, runtime)

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        """Main thread. Collect everything `compute` will need."""
        return {}

    @abstractmethod
    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        """Worker thread. Do the work. **Must not touch bpy.**"""

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        """Main thread. Write results into the scene."""
