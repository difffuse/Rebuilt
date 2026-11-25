"""User interface layer: panels, properties and operators."""

from . import flow_guides, icons, operators, panels, preferences, properties


_MODULES = (icons, properties, flow_guides, operators, panels, preferences)


def register() -> None:
    for module in _MODULES:
        module.register()


def unregister() -> None:
    for module in reversed(_MODULES):
        module.unregister()
