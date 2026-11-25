"""Suggest useful bake channels from the source material's visible surface."""

from .materials import find_colour_source
from ..core.bake.channels import EXTRA_CHANNELS

BASE_MAPS = frozenset({"NORMAL", "BASE_COLOR", "AO"})


def suggested_maps(source) -> set[str]:
    """Inspect assigned shaders, never infer channel meaning from image names."""
    maps = set(BASE_MAPS)
    values = {channel: [] for channel in EXTRA_CHANNELS}
    for slot in source.material_slots:
        material = slot.material
        color = find_colour_source(material) if material else None
        if color is None:
            continue
        shader = color.node
        if shader.type == "BSDF_PRINCIPLED":
            for channel, (_, name, _, default) in EXTRA_CHANNELS.items():
                socket = shader.inputs.get(name)
                if socket is None:
                    continue
                if socket.is_linked:
                    maps.add(channel)
                else:
                    value = socket.default_value
                    values[channel].append(tuple(value) if hasattr(value, "__len__") else value)
            for channel, name, default in (
                ("ROUGHNESS", "Roughness", 0.5),
                ("METALLIC", "Metallic", 0.0),
                ("OPACITY", "Alpha", 1.0),
            ):
                socket = shader.inputs.get(name)
                if socket and (socket.is_linked or abs(socket.default_value - default) > 1e-6):
                    maps.add(channel)
        elif shader.type == "EMISSION":
            output = next(
                (
                    n
                    for n in material.node_tree.nodes
                    if n.type == "OUTPUT_MATERIAL" and n.is_active_output
                ),
                None,
            )
            surface = output.inputs["Surface"] if output else None
            root = surface.links[0].from_node if surface and surface.is_linked else None
            if root is None or root.type != "MIX_SHADER":
                continue
            branches = [s.links[0].from_node if s.is_linked else None for s in root.inputs[1:]]
            factor = root.inputs[0]
            camera_gate = (
                factor.is_linked
                and factor.links[0].from_node.type == "LIGHT_PATH"
                and factor.links[0].from_socket.name == "Is Camera Ray"
            )
            if (
                shader in branches
                and any(n and n.type == "BSDF_TRANSPARENT" for n in branches)
                and not camera_gate
            ):
                alpha = (
                    1.0 - factor.default_value if branches[0] == shader else factor.default_value
                )
                if factor.is_linked or alpha < 1.0 - 1e-6:
                    maps.add("OPACITY")

    for channel, constants in values.items():
        if len(set(constants)) > 1:
            maps.add(channel)
    return maps


def material_defaults(source) -> dict:
    """Copy only constants shared by all visible Principled source shaders."""
    shaders = []
    for slot in source.material_slots:
        color = find_colour_source(slot.material) if slot.material else None
        if color is None or color.node.type != "BSDF_PRINCIPLED":
            return {}
        shaders.append(color.node)
    if not shaders:
        return {}
    result = {}
    for socket in shaders[0].inputs:
        if socket.type not in {"VALUE", "RGBA", "BOOLEAN"} or socket.name == "Weight":
            continue
        matches = [s.inputs.get(socket.name) for s in shaders]
        if any(s is None or s.is_linked or not hasattr(s, "default_value") for s in matches):
            continue
        values = [
            list(s.default_value) if hasattr(s.default_value, "__len__") else s.default_value
            for s in matches
        ]
        if all(value == values[0] for value in values):
            result[socket.name] = values[0]
    return result
