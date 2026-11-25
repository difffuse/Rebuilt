"""Normalize common photogrammetry materials on pipeline-owned copies only."""

from __future__ import annotations

import bpy

MATERIAL_PROTOCOL_REVISION = 2


COLOUR_SOCKETS: tuple[tuple[str, str], ...] = (
    ("BSDF_PRINCIPLED", "Base Color"),
    ("EMISSION", "Color"),
    ("BSDF_DIFFUSE", "Color"),
    ("BSDF_TOON", "Color"),
    ("BSDF_GLOSSY", "Color"),
    ("BSDF_TRANSLUCENT", "Color"),
    ("BSDF_SHEEN", "Color"),
    ("SUBSURFACE_SCATTERING", "Color"),
    ("BACKGROUND", "Color"),
)


def find_colour_source(material: bpy.types.Material) -> bpy.types.NodeSocket | None:
    """The socket carrying this material's albedo, whatever it is built from."""

    tree = getattr(material, "node_tree", None)
    if tree is None:
        return None

    output = next(
        (n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
        None,
    ) or next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        return None

    known = dict(COLOUR_SOCKETS)
    order = {node_type: index for index, (node_type, _) in enumerate(COLOUR_SOCKETS)}

    queue: list[tuple[bpy.types.Node, int]] = []
    surface = output.inputs.get("Surface")
    if surface is not None and surface.is_linked:
        queue.append((surface.links[0].from_node, 0))

    seen: set[str] = set()
    found: list[tuple[int, int, bpy.types.NodeSocket]] = []

    while queue:
        node, depth = queue.pop(0)
        if node.name in seen:
            continue
        seen.add(node.name)

        name = known.get(node.type)
        if name is not None:
            socket = node.inputs.get(name)
            if socket is not None:
                found.append((depth, order[node.type], socket))
                continue

        for candidate in node.inputs:
            if candidate.is_linked and candidate.type == "SHADER":
                queue.append((candidate.links[0].from_node, depth + 1))

    if not found:
        return None
    found.sort(key=lambda entry: (entry[0], entry[1]))
    return found[0][2]


def normalize_scan_materials(obj) -> list[dict]:
    """Convert reachable Emission colors; preserve Principled and unknown trees."""
    reports = []
    converted = {}
    for index, material in enumerate(list(obj.data.materials)):
        if material is None:
            reports.append({"slot": index, "status": "missing"})
            continue
        color = find_colour_source(material)
        if color is None:
            reports.append(
                {
                    "slot": index,
                    "source": material.name,
                    "status": "unsupported",
                    "reason": "No reachable surface color; inspect the material before baking.",
                }
            )
            continue
        if color.node.type == "BSDF_PRINCIPLED":
            reports.append({"slot": index, "source": material.name, "status": "principled"})
            continue
        if color.node.type != "EMISSION":
            reports.append(
                {
                    "slot": index,
                    "source": material.name,
                    "status": "preserved",
                    "shader": color.node.type,
                }
            )
            continue
        active_output = next(
            (
                n
                for n in material.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output
            ),
            None,
        )
        surface = active_output.inputs["Surface"] if active_output else None
        root = surface.links[0].from_node if surface and surface.is_linked else None
        branches = (
            [s.links[0].from_node if s.is_linked else None for s in root.inputs[1:]]
            if root is not None and root.type == "MIX_SHADER"
            else []
        )
        supported = root == color.node or (
            color.node in branches
            and any(n is not None and n.type == "BSDF_TRANSPARENT" for n in branches)
        )
        if not supported:
            reports.append(
                {
                    "slot": index,
                    "source": material.name,
                    "status": "unsupported",
                    "reason": "Complex mixed Emission material was preserved; review its color before baking.",
                }
            )
            continue
        if material not in converted:
            copy = material.copy()
            copy.name = f"{material.name}_rebuilt_principled"
            tree = copy.node_tree
            original = tree.nodes[color.node.name]
            output = next(
                n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
            )
            bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.label = "Scan appearance · Principled conversion"
            bsdf.location = (output.location.x - 320, output.location.y)
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Roughness"].default_value = 0.6
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
            original_color = original.inputs["Color"]
            strength = original.inputs["Strength"]
            if strength.is_linked or abs(strength.default_value - 1.0) > 1e-6:
                multiply = tree.nodes.new("ShaderNodeMixRGB")
                multiply.blend_type = "MULTIPLY"
                multiply.inputs[0].default_value = 1.0
                if original_color.is_linked:
                    tree.links.new(original_color.links[0].from_socket, multiply.inputs[1])
                else:
                    multiply.inputs[1].default_value = original_color.default_value
                if strength.is_linked:
                    tree.links.new(strength.links[0].from_socket, multiply.inputs[2])
                else:
                    multiply.inputs[2].default_value = (strength.default_value,) * 3 + (1.0,)
                tree.links.new(multiply.outputs[0], bsdf.inputs["Base Color"])
            elif original_color.is_linked:
                tree.links.new(original_color.links[0].from_socket, bsdf.inputs["Base Color"])
            else:
                bsdf.inputs["Base Color"].default_value = original_color.default_value
            surface = output.inputs["Surface"]
            wrapper = surface.links[0].from_node if surface.is_linked else None
            if wrapper is not None and wrapper.type == "MIX_SHADER":
                branches = [
                    s.links[0].from_node if s.is_linked else None for s in wrapper.inputs[1:]
                ]
                factor = wrapper.inputs[0]
                camera_gate = (
                    factor.is_linked
                    and factor.links[0].from_node.type == "LIGHT_PATH"
                    and factor.links[0].from_socket.name == "Is Camera Ray"
                )
                if (
                    not camera_gate
                    and original in branches
                    and any(n is not None and n.type == "BSDF_TRANSPARENT" for n in branches)
                ):
                    if factor.is_linked:
                        alpha = factor.links[0].from_socket
                        if branches[0] == original:
                            invert = tree.nodes.new("ShaderNodeMath")
                            invert.operation = "SUBTRACT"
                            invert.inputs[0].default_value = 1.0
                            tree.links.new(alpha, invert.inputs[1])
                            alpha = invert.outputs[0]
                        tree.links.new(alpha, bsdf.inputs["Alpha"])
                    else:
                        bsdf.inputs["Alpha"].default_value = (
                            1.0 - factor.default_value
                            if branches[0] == original
                            else factor.default_value
                        )
            tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
            copy["rebuilt_material_protocol"] = MATERIAL_PROTOCOL_REVISION
            copy["rebuilt_original_shader"] = "EMISSION"
            converted[material] = copy
        obj.data.materials[index] = converted[material]
        reports.append(
            {
                "slot": index,
                "source": material.name,
                "material": converted[material].name,
                "status": "converted",
                "from": "EMISSION",
                "to": "PRINCIPLED",
                "protocol_revision": MATERIAL_PROTOCOL_REVISION,
                "note": "Color and UV network preserved; Principled responds to scene lighting.",
            }
        )
    return reports
