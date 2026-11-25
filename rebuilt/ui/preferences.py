"""Installation actions in Blender's expanded Add-ons preferences entry."""

from pathlib import Path

import bpy


_ADDON_MODULE = __package__.rpartition(".")[0]
_INSTALL_DIRECTORY = str(Path(__file__).parent.parent)


class REBUILT_Preferences(bpy.types.AddonPreferences):
    bl_idname = _ADDON_MODULE

    bake_memory_limit_gib: bpy.props.FloatProperty(
        name="Bake memory limit (GiB)",
        description=("0 uses 80% of installed RAM. Checks estimated bake memory before "
                     "starting; this is not a hard limit on Blender or GPU memory use"),
        default=0.0, min=0.0, max=4096.0, precision=1,
    )

    def draw(self, context):
        column = self.layout.column(align=True)
        column.label(text="Rebuilt · Retopo · UV · Bake")
        column.label(text="Turn complex models into clean production assets")
        column.separator()
        column.prop(self, "bake_memory_limit_gib")
        column.label(text="0 = Automatic (80% of installed RAM)")
        column.separator()
        column.operator(
            "wm.path_open",
            text="Open Installation Folder",
            icon="FILE_FOLDER",
        ).filepath = _INSTALL_DIRECTORY



def register() -> None:
    bpy.utils.register_class(REBUILT_Preferences)


def unregister() -> None:
    bpy.utils.unregister_class(REBUILT_Preferences)
