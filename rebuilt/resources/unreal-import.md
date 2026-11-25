# Unreal import

Drag `{model}` into Unreal's Content Browser and click Import, keeping material
and texture import enabled. Then place the imported Static Mesh in your level.
The GLB contains the model, materials and textures; no manual connections or
separate texture imports are needed. Keep actor scale at 1.

This uses the native Interchange importer in Unreal 5.4 or newer. Projects that
have disabled Interchange must enable it. Import the GLB through the editor;
do not copy it directly into the project's Content directory.

Lighting, exposure, tone mapping and shader differences affect appearance.
