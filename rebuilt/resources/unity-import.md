# {asset}: Unity {pipeline}

Copy this entire `{asset}` folder into your Unity project's **Assets** folder,
including every `.meta` file. You can drag the whole folder from Finder or Explorer
into Unity's Project window. Then drag the FBX into your scene.
The material is already assigned.
No manual texture connections or Extract Materials step is required.

The folder contains the FBX, native Unity material and only the required textures.
Keep the `.meta` files beside their assets: they preserve material references,
normal-map interpretation and color-space settings. Do not import only the FBX.
The hidden export manifest tracks generated files for subsequent exports.

Base Color, Normal, ambient occlusion, metallic/smoothness, opacity and emission
are configured when present. The exporter packs the surface channels, places
opacity in albedo alpha, combines emission strength with emission color and
selects the correct shader and render state. Non-default specular/IOR/tint values
use a specular workflow. URP and HDRP also support their native coat controls.

The Transform uses rotation (0, 0, 0) and scale (1, 1, 1). Blender Z up becomes
Unity Y up; Blender -Y forward becomes Unity +Z forward. New texture metadata
marks normals as Normal maps and data as linear.

Re-exporting to the same folder preserves asset GUIDs and updates the generated
material. Keep custom material variants in separate files if you want to retain
manual artistic changes across re-exports.

Choose the matching Render Pipeline in Rebuilt before exporting. URP and HDRP
must already be installed and configured in the destination project; this export does not change project-wide renderer settings.

Unity lighting, exposure and tone mapping affect appearance. Its native shaders
do not reproduce every Principled feature exactly: Built-in Standard has no
clearcoat or subsurface model, URP has no native subsurface/transmission equivalent,
and HDRP Lit does not expose Blender's coat roughness. The folder includes
textures referenced by the material; these limitations are shader differences,
not missing references.

- [Unity material references](https://docs.unity3d.com/6000.0/Documentation/Manual/FBXImporter-Materials.html)
- [URP Lit](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/lit-shader.html)
- [HDRP mask channels](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/Mask-Map-and-Detail-Map.html)
