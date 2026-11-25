# Unity export

Choose **URP, Built-in or HDRP** to match the destination project, then select a
parent export folder. Each export creates a folder named after the model.
Copy that entire folder into the Unity project's **Assets** directory, including
all `.meta` files, then place its FBX in a scene. The native material, textures
and import settings are included. No Extract Materials or manual connections
are required. No new `.unitypackage` is generated.

Files are prepared privately before replacing the model folder. Re-exporting
preserves FBX/material GUIDs and existing texture import settings. A hidden
`.rebuilt-export.json` tracks owned files, so unused generated textures can
be removed without deleting unrelated artist files or the other engine's export.
Keep custom material variants under separate names. A failed build or publication
restores the previous delivery and leaves the previous report intact. If filesystem
errors also prevent restoration, the backup is retained and its path reported.
Blender's working textures, materials and scene remain unchanged.

On first migration, metadata can be read from an existing sibling `.unitypackage`
or older loose sidecars. Those old files are retained. Once the new folder contains
FBX metadata, it is authoritative. Only referenced textures are delivered.
See [Unity's metadata contract](https://docs.unity3d.com/6000.0/Documentation/Manual/AssetMetadata.html):
metadata must travel with its assets to retain references and import settings.

## Diagnosis

The previous FBX left coordinate conversion in the object's Transform. Unity
compensated with about -90 degrees on X. That alone did not establish an incorrect
visual facing. The new static export bakes rotation, scale and axis conversion
into a temporary mesh, preserving world shape and the source Blender object.

Unity delivery now writes the exact evaluated Blender loop triangles to that
temporary mesh. Native Unity previously discarded six BUDDA and nine THINKER
triangles while interpreting difficult source polygons. Preserving Blender's
diagonals also keeps geometry consistent with baked tangent-space maps. Source
quads remain editable, and UVs, material indices and corner colors follow the
original triangle corners. Normals use inverse-transpose transport for nonuniform
scale and preserve corner correspondence under mirrored transforms. Blender's
custom-normal encoding introduces bounded quantization (below one degree on the
three real regression assets); the source normals are unchanged.

Blender's FBX material wrapper cannot follow the add-on's Base Color/AO Multiply
node. This omitted the diffuse texture connection. Stripped image paths also
referenced different filenames from the engine-specific copies.

A native Unity reproduction imported the previous FBX with a material but no
connected textures. Extraction succeeded in that reproduction. The colleague's
empty extraction folder cannot be diagnosed conclusively from the screenshot;
material remaps, import options or the exact exported file can differ.

## Automatic material conversion

| Blender channel | URP / Built-in | HDRP |
|---|---|---|
| Base Color | Base Map / Albedo | Base Color Map |
| Normal | Normal map | Tangent-space Normal Map |
| Metallic, AO, Roughness | Packed R metallic, G AO, A `1 - roughness`; both metallic and occlusion slots assigned | Mask Map with the same R/G/A channels; B disables unused detail |
| Opacity | Albedo alpha and transparent render state | Base Color alpha and transparent render state |
| Emission and Strength | Combined linear HDR Emission Map | Combined linear HDR Emissive Color Map |
| Non-default IOR, specular level or tint | Converted specular reflectance and smoothness map; specular shader workflow | Specular Color material model and map |
| Coat | URP Complex Lit; Built-in Standard has no native equivalent | Native coat weight/map |

Missing surface maps retain material constants. Color tint is white for baked
albedo. Normal/data texture metadata uses the proper import type and color space.
Blender keeps its AO Multiply preview; Unity applies AO through its native
occlusion input, without multiplying it into albedo twice.

Surface masks and derived RGBA maps use lossless PNG8. Independently exported
maps keep their selected formats and precision. Emission conversion uses EXR16.
The specular conversion preserves dielectric reflectance and tint, including the
BUDDA black-tint case; it is not a full reproduction of Blender's BRDF.

Native shaders differ: Built-in Standard has no coat or subsurface model; URP
has no native Principled subsurface/transmission equivalent; HDRP Lit does not
expose Blender's coat roughness. Only referenced textures enter the folder. Scene
lighting, tone mapping and unsupported shader features can change appearance.
Automatic assignment is not a claim of pixel-identical rendering in every engine.

## Implementation contracts

- `bridge/fbx_export.py` owns evaluated scratch geometry/materials, direct FBX
  texture links and cleanup. Live mesh data, materials, image paths and selection
  remain unchanged, including on failure.
- `bridge/unity_export.py` writes the surface mask, texture metadata and guide.
- `bridge/unity_material.py` writes Unity `.mat`, FBX material remaps, derived
  color/emission/specular maps and the material-referenced texture list.
- `resources/unity-import.md` is the shipped per-asset guide template.
- `OutputRecipe.unity_pipeline` records the most recent choice. Export does not
  invalidate completed retopo, unwrap or bake stages.
- FBX and material GUIDs survive re-export. Generated material contents and remaps
  update; keep user-authored material variants in separate files. Existing texture
  metadata is preserved, so old custom import overrides remain user choices.
- Unity/glTF settings in old files migrate to FBX on load; the export entry point
  also normalizes older callers.

Blender's float-image PNG writer unpremultiplies RGB by alpha. Channel-packed maps
therefore use a byte buffer and `Image.save()`. Tests cover nonzero/zero smoothness
and albedo opacity without changing RGB. Emission stays linear and HDR.

Legacy package metadata remains readable for migration. Folder delivery contains
no Unity scripts or third-party importer.

## References

- [Unity model axes](https://docs.unity3d.com/6000.0/Documentation/Manual/FBXImporter-Model.html)
- [Unity material references](https://docs.unity3d.com/6000.0/Documentation/Manual/FBXImporter-Materials.html)
- [URP Lit](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/lit-shader.html)
- [HDRP mask channels](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/Mask-Map-and-Detail-Map.html)
- [Unity glTFast import](https://github.com/Unity-Technologies/com.unity.cloud.gltfast/blob/main/Packages/com.unity.cloud.gltfast/Documentation~/ImportEditor.md)
