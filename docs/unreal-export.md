# Unreal export

1. In Rebuilt, select **Unreal** and click **Export**.
2. Choose the destination folder. The add-on creates `<asset>/` containing `SM_<asset>.glb` and a short import guide.
3. Drag the GLB from that folder into Unreal's **Content Browser** and click **Import**, keeping
   material and texture import enabled.
4. Drag the imported Static Mesh into the level. Materials and textures are assigned.
   Actor scale can remain 1; dimensions have already been converted correctly.

This uses Unreal's native Interchange GLB importer, available in Unreal 5.4 with
its built-in Interchange plugins enabled by default. Artists do not need Python,
an import script, an extra add-on, loose texture files or manual material wiring.
Projects that deliberately disabled Interchange must restore the native importer.
The 0.30.33 FBX/Python bundle is no longer the Unreal export workflow. Re-export old
assets to obtain the new GLB; previous files are not deleted.

## Material and geometry contracts

- Preserve all Blender objects, materials, images, UVs, selection and settings.
  Source quads remain editable. GLB stores triangles, as expected for runtime meshes.
- Create an evaluated export copy, bake its rotation/scale, center on its origin,
  and convert Blender scene units to meters. Unreal imports meters as centimeters.
  Custom normals use inverse-transpose transport through nonuniform scale; mirrored
  winding changes preserve their UV/corner correspondence on the temporary copy.
- Provide vertex normals and tangents. An n-gon-only Triangulate modifier on the
  temporary copy enables tangent calculation; its custom normals and UVs survive.
  Export uses Blender's MikkTSpace tangents rather than leaving reconstruction to
  another renderer.
- GLB normals remain OpenGL. Unreal's glTF importer explicitly enables green-channel
  conversion and Normal Map compression. Never pre-flip a GLB normal image.
- Multiply Base Color and AO in linear color at factor 1 before embedding the result.
  No separate occlusion contribution darkens the material a second time.
- Pack metallic/roughness as B/G, without inverting roughness. Pack Clear Coat and
  Coat Roughness as R/G. Unreal's native glTF material supports Clear Coat.
- Carry constant shader values when a channel has no baked map. Combine opacity in
  Base Color alpha; preserve straight RGB even where alpha is zero.
- Combine emission color and strength, normalize HDR radiance to a PNG texture, and
  retain brightness using `KHR_materials_emissive_strength`.
- Preserve 8/16-bit PNG choices for prepared channels. Source JPEGs can remain JPEG
  when no channel conversion is needed. Derived material textures use PNG to prevent
  further lossy encoding. EXR-derived channels use 16-bit PNG plus material factors
  where necessary, because standard glTF cannot embed EXR.
- Unreal 5.4 ignores glTF's colored specular inputs. Convert IOR, specular level and
  tint to scalar F0: `((IOR-1)/(IOR+1))² × 2 × level × luminance(tint)`. Choose an
  equivalent fixed IOR and a scalar factor/alpha texture. This preserves spatial
  reflectance and black tint; chromatic dielectric reflections become luminance.
- Export textures/materials with asset-specific names, preventing shared-folder
  collisions. The file contains all image and geometry buffers, without external URIs.
- Prepare the complete model folder privately and publish it with rollback on failure.
  Re-export preserves unrelated artist files and any Unity delivery in the same folder.

Lighting, exposure, tone mapping and BRDFs still differ between Blender and Unreal.
Arbitrary Blender node graphs and subsurface scattering do not have universal glTF
representations. Transmission follows the native importer and engine-version limits;
combined advanced shader effects are not guaranteed to match pixel for pixel.

References: [Epic's GLB support](https://dev.epicgames.com/documentation/en-us/unreal-engine/gltf-file-format-support-in-unreal-engine?application_version=5.4),
[Interchange import](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-assets-using-interchange-in-unreal-engine?application_version=5.4),
[Khronos glTF Validator](https://github.com/KhronosGroup/glTF-Validator).
