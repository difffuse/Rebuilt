# Scan material import protocol

Material conversion runs automatically during
ingest in both Auto and Manual. No preparatory editing of the scan is required.

1. Inspect the shader connected to the active Material Output. Disconnected
   Principled nodes do not override the visible shader when extracting color.
2. Preserve the original scan and its materials. Convert only a material copy
   assigned to the pipeline's working mesh, once per shared material.
3. Keep an existing reachable Principled BSDF. Convert a direct Emission shader,
   or Emission mixed directly with Transparent, to Principled BSDF. Connect the
   original color network to Base Color, with Metallic 0, Roughness 0.6 and
   Specular IOR Level 0. An unlit scan provides no authored reflection response;
   do not add Principled's default specular layer during conversion. Existing
   Principled materials keep their authored specular values and links.
4. Preserve image nodes, UV/mapping links, image color spaces and authored
   Emission Strength. A non-unit or linked strength multiplies the transferred
   color. Do not apply an extra gamma correction or AO multiplication.
5. Remove the common `Is Camera Ray` preview gate. For an ordinary transparency
   mix, transfer its factor to Principled Alpha, respecting branch polarity.
   This preserves the working material's mask; it does not add an alpha bake
   or extend the existing output/export channel set.
6. Preserve unsupported complex shader mixes and report the reason in preflight
   notes. Do not guess a replacement color. Missing materials, textures or UVs
   still require their existing preflight checks.
7. Bake Base Color from the reachable color input through emission substitution,
   avoiding scene lighting in the color map. Assemble the finished Principled
   material. Since v0.30.6, the artist explicitly requests Base Color × AO
   through Mix Color / Multiply with factor 1. Each generated AO is connected;
   without AO, Base Color is connected directly. Image pixels are not multiplied
   or overwritten, and export still receives the separate maps. AO's bake radius
   is 2% of the source diagonal, set on a temporary world that is always removed
   and restored to the artist's original world after the pass.
8. Record each material's disposition in `ingest.preflight.material_protocol`.
   Converted materials also show **Material: Emission → Principled BSDF** in
   Scan Report. The protocol revision invalidates old ingest/downstream caches.

The protocol preserves texture color and coordinates. Principled responds to
scene lighting, so it cannot match an unlit preview under every light setup.
Verify source and result under matched viewing conditions, alongside the bake's
valid-texel measurement: a shader conversion cannot compensate for a collapsed
mesh or missed projection rays.

The 10_HEAD comparison found no appreciable global Base Color drift: 19,727
near-surface samples had signed mean RGB differences of +0.30, -0.01, +0.15 on
an 8-bit scale, with mean absolute differences of 2.32, 2.14, 2.43. This is an
approximate nearest-surface comparison, not identical bake-ray sampling. The
output uses JPEG quality 60 and resampled UVs; these measurements do not isolate
compression error. AO averaged 0.9867 and its median was 1. The visible washed-out
shirt was reduced by removing the automatically added specular response under
the live forest HDRI. Diffuse lighting, normals and the requested AO still differ
from the source's camera-gated Emission shader. Do not compensate texture pixels
or scene exposure to match one HDRI. Evidence: `dist/fixes/head-color-0.30.37/`.

Source inspection and conversion live in `bridge/materials.py`; generated
presentation assembly lives in `bridge/output_materials.py`. The bake driver
shares source inspection and retains its isolated channel substitution.

Regression coverage: `tests/integration/test_material_protocol.py` exercises
camera-gated emission, source preservation and repeated normalization,
disconnected Principled nodes, constant/linked intensity, both transparency
branch orders and preservation/reporting of unsupported mixed-emission graphs.


## PBR appearance transfer — v0.30.28

Output carries shared, unlinked Principled scalar/color constants (including
IOR and specular controls), instead of silently resetting everything not baked.
Linked and per-material-varying advanced inputs are suggested as bake maps.
The supported additions are Specular IOR Level, Specular Tint, IOR, Emission
Color/Strength, Transmission Weight, Coat Weight/Roughness, Subsurface
Weight/Scale. They use isolated emission substitution and connect to the
corresponding Principled inputs. Direct emission scan conversion remains the
default import behavior; explicitly selected Emission maps can read the retained
original emitter. Unsupported arbitrary mixed shader trees are not flattened
into a claimed equivalent Principled material.

PNG/TIFF permit 8/16-bit, EXR 16/32-bit float; selected codecs are lossless. HDR
channels outside the normalized range use EXR even under PNG/TIFF output to
avoid clipping; the UI states this. Encoding is attached to each map report and
used by preview, finalization and export. Texture writes use a disposable scene
with neutral color management; user render exposure/looks remain untouched.
Bump/Displacement and alternate Glossiness extraction were not added: the current
normal bake and Roughness channel retain their existing meanings.

## JPEG output — v0.30.29

JPEG (.jpg) is available for Base Color, Specular Tint and non-HDR Emission,
with Quality 1–100 (default 85). Data maps remain PNG16 and HDR still uses EXR.
Opacity remains a separate lossless map because JPEG does not store alpha.
See the [user guide](user-guide.md) for all format controls.
