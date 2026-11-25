# Rebuilt user guide

**Retopo · UV · Bake** — Turn complex models into clean production assets.

Use high-poly models such as sculpts, detailed meshes or photogrammetry scans.

## Prepare a source

Save the Blender project and select a mesh. The Rebuilt sidebar immediately
shows Auto/Manual and the processing settings. Rebuild Asset or the first
Retopologise action registers the active mesh and starts processing. Selecting
a mesh alone does not register it or create working objects. With multiple
meshes selected, the active mesh is processed; its name appears in the summary.
Processing uses owned copies and preserves the source.
Choose Organic or Hard Surface; your explicit choice overrides detection.

Choose Quads or Triangles. **Strict** prioritizes the target count and aims to stay within 1% of the requested
finished quad count. **Dynamic** aims for the starting count or higher, up to 4x,
to preserve detail. Both retain the best usable mesh with an explicit warning
if bounded correction cannot reach the requested count. Surface recovery can
prioritize a healthier surface over a closer count. The 1% value is an acceptance
target, not a guaranteed output count. Read the achieved count and Report after
each build; increasing retries indefinitely would make processing unpredictable.

Native surfaces keep their connectivity and small components. Residual openings,
orientation problems and degenerate faces are reported for review instead of
starting another remesh automatically. Repeated unsupported spikes along a rim
can trigger one simplified AutoRemesher retry and, if needed, one emergency
QuadriFlow attempt. This adds at most a 90-second engine recovery allowance;
Blender finishing work has a separate cost. If recovery fails or makes the
surface worse, the best usable result is retained with a warning. QuadriFlow
also remains available when the native engine produces no usable surface or is
unavailable. An input with no usable surface may still fail.

Dynamic optional shape refinement has a 30-second estimated search allowance.
Count correction in either mode stops before another predicted pass would exceed
120 seconds; Strict uses at most four native passes. Running passes finish normally,
so a single expensive engine call can exceed these allowances. Reports retain the
actual count and shape errors. This is not a two-minute full Auto guarantee.

During processing, the viewport header shows the current stage, remesh pass,
total elapsed time and stage elapsed time. The percentage is labeled **steps**;
it represents stage completion, not an estimate of remaining time.

## Auto

Choose a count and texture resolution, then press **Rebuild Asset**. The add-on
prepares the source, remeshes, creates seams/UVs, bakes, checks quality and
completes the asset. Export is optional.

Escape cancels the current Auto attempt. Cancellation or a failure before
commitment restores the prior completed result; a first build removes its partial
work and restores overwritten textures. Native
Blender operations may block the interface until they return.
Your source settings remain available for the next attempt. After completion or
cancellation cleanup, the next click works without an extra click to dismiss the job.

Closing the window running the job or disabling Rebuilt also requests cancellation.
Cleanup continues after the worker stops, even after its progress display closes.
Blocking Blender operations, including Cycles baking, must return before pending
cancellation can be handled. Keep Blender running until cleanup finishes.

Deleting the source during processing stops the job. Auto restores the surviving
previous result and its files, but does not recreate the deleted source or apply
the old job to a replacement object. Renaming the source is supported.

If a saved `.blend` contains an interrupted build or Manual stage, select its source or
generated result and choose **Recover Interrupted Build** in Rebuilt. Review the
previous objects to restore, unfinished results to remove, and texture paths.
For Auto, confirming restores the previous build or discards an unfinished first
build. For Manual, it restores that stage's previous result and texture files,
preserving earlier completed steps. If the result already committed but cleanup
failed, the dialog instead offers to finish cleanup while keeping the completed
scene and textures. Check which action the dialog describes before confirming.
Recovery also changes files on disk and is not a Blender Undo action. Save the
recovered `.blend` after checking the result.

Recovery checks object ownership, saved settings, texture contents and backup
availability before changing anything. A later texture edit, missing backup,
occupied original name, active job or conflicting scene copy stops recovery and
keeps its recovery data. Follow the reported conflict instead of deleting the
backup. Texture backups live in Blender's user configuration under
`rebuilt/recovery`; moving only the `.blend` to another computer does not move
these backups. Older interrupted builds without a texture journal cannot restore
files they already changed. Avoid editing the same texture files in another
application while Rebuilt writes or restores them.

An older pending scene cannot undo texture files from a later completed build.
If the commit outcome cannot be confirmed, Rebuilt retains both versions and
blocks further processing of that asset until the recovery conflict is resolved.

For older assets that predate ownership tracking, **Review Existing Results**
lists the objects Rebuilt would manage. Accept only when these are the generated
results you intend future builds to replace. Artist copies need their own asset
through **Use Selected Model**.

## Manual

1. **Retopologise** prepares the source and builds the low mesh. Every successful
   run enables Wireframe for review.
2. **Seams** generates editable cuts. Mark or clear red edges in Edit Mode before
   continuing. **Recalculate Seams** replaces the cuts unless **Keep Marked Seams** is enabled. **Other** appears only when
   genuine stored alternatives exist.
3. **Unwrap** unfolds and packs the islands. Expand its section to access
   **Show UVs**, which opens or reuses a UV Editor; the optional UV Grid preview
   helps review distortion.
4. **Bake Maps** transfers maps, checks quality and completes the asset without
   requiring a separate finish action.

An error while applying Bake or finishing restores that stage's previous mesh,
material, reports and texture files. Earlier successful Manual steps remain.
Required maps must all be present and encode successfully; a partial texture set
does not count as completion. Unsaved projects can still keep their maps in memory.
Replacing the UV Grid or baked result preserves materials and images still used
by your independent copies. Rebuilt creates fresh bake resources instead of
replacing data merely because it has the same name.

All four stage headers use the same disclosure control. Successful Manual
processing collapses the settings; expand a stage to review options or redo it.
Seam alternatives and existing-layout controls stay inside their stage section.

Expand **Seams → UV Islands** to choose **Balanced** or **Fewer**. Balanced keeps
the standard layout. Fewer additionally tries joining larger adjacent generated
islands while preserving their internal shape. Select **Recalculate Seams**, then
**Unwrap**, to apply it. A successful seam reduction shows the before/after island
count; otherwise the previous generated proposal is retained. This is a preference,
not a fixed island count. It may add processing time, and required UV repairs can
still add islands; the UV report records those additions. Auto always uses Balanced.
Hand-edited seams and accepted external UVs are not automatically consolidated.

Enable **Seams → Keep Marked Seams** to retain chosen cuts while generating the
remaining seams. On the retopology mesh, select edges in Edit Mode and use
**Edge → Mark Seam**, then run **Recalculate Seams** and **Unwrap**. The option
protects the marked cuts with both Balanced and Fewer, including during UV repair
and packing. It retains every currently marked seam, including cuts from earlier
automatic runs: clear unwanted marks before recalculating. It does not preserve
UV coordinates or transfer seams from the high-poly source through retopology.
The option is off by default and does not affect Auto. Re-bake existing maps after
changing the UV layout.

UV pins on the working low mesh preserve their coordinates during **Unwrap**.
Density normalization and packing hold every island containing a pin in place,
with either packer. A requested layout density includes those fixed islands;
the other islands supply the remaining texture area. If the fixed islands violate
the tile border or spacing, or the requested density cannot fit, Unwrap reports
the failure and restores the input mesh and UVs. It does not move pins or lower
the requested density to force a fit. Manual unwrapping can still reshape an
island's unpinned corners; use **Use Existing UVs** to retain the complete map.

**Finish Here** is available at intermediate stages. It keeps the result and
removes owned preparation. You may resume later; unchanged upstream settings
allow preparation to be restored without replacing the low mesh.

### Use your own seams or UVs

Edit the generated low mesh, keeping it in the same Rebuilt asset.

- Expand **Seams**, then select **Use Existing Seams**
  to accept your cuts and continue with our Unwrap.
- Expand **Unwrap** and select **Use Existing UVs** at the bottom to accept the
  active map and continue directly to Bake Maps. This can bypass both our seams
  and unwrapping.

These secondary controls are hidden by default. Acceptance does not move UVs or
repack them. The active accepted UV layer becomes the render layer; other layers
remain available. Empty, non-finite, fully collapsed or out-of-tile layouts are
rejected. Partial collapses and overlaps warn without automatic repair. Baking
currently supports one 0–1 tile, not UDIM layouts.

Accepting changed work invalidates downstream maps. Texture or packer setting
changes do not regenerate accepted external layouts. Explicitly running Seams or
Unwrap again, or rebuilding retopology, can replace them. Acceptance supports
undo and redo. Changes to seam marks or pinned UV corners make generated Unwrap
results stale. Editing unpinned UV coordinates makes baked maps stale without
automatically replacing the edited layout. Changing the selected UV layer also refreshes these
dependencies; accepted external layouts continue to bypass generation. These
checks still apply after saving and reopening the project.

## Materials and maps

New bakes are saved beside the project under `textures/<asset name>/`, keeping
each model's maps together. Existing image files are not moved or deleted;
rebaking writes the new folder and updates the material's image paths. Work in
progress lives in `<asset name>_rebuilt_work` in the Outliner. Default collection
names from older versions are updated without replacing their contents or tags.

Normal, Base Color and AO form the initial set. Source analysis suggests other
relevant channels; explicit checkbox selections take precedence. Common unlinked
Principled values transfer directly without unnecessary images.

| Basic maps | Additional Material Channels |
|---|---|
| Normal, Base Color, AO | Specular, Specular Tint, IOR |
| Roughness, Metallic, Opacity | Emission, Emission Strength, Transmission |
| | Coat, Coat Roughness, Subsurface, Subsurface Scale |

Base Color and AO connect through **Mix Color > Multiply** at factor **1**.
They remain separate images for export. Common Emission scan materials convert
on working copies; see the [material protocol](material-import-protocol.md).

### Texture Output

Available in Auto and Manual. Settings apply to new maps; changing the format
does not convert existing images automatically.

| Format | Options |
|---|---|
| PNG | 8/16-bit, lossless compression level |
| TIFF | 8/16-bit, None/Deflate/LZW compression |
| OpenEXR | 16/32-bit float, ZIP/PIZ/None compression |
| JPEG (.jpg) | RGB 8-bit, Quality 1–100; lower values reduce size with visible losses |

JPEG applies to Base Color, Specular Tint and non-HDR Emission. Data maps such as
Normal and Opacity stay lossless PNG16. HDR values automatically use OpenEXR.
JPEG quality defaults to 85; the default format remains PNG.

### Bake memory limit

Rebuilt estimates memory before starting a bake. In **Preferences → Add-ons →
Rebuilt**, **Bake memory limit (GiB)** defaults to **0**, which uses 80% of the
installed system RAM reported by the operating system. Enter a positive value
to choose an explicit limit. If capacity detection is unavailable, set an
explicit limit before baking.

The estimate includes the requested maps, surface/ray buffers, recovery work,
mesh data and host overhead. If it exceeds the limit, Rebuilt refuses the bake
before copying mesh arrays or allocating image-sized buffers. An existing
Manual delivery remains intact. Choose fewer maps or a lower resolution, or
raise the preference only when the machine has enough capacity. Rebuilt does
not lower your requested settings automatically.

This is an estimate check, not a hard cap on Blender's allocations. Installed
RAM is not currently available RAM; other applications, process/container
limits, procedural source materials and GPU/driver allocations can still cause
memory exhaustion. Dedicated GPU memory is not measured by this setting.

## Review and export

Compare the result visually and expand **Report** for mesh, UV, projection and
quality measurements. Completion does not mean every quality target passed.

Renaming baked images or annotating their texture nodes preserves export links.
A delivered copy keeps its own texture references when you rebuild the original.
If you deliberately relink a generated texture node, export uses the replacement
without changing that image's pixels or path. Images shared between map roles get
separate export files and import settings. Missing or disconnected required maps
must be restored before export.

Output filenames escape characters that Windows cannot safely store, including
reserved names such as `CON`. For example, `Wheel?front` uses `Wheel~3Ffront` in
its output filenames. The original Blender object name stays unchanged. Escaping
also distinguishes names that would otherwise collapse to the same filename.
Very long output names use a shortened stem with a stable hash suffix, leaving
room for texture and metadata suffixes. The model's name in Blender is preserved.
Names using a non-NFC Unicode spelling also receive a stable hash suffix, so
visually identical names with different character encodings keep separate files.
Re-exporting a non-NFC asset uses its new path and leaves its legacy outputs
untouched. Re-exporting to the new paths preserves their Unity metadata.
Ordinary NFC names keep their existing output paths.
On case-insensitive volumes, output stops before overwriting an existing path
whose spelling differs only by capitalization (for example, `Rock` and `rock`).
Choose a different project/export folder, or give the source a distinct name
before processing it. Re-exporting the same asset to its own files still works.

Choose Unity, Unreal or Godot, then **Export** and select a destination folder.
Each target creates a folder named after the model. Unity includes the FBX, native
material, required textures and `.meta` files; Unreal and Godot include a self-contained GLB.
Format is displayed as fixed text because each target has one supported format. For other destinations,
use Blender’s standard File > Export menu.
For Unity, choose **URP**, **Built-in** or
**HDRP** to match the destination project. Copy the complete model folder, including `.meta` files, into Unity Assets:
the native material, texture settings and map connections are included. No manual
material wiring is required. See [Unity export](unity-export.md) for delivery,
reimport instructions and native shader limitations.
Generic textures beside the project remain attached to the Blender material.
For Unreal, drag the GLB into the Content Browser and click **Import**. Its native
importer assigns the material and textures; no Python setup or manual wiring is
needed. GLB keeps OpenGL normals and Unreal converts them on import. See
[Unreal export](unreal-export.md). For Godot, copy the GLB into the project folder
and drag the imported scene from the FileSystem dock into your scene. Materials
and textures are embedded. Godot currently shares Unreal's GLB preparation; the
import guide in the output folder names the selected engine. Blender preview
textures remain unchanged.

Runtime depends on geometry, correction passes, texture size, selected maps and
hardware. There is no fixed two-minute completion guarantee. In the documented
0.30.39 Mac validation, the 20 full Auto runs took 64–276 seconds at 4096px.
These are measured examples on an M1 Ultra, not estimates for every computer or
input.

The asset summary shows the mesh name, build status, readable asset type, geometry
count, budget mode and texture dimensions. Completed retopology displays measured
counts; changed settings appear separately under **Next Build**. The progress bar
shows how many processing stages are up to date.

## Help

The Help panel is available even before choosing a source. Discord Channel and
Contact open the studio’s support destinations. Youtube Tutorial opens the
Rebuilt video guide; Documentation opens the online Rebuilt manual.
Auto groups geometry controls under Model Settings and resolution, format and
compression under Texture Output. Both sections expand independently. Manual
actions use separate stage cards; Export and Report are independent panels.

Rebuild Asset and Retopologise frame the model in the invoking Object-mode
viewport when starting work, without changing the viewing angle. Camera views
remain untouched, including when Lock Camera to View is enabled. Quad views and
other editors keep their existing framing. Selection, Seams, Unwrap and Bake Maps
do not trigger framing. The legacy adoption operator remains available for
script compatibility but is no longer part of the sidebar workflow.

## Flow Guides status

Flow Guides remains suspended. Its panel and recipe dispatch are disabled;
old saved guide data is retained for investigation. Use ordinary AutoRemesher
retopology. 

## Current installation and quality behavior

Use Blender 5.2.x and the matching extension ZIP from Preferences → Get Extensions
→ Install from Disk. Before migrating from the classic addon, save the scene,
note any custom Bake memory limit, disable/remove the classic copy and restart
Blender. Re-enter that preference if necessary; saved scene settings are retained.
Do not enable two copies. See [extension installation and updates](extensions.md).
Export stays visible and explains when the model needs processing before export.
Workflow controls are non-animatable.

Cloth Folds is enabled in Auto and can be disabled in Manual. The native AutoRemesher
path preserves raw vertex positions and unique face cycles, including intentional
open edges. One-face UV islands are reduced where possible; remaining singletons
alone do not block an otherwise valid pipeline.
