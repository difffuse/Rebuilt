# AutoRemesher Fine Details: source and licensing

AutoRemesher Fine Details is an independently maintained dependency. This addon
consumes its headless engine; the engine's maintainer owns upstream releases.
The bundled runtime manifest records the exact source revision and archive hash.
Corresponding source must accompany binary distribution through the release's
source delivery mechanism.

The headless driver and local-charge optimization additions originate from
Rebuilt and retain their GPL-3.0-or-later notices. The original MIT notice in
modified engine source is preserved; unchanged engine sources retain their MIT
license. The root MIT statement does not relicense those GPL additions or imply
that the modified core is wholly MIT. The frozen equivalence-test oracle comes
from the original MIT source at commit
`fef338d27aeb9c5f0eca12b305538ed670c59776`; its source hash is recorded in the
corresponding source's `tests/reference/singularity_manifest.json`.

The bundled `LICENSE` and `licenses/GPL-3.0-or-later.txt` retain the applicable
license texts. Third-party components retain their own notices under `licenses/`,
including Eigen, isotropicremesher, meshoptimizer, TBB and tinyobjloader.

This notice describes the included source and licenses. It does not change
ownership, grant additional publication rights, or replace the component notices.
