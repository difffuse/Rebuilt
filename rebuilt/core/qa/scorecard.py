"""Turning every measurement into a verdict an artist can act on."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        return {"pass": 0, "unknown": 1, "warn": 2, "fail": 3}[self.value]

    @classmethod
    def worst(cls, verdicts: Sequence["Verdict"]) -> "Verdict":
        return max(verdicts, key=lambda v: v.rank) if verdicts else cls.UNKNOWN


@dataclass(frozen=True)
class Check:
    """One measurement against one threshold."""

    name: str
    verdict: Verdict
    value: float | None = None
    threshold: float | None = None

    higher_is_better: bool = False
    detail: str = ""

    @property
    def score(self) -> float:
        """0-100 for this check, from how close the value is to its threshold."""
        if self.verdict is Verdict.UNKNOWN or self.value is None or self.threshold is None:
            return 0.0 if self.verdict is Verdict.FAIL else 50.0

        if self.threshold == 0:
            return 100.0 if self.value == 0 else 0.0

        ratio = (
            self.value / self.threshold
            if not self.higher_is_better
            else (self.threshold / self.value if self.value else 0.0)
        )
        if ratio <= 0:
            return 100.0

        return float(max(0.0, min(100.0, 100.0 - 40.0 * min(ratio, 2.5))))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        payload["score"] = self.score
        return payload


@dataclass(frozen=True)
class Section:
    """One of the scorecard's headings."""

    name: str
    checks: tuple[Check, ...] = ()

    reason: str = ""

    @property
    def verdict(self) -> Verdict:
        return Verdict.worst([c.verdict for c in self.checks])

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return float(sum(c.score for c in self.checks) / len(self.checks))

    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.verdict is Verdict.FAIL]

    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.verdict is Verdict.WARN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "score": self.score,
            "reason": self.reason,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass(frozen=True)
class Scorecard:
    """The whole verdict."""

    sections: tuple[Section, ...] = ()
    warnings_as_errors: bool = False

    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        worst = Verdict.worst([s.verdict for s in self.sections])
        if self.warnings_as_errors and worst is Verdict.WARN:
            return Verdict.FAIL
        return worst

    @property
    def score(self) -> float:
        judged = [s for s in self.sections if s.checks]
        if not judged:
            return 0.0
        return float(sum(s.score for s in judged) / len(judged))

    @property
    def is_ready(self) -> bool:
        return self.verdict is Verdict.PASS

    def issues(self) -> list[Check]:
        """Everything that needs a human, worst first."""
        found = [c for s in self.sections for c in s.checks if c.verdict is not Verdict.PASS]
        return sorted(found, key=lambda c: (-c.verdict.rank, c.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "score": self.score,
            "ready": self.is_ready,
            "missing": list(self.missing),
            "sections": [s.to_dict() for s in self.sections],
        }


def _limit(
    name: str,
    value: float | None,
    threshold: float,
    warn_at: float | None = None,
    detail: str = "",
) -> Check:
    """A check where smaller is better."""
    if value is None:
        return Check(name, Verdict.UNKNOWN, detail=detail or "not measured")
    if value > threshold:
        verdict = Verdict.FAIL
    elif warn_at is not None and value > warn_at:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS
    return Check(name, verdict, float(value), float(threshold), False, detail)


def _minimum(
    name: str,
    value: float | None,
    threshold: float,
    warn_at: float | None = None,
    detail: str = "",
) -> Check:
    """A check where larger is better."""
    if value is None:
        return Check(name, Verdict.UNKNOWN, detail=detail or "not measured")
    if value < threshold:
        verdict = Verdict.FAIL
    elif warn_at is not None and value < warn_at:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS
    return Check(name, verdict, float(value), float(threshold), True, detail)


def _count(name: str, found: int, allowed: int = 0, detail: str = "") -> Check:
    """A check on a count of things that should not exist."""
    verdict = Verdict.PASS if found <= allowed else Verdict.FAIL
    return Check(name, verdict, float(found), float(allowed), False, detail)


def shape_fidelity(retopo: dict[str, Any] | None, limits: dict[str, float]) -> Section:
    """Does the low mesh still have the scan's shape?"""
    if not retopo:
        return Section("Shape Fidelity", reason="retopology has not run")

    deviation = retopo.get("deviation", {})
    p95 = deviation.get("p95")
    detail = "distance from the low mesh to the scan, as a fraction of its size"
    if p95 is not None and p95 > limits["max_deviation_p95"]:
        asked = int(retopo.get("target_faces", 0))
        detail = (
            f"too few quads for this scan — raise the count above {asked:,}"
            if asked
            else "too few quads for this scan — raise the count"
        )
    checks = [
        _limit(
            "Deviation P95",
            p95,
            limits["max_deviation_p95"],
            warn_at=limits["max_deviation_p95"] * 0.7,
            detail=detail,
        ),
        _limit(
            "Worst deviation",
            deviation.get("maximum"),
            limits["max_deviation_p95"] * 10.0,
            warn_at=limits["max_deviation_p95"] * 4.0,
            detail="a single collapsed feature shows here and nowhere else",
        ),
        _count(
            "Unresolved samples",
            int(deviation.get("unresolved", 0)),
            detail="points the surface query could not place at all",
        ),
    ]
    return Section("Shape Fidelity", tuple(checks))


def topology_health(retopo: dict[str, Any] | None, limits: dict[str, float]) -> Section:
    """Is the mesh structurally sound?"""
    if not retopo:
        return Section("Topology Health", reason="retopology has not run")

    topology = retopo.get("topology", {})
    faces = max(int(topology.get("face_count", 0)), 1)
    checks = [
        _count("Non-manifold edges", int(topology.get("non_manifold_edges", 0))),
        _count("Flipped faces", int(topology.get("flipped_faces", 0))),
        _count("Degenerate faces", int(topology.get("degenerate_faces", 0))),
        _limit(
            "Sliver faces",
            int(topology.get("skinny_faces", 0)) / faces,
            limits["max_sliver_ratio"],
            warn_at=limits["max_sliver_ratio"] * 0.5,
            detail="faces far thinner than they are long, as a fraction of the mesh",
        ),
    ]
    return Section("Topology Health", tuple(checks))


def uv_quality(uv: dict[str, Any] | None, limits: dict[str, float]) -> Section:
    """Is the layout something a texture can live on?"""
    if not uv:
        return Section("UV Quality", reason="unwrapping has not run")

    stretch = uv.get("stretch", {})
    overlaps = uv.get("overlaps", {})
    checks = [
        _limit(
            "Stretch P95",
            stretch.get("p95"),
            limits["max_uv_stretch_p95"],
            warn_at=limits["max_uv_stretch_p95"] * 0.7,
            detail="how much more the texture is stretched one way than the other",
        ),
        _limit(
            "Overlapping faces",
            _overlap_share(overlaps),
            10**9 if limits.get("allow_uv_overlap") else limits["max_uv_overlap_area"],
            warn_at=None
            if limits.get("allow_uv_overlap")
            else limits["max_uv_overlap_area"] * 0.25,
            detail=(
                f"{int(overlaps.get('overlapping_pairs', 0)):,} pairs of faces painted "
                "with the same texels, as a share of the map"
            ),
        ),
        _limit(
            "Flipped UV faces",
            _flipped_share(uv, overlaps),
            limits["max_flipped_uv_faces"],
            warn_at=limits["max_flipped_uv_faces"] * 0.25,
            detail=(
                f"{int(overlaps.get('flipped_triangles', 0)):,} triangles turned over "
                "in the layout, as a share of the mesh"
            ),
        ),
        _minimum(
            "Tile occupancy",
            uv.get("occupancy"),
            limits["min_uv_occupancy"],
            detail="how much of the texture the islands actually use",
        ),
    ]
    if not uv.get("fits_in_tile", True):
        checks.append(
            Check(
                "Islands inside the tile",
                Verdict.FAIL,
                detail="the layout overflows 0-1 at the texel density asked for",
            )
        )
    return Section("UV Quality", tuple(checks))


def bake_quality(bake: dict[str, Any] | None, limits: dict[str, float]) -> Section:
    """Did the detail land where it should have?"""
    if not bake:
        return Section("Bake Quality", reason="baking has not run")

    quality = bake.get("quality", {})
    covered = max(int(quality.get("covered_texels", 0)), 1)
    checks = [
        _minimum(
            "Valid texels",
            quality.get("valid_fraction"),
            limits["min_valid_bake_pixels"],
            detail="texels written from the surface they belong to",
        ),
        _limit(
            "Strays",
            int(quality.get("stray_texels", 0)) / covered,
            limits["max_stray_ratio"],
            detail="texels baked from unrelated geometry, which look plausible and are not",
        ),
        _count(
            "Maps not produced",
            sum(1 for m in bake.get("maps", []) if not m.get("baked")),
        ),
    ]
    approximated = int(bake.get("approximation", {}).get("texels", 0))
    if approximated:
        checks.append(Check(
            "Approximate texels", Verdict.WARN, float(approximated), 0.0, False,
            "filled from nearby texels in the same UV island; unavailable regions use neutral values",
        ))
    return Section("Bake Quality", tuple(checks))


def production_readiness(
    retopo: dict[str, Any] | None,
    uv: dict[str, Any] | None,
    bake: dict[str, Any] | None,
    limits: dict[str, float],
) -> Section:
    """Would this asset drop into a project without an argument?"""
    checks: list[Check] = []

    if retopo and retopo.get("budget_mode") == "strict":
        checks.append(
            _limit(
                "Strict quad count",
                abs(retopo["quad_budget_error"]),
                0.01,
                detail="final quads must be within 1% of the requested count",
            )
        )
    elif retopo and retopo.get("budget_mode") == "dynamic":
        checks.append(
            _limit(
                "Dynamic quad budget",
                max(
                    retopo["achieved_quads"] / max(retopo["quad_budget_limit"], 1),
                    retopo.get("requested_quads", retopo.get("target_faces", 0))
                    / max(retopo["achieved_quads"], 1),
                ),
                1.0,
                detail="final quads must reach the starting budget and stay within the dynamic limit",
            )
        )
    elif retopo:
        checks.append(
            _limit(
                "Polygon budget",
                max(retopo.get("budget_used", 0.0), 0.0),
                1.0 + limits["budget_tolerance"],
                detail="under budget is fine; over it is not",
            )
        )
    if uv:
        target = uv.get("texel_density_target", 0.0)
        achieved = uv.get("texel_density", 0.0)

        if target > 0 and uv.get("preserve_texel_density"):
            checks.append(
                _limit(
                    "Texel density",
                    abs(achieved - target) / target,
                    limits["texel_density_tolerance"],
                    detail=f"{achieved:.0f} px/unit against {target:.0f} asked for",
                )
            )
    if bake:
        checks.append(
            _count(
                "Maps missing",
                sum(1 for m in bake.get("maps", []) if not m.get("baked")),
                detail="a map the profile asked for that did not come out",
            )
        )

    if not checks:
        return Section("Production Readiness", reason="nothing to judge yet")
    return Section("Production Readiness", tuple(checks))


def _overlap_share(overlaps: dict[str, Any]) -> float | None:
    """Fraction of the tile painted twice; a pair count alone stands in for it as all-or-nothing when the area was never measured."""
    if "overlapping_area" in overlaps:
        return float(overlaps["overlapping_area"])
    if "overlapping_pairs" in overlaps:
        return 1.0 if int(overlaps["overlapping_pairs"]) > 0 else 0.0
    return None


def _flipped_share(uv: dict[str, Any], overlaps: dict[str, Any]) -> float | None:
    flipped = overlaps.get("flipped_triangles", 0)
    faces = int(uv.get("faces", 0))

    return float(flipped) / (2 * faces) if faces else (1.0 if int(flipped) > 0 else 0.0)


DEFAULT_LIMITS: dict[str, float] = {
    "max_deviation_p95": 0.002,
    "max_sliver_ratio": 0.02,
    "max_uv_stretch_p95": 0.15,
    "max_uv_overlap_area": 0.02,
    "max_flipped_uv_faces": 0.01,
    "min_uv_occupancy": 0.35,
    "min_valid_bake_pixels": 0.995,
    "max_stray_ratio": 0.005,
    "budget_tolerance": 0.05,
    "texel_density_tolerance": 0.05,
    "allow_uv_overlap": 0.0,
}


def build(
    reports: dict[str, dict[str, Any] | None],
    limits: dict[str, float] | None = None,
    warnings_as_errors: bool = False,
) -> Scorecard:
    """Assemble the scorecard from whatever the pipeline has produced so far."""
    thresholds = {**DEFAULT_LIMITS, **(limits or {})}

    retopo = reports.get("retopo")
    uv = reports.get("uv")
    bake = reports.get("bake")

    sections = (
        shape_fidelity(retopo, thresholds),
        topology_health(retopo, thresholds),
        uv_quality(uv, thresholds),
        bake_quality(bake, thresholds),
        production_readiness(retopo, uv, bake, thresholds),
    )

    return Scorecard(
        sections=sections,
        warnings_as_errors=warnings_as_errors,
        missing=tuple(s.name for s in sections if not s.checks),
    )
