"""Budget policy for finished quad meshes, independent of the remeshing engine."""

STRICT_TOLERANCE = 0.01
STRICT_PASSES = 8
DYNAMIC_PASSES = 4
DYNAMIC_MULTIPLIER = 4
DYNAMIC_REFINEMENT_SECONDS = 30.0


def budget_status(mode: str, target: int, quads: int) -> dict:
    error = (quads - target) / max(target, 1)
    limit = target * DYNAMIC_MULTIPLIER if mode == "dynamic" else target
    met = quads > 0 and (
        target <= quads <= limit
        if mode == "dynamic"
        else abs(quads - target) <= target * STRICT_TOLERANCE
    )
    return {
        "budget_mode": mode,
        "requested_quads": target,
        "achieved_quads": quads,
        "quad_budget_error": error,
        "quad_budget_limit": limit,
        "quad_budget_minimum": target if mode == "dynamic" else target * (1 - STRICT_TOLERANCE),
        "budget_met": met,
    }


def dynamic_improves(candidate: dict, previous: dict) -> bool:
    """Buy measurable shape accuracy without degrading quad purity or folds."""
    if candidate["shape_error"] >= previous["shape_error"] * 0.98:
        return False
    if candidate["topology"]["quad_ratio"] + 1e-9 < previous["topology"]["quad_ratio"]:
        return False
    if candidate["fold_fraction"] > previous["fold_fraction"] + 1e-9:
        return False
    return all(
        candidate["topology"][key] / max(candidate["achieved_faces"], 1)
        <= previous["topology"][key] / max(previous["achieved_faces"], 1) + 1e-9
        for key in ("non_manifold_edges", "degenerate_faces", "flipped_faces")
    )


def prefer_candidate(
    candidate: dict, previous: dict, mode: str, target: int, *, recover_surface: bool = True
) -> bool:
    """Keep a healthier surface before comparing count or optional shape gains."""

    def quality(report):
        finding = report.get("surface_quality", {})
        return bool(finding.get("needs_recovery")), float(finding.get("severity", 0))

    current_quality, previous_quality = quality(candidate), quality(previous)
    if recover_surface and current_quality != previous_quality:
        return current_quality < previous_quality
    if mode == "strict":
        return abs(candidate["achieved_quads"] - target) < abs(previous["achieved_quads"] - target)
    if candidate["budget_met"]:
        return not previous["budget_met"] or dynamic_improves(candidate, previous)
    if previous["budget_met"]:
        return False

    def distance(count):
        return max(target - count, count - target * DYNAMIC_MULTIPLIER, 0)

    return distance(candidate["achieved_quads"]) < distance(previous["achieved_quads"])
