"""Primitives for OpenPose-style flat keypoint arrays.

A "flat array" here is the OpenPose JSON layout: ``[x0, y0, c0, x1, y1, c1, ...]``.
A point is considered *present* when its confidence clears a caller-supplied gate and
it is not the sentinel ``(0, 0)`` that OpenPose emits for missing detections.

Every helper takes its thresholds as explicit arguments, so this module has no
dependency on :mod:`ts_utils.config` and only needs the standard library.  That makes
it the cheapest module in the package to unit test.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

__all__ = [
    "COCO18_EDGES",
    "COCO18_NEIGHBORS",
    "HAND21_EDGES",
    "_is_valid_xyc",
    "_reshape_keypoints_2d",
    "_flatten_keypoints_2d",
    "_sum_conf",
    "_body_center_from_pose",
    "_dist",
    "_estimate_torso_scale",
    "_count_valid_points",
    "_zero_out_kps",
    "_body_head_root_scale_from_pose",
    "_body_wrist_root_scale_from_pose",
]

Point = Tuple[float, float]
Keypoint = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Skeleton topology used by the *filters*.
#
# NOTE: these tables are intentionally NOT the ones used by ts_utils.render.
# The render tables encode colour by edge index and omit the (8, 11) hip bar;
# aliasing the two would silently recolour the skeleton and draw an extra limb.
# ---------------------------------------------------------------------------

COCO18_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (8, 11),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

HAND21_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def _derive_neighbors(edges: Tuple[Tuple[int, int], ...]) -> Mapping[int, FrozenSet[int]]:
    """Build an immutable adjacency map from an undirected edge list."""
    acc: Dict[int, set] = {}
    for a, b in edges:
        acc.setdefault(a, set()).add(b)
        acc.setdefault(b, set()).add(a)
    return MappingProxyType({j: frozenset(ns) for j, ns in acc.items()})


COCO18_NEIGHBORS: Mapping[int, FrozenSet[int]] = _derive_neighbors(COCO18_EDGES)
"""Adjacency map of :data:`COCO18_EDGES`, derived eagerly and read-only.

This replaces the old lazily-populated ``_NEIGHBORS`` module cache: module-level
mutable state breaks as soon as the package is imported twice under two names
(``ts_utils`` in tests, ``comfyui-teskors-utils.ts_utils`` inside ComfyUI).
"""


# ---------------------------------------------------------------------------
# Flat-array plumbing
# ---------------------------------------------------------------------------


def _is_valid_xyc(x: float, y: float, c: float) -> bool:
    """Return ``True`` when a raw ``(x, y, confidence)`` triple is a real detection."""
    if c is None or c <= 0:
        return False
    if x == 0 and y == 0:
        return False
    if math.isnan(x) or math.isnan(y) or math.isnan(c):
        return False
    return True


def _reshape_keypoints_2d(arr: Optional[List[float]]) -> List[Keypoint]:
    """Group a flat ``[x, y, c, ...]`` array into a list of ``(x, y, c)`` tuples."""
    if arr is None:
        return []
    out: List[Keypoint] = []
    for i in range(0, len(arr), 3):
        out.append((float(arr[i]), float(arr[i + 1]), float(arr[i + 2])))
    return out


def _flatten_keypoints_2d(kps: List[Keypoint]) -> List[float]:
    """Inverse of :func:`_reshape_keypoints_2d`."""
    out: List[float] = []
    for x, y, c in kps:
        out.extend([float(x), float(y), float(c)])
    return out


def _sum_conf(arr: Optional[List[float]], sample_step: int = 1) -> float:
    """Sum the positive confidences of a flat array, visiting every ``sample_step`` point.

    Used as a cheap "how much of this person did the detector actually see" score;
    sub-sampling keeps dense face/hand arrays from dominating the body term.
    """
    if not arr:
        return 0.0
    s = 0.0
    for i in range(2, len(arr), 3 * sample_step):
        try:
            c = float(arr[i])
        except Exception:
            c = 0.0
        if c > 0:
            s += c
    return s


def _count_valid_points(arr: Optional[List[float]], *, conf_gate: float) -> int:
    """Count points in a flat array that clear ``conf_gate`` and are not ``(0, 0)``."""
    if not isinstance(arr, list) or len(arr) % 3 != 0:
        return 0
    return sum(
        1
        for i in range(0, len(arr), 3)
        if float(arr[i + 2]) >= conf_gate and not (float(arr[i]) == 0 and float(arr[i + 1]) == 0)
    )


def _zero_out_kps(arr: Optional[List[float]]) -> Optional[List[float]]:
    """Return a copy of a flat array with every point blanked to ``(0, 0, 0)``.

    Non-list / malformed input is passed through untouched, because sequences
    routinely contain ``None`` for frames where nobody was detected.
    """
    if not isinstance(arr, list) or len(arr) % 3 != 0:
        return arr
    out = list(arr)
    for i in range(0, len(out), 3):
        out[i : i + 3] = [0.0, 0.0, 0.0]
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _dist(a: Point, b: Point) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _body_center_from_pose(pose_arr: Optional[List[float]]) -> Optional[Point]:
    """Estimate a person's centre from their torso joints.

    Prefers shoulders/hips/neck (COCO indices 2, 5, 8, 11, 1); falls back to the
    mean of every visible joint, and returns ``None`` if nothing is visible.
    """
    if not pose_arr:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    idxs = [2, 5, 8, 11, 1]
    pts: List[Point] = []
    for idx in idxs:
        if idx < len(kps) and _is_valid_xyc(*kps[idx]):
            pts.append((kps[idx][0], kps[idx][1]))
    if not pts:
        for x, y, c in kps:
            if _is_valid_xyc(x, y, c):
                pts.append((x, y))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _estimate_torso_scale(pose: List[Keypoint]) -> Optional[float]:
    """Estimate torso size in pixels, or ``None`` when no torso span is measurable.

    Averages whichever of shoulder-span, hip-span and neck-to-hip distances are
    available, giving every other threshold in the pipeline a subject-relative unit.
    """

    def dist(i: int, k: int) -> Optional[float]:
        if i >= len(pose) or k >= len(pose):
            return None
        if not _is_valid_xyc(*pose[i]) or not _is_valid_xyc(*pose[k]):
            return None
        return math.hypot(pose[i][0] - pose[k][0], pose[i][1] - pose[k][1])

    cand = [c for c in [dist(2, 5), dist(8, 11), dist(1, 8), dist(1, 11)] if c is not None and c > 1e-3]
    if not cand:
        return None
    return float(sum(cand) / len(cand))


def _body_head_root_scale_from_pose(
    pose_arr: Optional[List[float]], *, conf_gate: float
) -> Optional[Tuple[Point, float]]:
    """Derive a ``(root, scale)`` frame of reference for the head from a body pose.

    The root is the mean of the visible head joints (nose, neck, eyes, ears) and the
    scale is the mean of the eye span, ear span and shoulder span.  Used to express
    face keypoints in a body-relative space so they can be smoothed without fighting
    global subject motion.  Returns ``None`` if no scale can be measured.
    """
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return None
    kps = _reshape_keypoints_2d(pose_arr)

    def vis(j: int) -> Optional[Point]:
        return (
            (float(kps[j][0]), float(kps[j][1]))
            if j < len(kps) and kps[j][2] >= conf_gate and not (kps[j][0] == 0 and kps[j][1] == 0)
            else None
        )

    pts = [p for p in (vis(j) for j in [0, 1, 14, 15, 16, 17]) if p is not None]
    if not pts:
        return None
    root = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    cands = [
        math.hypot(a[0] - b[0], a[1] - b[1])
        for a, b in ((vis(14), vis(15)), (vis(16), vis(17)), (vis(2), vis(5)))
        if a and b and math.hypot(a[0] - b[0], a[1] - b[1]) > 1e-3
    ]
    return (root, float(sum(cands) / len(cands))) if cands else None


def _body_wrist_root_scale_from_pose(
    pose_arr: Optional[List[float]], *, side: str, conf_gate: float
) -> Optional[Tuple[Point, float]]:
    """Derive a ``(root, scale)`` frame of reference for one hand from a body pose.

    The root is the body wrist joint of ``side`` (``"left"`` or ``"right"``); the
    scale is the forearm length, falling back to the shoulder span.  Returns ``None``
    when the wrist is missing or no scale can be measured.
    """
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    w, e = (4, 3) if side == "right" else (7, 6)

    def vis(j: int) -> Optional[Point]:
        return (
            (float(kps[j][0]), float(kps[j][1]))
            if j < len(kps) and kps[j][2] >= conf_gate and not (kps[j][0] == 0 and kps[j][1] == 0)
            else None
        )

    pw = vis(w)
    if not pw:
        return None
    pe = vis(e)
    s = math.hypot(pw[0] - pe[0], pw[1] - pe[1]) if pe and math.hypot(pw[0] - pe[0], pw[1] - pe[1]) > 1e-3 else None
    if s is None:
        p2, p5 = vis(2), vis(5)
        if p2 and p5 and math.hypot(p2[0] - p5[0], p2[1] - p5[1]) > 1e-3:
            s = math.hypot(p2[0] - p5[0], p2[1] - p5[1])
    return (pw, float(s)) if s else None
