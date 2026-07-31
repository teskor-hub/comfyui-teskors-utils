"""Sequence- and frame-level cleanup passes for OpenPose keypoints.

These are the "repair" half of the pipeline: they delete implausible detections,
interpolate short dropouts, keep body parts appearing and disappearing together, and
reconcile the body skeleton with the more accurate hand keypoints.

Two invariants hold throughout and must be preserved by any future change:

* Every threshold arrives as an explicit argument.  This module never reads
  :mod:`ts_utils.config`, which keeps it trivially unit-testable and keeps the
  dependency graph acyclic (``filters`` is imported by ``smoothing``, never the
  reverse).
* Everything is ``None``-tolerant.  Sequence entries are routinely ``None`` (no
  person detected in that frame) and malformed arrays are passed through unchanged
  rather than raising, because real videos have dropped detections.
"""

from __future__ import annotations

import math
from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Tuple

from .keypoints import (
    COCO18_EDGES,
    COCO18_NEIGHBORS,
    HAND21_EDGES,
    _body_center_from_pose,
    _count_valid_points,
    _estimate_torso_scale,
    _flatten_keypoints_2d,
    _reshape_keypoints_2d,
    _zero_out_kps,
)

__all__ = [
    "_suppress_spatial_outliers_in_pose_arr",
    "_suppress_isolated_joints_in_pose_arr",
    "_denoise_and_fill_gaps_pose_seq",
    "_median3_pose_seq",
    "_sync_group_appearances",
    "_apply_root_scale",
    "_carry_pose_when_torso_missing",
    "_force_full_torso_pair",
    "_remove_short_presence_runs_kps_seq",
    "_zero_sparse_frames_kps_seq",
    "_suppress_spatial_outliers_in_hand_arr",
    "_pin_body_wrist_to_hand",
    "_fix_elbow_using_wrist",
]

FlatArr = Optional[List[float]]
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Per-frame spatial sanity
# ---------------------------------------------------------------------------


def _suppress_spatial_outliers_in_pose_arr(
    pose_arr: FlatArr, *, conf_gate: float, torso_radius_factor: float, bone_max_factor: float
) -> FlatArr:
    """Delete body joints that are geometrically impossible for this subject.

    Two tests, both scaled by the frame's own torso size so they work at any subject
    distance: a joint further than ``torso_radius_factor`` torsos from the body
    centre is dropped, and for any bone longer than ``bone_max_factor`` torsos the
    lower-confidence endpoint is dropped.  Frames whose torso scale cannot be
    measured are returned untouched.
    """
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr
    pose = _reshape_keypoints_2d(pose_arr)
    J = len(pose)
    center = _body_center_from_pose(pose_arr)
    scale = _estimate_torso_scale(pose)
    if center is None or scale is None:
        return pose_arr

    max_r, max_bone = torso_radius_factor * scale, bone_max_factor * scale
    out = [list(p) for p in pose]

    def visible(j: int) -> bool:
        return j < J and out[j][2] >= conf_gate and not (out[j][0] == 0 and out[j][1] == 0)

    for j in range(J):
        if visible(j) and math.hypot(out[j][0] - center[0], out[j][1] - center[1]) > max_r:
            out[j] = [0.0, 0.0, 0.0]

    for a, b in COCO18_EDGES:
        if a >= J or b >= J:
            continue
        if not visible(a) or not visible(b):
            continue
        if math.hypot(out[a][0] - out[b][0], out[a][1] - out[b][1]) > max_bone:
            if out[a][2] <= out[b][2]:
                out[a] = [0.0, 0.0, 0.0]
            else:
                out[b] = [0.0, 0.0, 0.0]

    flat: List[float] = []
    for p in out:
        flat.extend(p)
    return flat


def _suppress_isolated_joints_in_pose_arr(
    pose_arr: FlatArr, *, conf_gate: float, keep: Optional[AbstractSet[int]] = None
) -> FlatArr:
    """Delete visible joints that have no visible neighbour in the skeleton graph.

    A lone floating joint is almost always a false positive.  Joints listed in
    ``keep`` (typically the torso) are exempt so that a partially occluded subject
    does not lose their anchor.
    """
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr
    pose = _reshape_keypoints_2d(pose_arr)
    J, out = len(pose), [list(p) for p in pose]
    keep = keep or frozenset()

    def vis(j: int) -> bool:
        return j < J and out[j][2] >= conf_gate and not (out[j][0] == 0 and out[j][1] == 0)

    for j in range(J):
        if j in keep or not vis(j):
            continue
        if not any(n < J and vis(n) for n in COCO18_NEIGHBORS.get(j, frozenset())):
            out[j] = [0.0, 0.0, 0.0]

    flat: List[float] = []
    for p in out:
        flat.extend(p)
    return flat


def _suppress_spatial_outliers_in_hand_arr(
    hand_arr: FlatArr, *, conf_gate: float, max_bone_factor: float = 3.0
) -> FlatArr:
    """Drop hand keypoints joined by an implausibly long finger bone.

    Scale comes from the bounding box of the visible points, so it adapts to hand
    size on screen.  Needs a full 21-point hand and at least 6 visible points;
    otherwise the input is returned untouched.
    """
    if not isinstance(hand_arr, list) or len(hand_arr) % 3 != 0:
        return hand_arr
    pts = _reshape_keypoints_2d(hand_arr)
    J = len(pts)
    if J < 21:
        return hand_arr
    out = [list(p) for p in pts]

    def vis(j: int) -> bool:
        return out[j][2] >= conf_gate and not (out[j][0] == 0 and out[j][1] == 0)

    vv = [(x, y) for x, y, c in out if c >= conf_gate and not (x == 0 and y == 0)]
    if len(vv) < 6:
        return hand_arr
    xs, ys = [p[0] for p in vv], [p[1] for p in vv]
    s = max(max(xs) - min(xs), max(ys) - min(ys))
    if s <= 1e-3:
        return hand_arr
    max_bone = max_bone_factor * s
    for a, b in HAND21_EDGES:
        if a >= J or b >= J or not vis(a) or not vis(b):
            continue
        if math.hypot(out[a][0] - out[b][0], out[a][1] - out[b][1]) > max_bone:
            if out[a][2] <= out[b][2]:
                out[a] = [0.0, 0.0, 0.0]
            else:
                out[b] = [0.0, 0.0, 0.0]
    return _flatten_keypoints_2d([(x, y, c) for x, y, c in out])


# ---------------------------------------------------------------------------
# Temporal repair
# ---------------------------------------------------------------------------


def _denoise_and_fill_gaps_pose_seq(
    pose_arr_seq: List[FlatArr], *, conf_gate: float, min_run: int, max_gap: int
) -> List[FlatArr]:
    """Per joint: delete visibility runs shorter than ``min_run``, then bridge short gaps.

    Runs shorter than ``min_run`` frames are flicker and get zeroed.  Gaps of at most
    ``max_gap`` frames between two visible runs are linearly interpolated, carrying
    the lower of the two bracketing confidences.  Pass ``max_gap=0`` to run the
    de-flicker half only.
    """
    if not pose_arr_seq:
        return pose_arr_seq
    J = next(
        (len(arr) // 3 for arr in pose_arr_seq if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0), None
    )
    if J is None:
        return pose_arr_seq
    T = len(pose_arr_seq)
    out_seq: List[FlatArr] = [list(arr) if isinstance(arr, list) and len(arr) == J * 3 else arr for arr in pose_arr_seq]

    def usable(arr: Any) -> bool:
        # A list of the wrong length is treated as absent. Without the length test
        # is_vis() reads past the end, and the slice assignments below would grow
        # the caller's own list, since short lists are passed through by reference.
        return isinstance(arr, list) and len(arr) == J * 3

    def is_vis(arr: List[float], j: int) -> bool:
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j + 0]) == 0 and float(arr[3 * j + 1]) == 0)

    for j in range(J):
        start: Optional[int] = None
        for t in range(T + 1):
            cur = t < T and usable(out_seq[t]) and is_vis(out_seq[t], j)
            if cur and start is None:
                start = t
            if not cur and start is not None:
                if (t - start) < min_run:
                    for k in range(start, t):
                        if usable(out_seq[k]):
                            out_seq[k][3 * j : 3 * j + 3] = [0.0, 0.0, 0.0]
                start = None

    for j in range(J):
        t = 0
        while t < T:
            arr = out_seq[t]
            if usable(arr) and is_vis(arr, j):
                last_vis_t = t
                t += 1
                while t < T:
                    if usable(out_seq[t]) and is_vis(out_seq[t], j):
                        break
                    t += 1
                if t < T and (t - last_vis_t - 1) > 0 and (t - last_vis_t - 1) <= max_gap:
                    a, b = out_seq[last_vis_t], out_seq[t]
                    ax, ay, ac = float(a[3 * j]), float(a[3 * j + 1]), float(a[3 * j + 2])
                    bx, by, bc = float(b[3 * j]), float(b[3 * j + 1]), float(b[3 * j + 2])
                    for k in range(last_vis_t + 1, t):
                        if usable(out_seq[k]):
                            r = (k - last_vis_t) / (t - last_vis_t)
                            out_seq[k][3 * j : 3 * j + 3] = [ax + (bx - ax) * r, ay + (by - ay) * r, min(ac, bc)]
            else:
                t += 1
    return out_seq


def _median3_pose_seq(pose_seq: List[FlatArr], *, conf_gate: float) -> List[FlatArr]:
    """Replace each visible point with the median of itself and its two neighbours.

    A 3-tap median kills single-frame spikes without the lag of an average.  Only
    the neighbours in which the joint is also visible take part, and points that are
    not visible in the current frame are left alone.
    """
    if not pose_seq:
        return pose_seq
    J = next((len(a) // 3 for a in pose_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_seq
    T = len(pose_seq)

    def is_vis(arr: List[float], j: int) -> bool:
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)

    out_seq: List[FlatArr] = []
    for t in range(T):
        if not isinstance(pose_seq[t], list) or len(pose_seq[t]) != J * 3:
            out_seq.append(pose_seq[t])
            continue
        out = list(pose_seq[t])
        a0, a1, a2 = pose_seq[max(0, t - 1)], pose_seq[t], pose_seq[min(T - 1, t + 1)]
        for j in range(J):
            if not is_vis(pose_seq[t], j):
                continue
            xs, ys = [], []
            for aa in (a0, a1, a2):
                if isinstance(aa, list) and len(aa) == J * 3 and is_vis(aa, j):
                    xs.append(float(aa[3 * j]))
                    ys.append(float(aa[3 * j + 1]))
            if len(xs) >= 2:
                xs.sort()
                ys.sort()
                out[3 * j], out[3 * j + 1] = float(xs[len(xs) // 2]), float(ys[len(ys) // 2])
        out_seq.append(out)
    return out_seq


def _sync_group_appearances(
    pose_arr_seq: List[FlatArr], *, group: AbstractSet[int], conf_gate: float, lookahead: int
) -> List[FlatArr]:
    """Make the joints of ``group`` appear together rather than one at a time.

    Whenever part of the group is visible but a member is not, and that member
    becomes visible within ``lookahead`` frames, the missing stretch is filled: by
    interpolation from its last known position, or by holding its future position if
    it has never been seen.  Applied to the torso, this stops the hips popping in and
    out a few frames apart.
    """
    if not pose_arr_seq:
        return pose_arr_seq
    J = next((len(a) // 3 for a in pose_arr_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_arr_seq
    T = len(pose_arr_seq)
    out: List[FlatArr] = [list(a) if isinstance(a, list) and len(a) == J * 3 else a for a in pose_arr_seq]

    def usable(arr: Any) -> bool:
        # See _denoise_and_fill_gaps_pose_seq: a wrong-length list is not a pose.
        return isinstance(arr, list) and len(arr) == J * 3

    def is_vis(arr: List[float], j: int) -> bool:
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)

    for t in range(T):
        arr = out[t]
        if not usable(arr):
            continue
        vis = {j for j in group if j < J and is_vis(arr, j)}
        if not vis:
            continue
        for j in list({j for j in group if j < J and j not in vis}):
            t2 = next(
                (tt for tt in range(t + 1, min(T, t + 1 + lookahead)) if usable(out[tt]) and is_vis(out[tt], j)),
                None,
            )
            if t2 is None:
                continue
            last_t = next((tb for tb in range(t - 1, -1, -1) if usable(out[tb]) and is_vis(out[tb], j)), None)
            b = out[t2]
            if last_t is None:
                for k in range(t, t2):
                    if usable(out[k]):
                        out[k][3 * j : 3 * j + 3] = b[3 * j : 3 * j + 3]
            else:
                a = out[last_t]
                if float(a[3 * j]) == 0 and float(a[3 * j + 1]) == 0:
                    continue
                c_fill = min(float(a[3 * j + 2]), float(b[3 * j + 2]))
                for tt in range(t, t2):
                    if usable(out[tt]):
                        r = (tt - last_t) / (t2 - last_t)
                        out[tt][3 * j : 3 * j + 3] = [
                            float(a[3 * j]) + (float(b[3 * j]) - float(a[3 * j])) * r,
                            float(a[3 * j + 1]) + (float(b[3 * j + 1]) - float(a[3 * j + 1])) * r,
                            float(c_fill),
                        ]
    return out


def _remove_short_presence_runs_kps_seq(
    seq: List[FlatArr], *, conf_gate: float, min_points_present: int, min_run: int
) -> List[FlatArr]:
    """Blank whole limbs that are only "present" for a handful of consecutive frames.

    Presence is defined as at least ``min_points_present`` valid points.  Runs
    shorter than ``min_run`` frames are zeroed entirely - a hand that flashes on for
    two frames is noise, not a hand.
    """
    if not seq:
        return seq
    out: List[FlatArr] = [None if a is None else list(a) for a in seq]
    start: Optional[int] = None
    for t in range(len(seq) + 1):
        cur = t < len(seq) and _count_valid_points(seq[t], conf_gate=conf_gate) >= min_points_present
        if cur and start is None:
            start = t
        if not cur and start is not None:
            if (t - start) < min_run:
                for k in range(start, t):
                    out[k] = _zero_out_kps(out[k])
            start = None
    return out


def _zero_sparse_frames_kps_seq(seq: List[FlatArr], *, conf_gate: float, min_points_present: int) -> List[FlatArr]:
    """Blank any frame holding fewer than ``min_points_present`` valid points.

    A hand detected as three scattered points is worse than no hand at all.
    """
    if not seq:
        return seq
    return [
        (
            _zero_out_kps(a)
            if isinstance(a, list) and _count_valid_points(a, conf_gate=conf_gate) < min_points_present
            else a
        )
        for a in seq
    ]


# ---------------------------------------------------------------------------
# Root-scale carry
# ---------------------------------------------------------------------------


def _apply_root_scale(
    pose_arr: FlatArr, *, src_root: Point, src_scale: float, dst_root: Point, dst_scale: float
) -> FlatArr:
    """Re-project a pose from one ``(root, scale)`` frame of reference into another.

    Invalid points keep their sentinel values so they stay recognisably missing.
    """
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0 or src_scale <= 1e-6 or dst_scale <= 1e-6:
        return pose_arr
    kps = _reshape_keypoints_2d(pose_arr)
    s = dst_scale / src_scale
    out = [
        (
            (dst_root[0] + (x - src_root[0]) * s, dst_root[1] + (y - src_root[1]) * s, c)
            if c > 0 and not (x == 0 and y == 0)
            else (x, y, c)
        )
        for x, y, c in kps
    ]
    return _flatten_keypoints_2d(out)


def _carry_pose_when_torso_missing(
    pose_seq: List[FlatArr],
    *,
    conf_gate: float,
    max_carry: int,
    anchor_joints: Iterable[int],
    min_anchors: int,
    allow_disappear_joints: AbstractSet[int],
) -> List[FlatArr]:
    """Hold the last complete torso in place while the detector loses it.

    While at least ``min_anchors`` of ``anchor_joints`` (head and arms) are still
    visible, the most recent frame that had a good torso is re-projected onto the
    current frame's anchor root/scale and used to fill in the missing torso and leg
    joints, for at most ``max_carry`` consecutive frames.  Joints in
    ``allow_disappear_joints`` are excluded: they are allowed to genuinely leave the
    frame and inventing them looks worse than losing them.
    """
    if not pose_seq:
        return pose_seq
    J = next((len(a) // 3 for a in pose_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_seq
    out: List[FlatArr] = [a if a is None else list(a) for a in pose_seq]
    FILL = {1, 8, 9, 10, 11, 12, 13} - set(allow_disappear_joints)

    def is_vis(arr: List[float], j: int) -> bool:
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)

    def rs_anchors(arr: List[float]) -> Optional[Tuple[Point, float]]:
        pts = [(float(arr[3 * j]), float(arr[3 * j + 1])) for j in anchor_joints if j < J and is_vis(arr, j)]
        if len(pts) < min_anchors:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        s = max(max(xs) - min(xs), max(ys) - min(ys))
        if s <= 1e-3:
            return None
        return (sum(xs) / len(pts), sum(ys) / len(pts)), float(s)

    last_good, last_rs, carry = None, None, 0
    for t, arr in enumerate(out):
        if not isinstance(arr, list) or len(arr) != J * 3:
            continue
        rs = rs_anchors(arr)
        if (
            sum(1 for j in anchor_joints if j < J and is_vis(arr, j)) >= min_anchors
            and rs
            and sum(1 for j in FILL if j < J and is_vis(arr, j)) >= 2
        ):
            last_good, last_rs, carry = list(arr), rs, max_carry
            continue
        if rs and last_good and last_rs and carry > 0:
            carried = _apply_root_scale(
                last_good, src_root=last_rs[0], src_scale=last_rs[1], dst_root=rs[0], dst_scale=rs[1]
            )
            if isinstance(carried, list) and len(carried) == J * 3:
                for j in FILL:
                    if (
                        j < J
                        and not is_vis(arr, j)
                        and (float(carried[3 * j]) != 0 or float(carried[3 * j + 1]) != 0)
                        and float(carried[3 * j + 2]) > 0
                    ):
                        arr[3 * j : 3 * j + 3] = [
                            float(carried[3 * j]),
                            float(carried[3 * j + 1]),
                            max(min(float(carried[3 * j + 2]), 0.60), conf_gate),
                        ]
                out[t], carry = arr, carry - 1
                continue
        carry = max(carry - 1, 0)
    return out


def _force_full_torso_pair(
    pose_seq: List[FlatArr],
    *,
    conf_gate: float,
    anchor_joints: Iterable[int],
    min_anchors: int,
    max_lookback: int = 240,
    fill_legs_with_hip: bool = True,
    always_fill_if_one_hip: bool = True,
) -> List[FlatArr]:
    """Keep both hips (and optionally both legs) present or absent together.

    A single visible hip renders as a lopsided skeleton that flickers between sides.
    Whenever exactly one hip is missing - or both are, if ``always_fill_if_one_hip``
    - the last frame that had both is re-projected onto the current anchors and used
    to fill the missing side, provided it is within ``max_lookback`` frames.
    """
    if not pose_seq:
        return pose_seq
    J = next((len(a) // 3 for a in pose_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_seq
    out: List[FlatArr] = [a if a is None else list(a) for a in pose_seq]

    def is_vis(arr: List[float], j: int) -> bool:
        return (
            j < J and float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)
        )

    def rs_anchors(arr: List[float]) -> Optional[Tuple[Point, float]]:
        pts = [(float(arr[3 * j]), float(arr[3 * j + 1])) for j in anchor_joints if j < J and is_vis(arr, j)]
        if len(pts) < min_anchors:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        s = max(max(xs) - min(xs), max(ys) - min(ys))
        return ((sum(xs) / len(pts), sum(ys) / len(pts)), float(s)) if s > 1e-3 else None

    last_idx, last_f, last_rs = None, None, None
    for t, arr in enumerate(out):
        if not isinstance(arr, list) or len(arr) != J * 3:
            continue
        rs = rs_anchors(arr)
        r_ok, l_ok = is_vis(arr, 8), is_vis(arr, 11)
        if rs and sum(1 for j in anchor_joints if is_vis(arr, j)) >= min_anchors and r_ok and l_ok:
            last_idx, last_f, last_rs = t, list(arr), rs
            continue
        if (
            not last_f
            or not last_rs
            or (t - last_idx) > max_lookback
            or not rs
            or (r_ok and l_ok)
            or (not r_ok and not l_ok and not always_fill_if_one_hip)
        ):
            continue
        carried = _apply_root_scale(last_f, src_root=last_rs[0], src_scale=last_rs[1], dst_root=rs[0], dst_scale=rs[1])
        if not isinstance(carried, list) or len(carried) != J * 3:
            continue

        def cp(j: int) -> None:
            if (
                j < J
                and not is_vis(arr, j)
                and (float(carried[3 * j]) != 0 or float(carried[3 * j + 1]) != 0)
                and float(carried[3 * j + 2]) > 0
            ):
                arr[3 * j : 3 * j + 3] = [
                    float(carried[3 * j]),
                    float(carried[3 * j + 1]),
                    max(min(float(carried[3 * j + 2]), 0.60), conf_gate),
                ]

        if not r_ok:
            cp(8)
            if fill_legs_with_hip:
                cp(9)
                cp(10)
        if not l_ok:
            cp(11)
            if fill_legs_with_hip:
                cp(12)
                cp(13)
        out[t] = arr
    return out


# ---------------------------------------------------------------------------
# Body <-> hand reconciliation (mutate the person dict in place)
# ---------------------------------------------------------------------------


def _pin_body_wrist_to_hand(
    p_out: Dict[str, Any], *, side: str, conf_gate_body: float, conf_gate_hand: float, blend: float
) -> None:
    """Snap the body wrist joint onto the hand root, in place.

    The 21-point hand model localises the wrist far more precisely than the 18-point
    body model, so where a confident hand exists the body wrist is moved onto it
    (``blend=1.0``) or interpolated towards it.  Does nothing without a confident
    hand root.
    """
    bw, hk = (4, "hand_right_keypoints_2d") if side == "right" else (7, "hand_left_keypoints_2d")
    pose, hand = p_out.get("pose_keypoints_2d"), p_out.get(hk)
    if not isinstance(pose, list) or not isinstance(hand, list) or len(pose) < (bw * 3 + 3) or len(hand) < 3:
        return
    hx, hy, hc = float(hand[0]), float(hand[1]), float(hand[2])
    if hc < conf_gate_hand or (hx == 0.0 and hy == 0.0):
        return
    bx, by, bc = float(pose[bw * 3]), float(pose[bw * 3 + 1]), float(pose[bw * 3 + 2])
    if bc < conf_gate_body or (bx == 0.0 and by == 0.0):
        pose[bw * 3 : bw * 3 + 3] = [hx, hy, float(max(bc, min(hc, 0.9)))]
    else:
        pose[bw * 3 : bw * 3 + 3] = [
            bx * (1.0 - blend) + hx * blend,
            by * (1.0 - blend) + hy * blend,
            float(min(bc, hc)),
        ]
    p_out["pose_keypoints_2d"] = pose


def _fix_elbow_using_wrist(p_out: Dict[str, Any], *, side: str, conf_gate: float) -> None:
    """Re-solve the elbow position by two-bone inverse kinematics, in place.

    Once the wrist has been pinned to the hand, the elbow no longer sits on the arm.
    This intersects the two circles of radius upper-arm around the shoulder and
    forearm around the wrist and picks the solution nearest the old elbow (bone
    lengths are measured from this frame, or assumed 55/45 when the elbow is
    missing).  Does nothing unless shoulder and wrist are both visible.
    """
    pose = p_out.get("pose_keypoints_2d")
    if not isinstance(pose, list) or len(pose) % 3 != 0:
        return
    sh, el, wr = (2, 3, 4) if side == "right" else (5, 6, 7)

    def vis(x: float, y: float, c: float) -> bool:
        return c >= conf_gate and not (x == 0.0 and y == 0.0)

    sx, sy, sc = float(pose[3 * sh]), float(pose[3 * sh + 1]), float(pose[3 * sh + 2])
    ex, ey, ec = float(pose[3 * el]), float(pose[3 * el + 1]), float(pose[3 * el + 2])
    wx, wy, wc = float(pose[3 * wr]), float(pose[3 * wr + 1]), float(pose[3 * wr + 2])
    if not vis(sx, sy, sc) or not vis(wx, wy, wc):
        return
    if vis(ex, ey, ec):
        Lse, Lew = math.hypot(ex - sx, ey - sy), math.hypot(wx - ex, wy - ey)
    else:
        dsw = math.hypot(wx - sx, wy - sy)
        if dsw < 1e-3:
            return
        Lse, Lew = 0.55 * dsw, 0.45 * dsw
    dx, dy = wx - sx, wy - sy
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return
    d2 = max(min(d, (Lse + Lew) - 1e-3), abs(Lse - Lew) + 1e-3)
    a = (Lse * Lse - Lew * Lew + d2 * d2) / (2.0 * d2)
    h = math.sqrt(max(Lse * Lse - a * a, 0.0))
    px, py = sx + a * (dx / d), sy + a * (dy / d)
    rx, ry = -dy / d, dx / d
    e1x, e1y, e2x, e2y = px + h * rx, py + h * ry, px - h * rx, py - h * ry
    nx, ny = (
        (e1x, e1y)
        if not vis(ex, ey, ec) or math.hypot(e1x - ex, e1y - ey) <= math.hypot(e2x - ex, e2y - ey)
        else (e2x, e2y)
    )
    pose[3 * el : 3 * el + 3] = [float(nx), float(ny), float(max(min(ec, 0.8), conf_gate))]
    p_out["pose_keypoints_2d"] = pose
