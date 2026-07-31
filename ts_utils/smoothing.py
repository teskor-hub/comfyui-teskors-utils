"""Temporal smoothers for pose sequences.

Three complementary passes live here:

* :func:`_smooth_body_pose` - a causal, velocity-damped EMA applied one frame at a
  time.  It carries state between frames and is what makes live playback look calm.
* :func:`_zero_lag_ema_pose_seq` - a forward+backward EMA over a whole sequence.
  Running the same filter in both directions cancels the phase lag a single-pass EMA
  introduces, at the cost of needing the entire clip up front.
* :func:`_smooth_dense_seq_anchored_to_body` - smooths face/hand points in a
  body-relative frame of reference so that global subject motion is not mistaken for
  jitter.

Depends on :mod:`ts_utils.filters` (for the 3-tap median).  That edge is one-way:
``filters`` must never import ``smoothing``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import SmoothConfig
from .filters import _median3_pose_seq
from .keypoints import (
    _body_head_root_scale_from_pose,
    _body_wrist_root_scale_from_pose,
    _flatten_keypoints_2d,
    _is_valid_xyc,
    _reshape_keypoints_2d,
)

__all__ = [
    "BodyState",
    "_smooth_body_pose",
    "_zero_lag_ema_pose_seq",
    "_smooth_dense_seq_anchored_to_body",
]

FlatArr = Optional[List[float]]


@dataclass
class BodyState:
    """Per-joint filter memory for :func:`_smooth_body_pose`.

    IMPORTANT: this object is mutated in place and must stay a local of the caller's
    smoothing run.  Parking it on a node instance or at module level would leak EMA
    state between separate videos - ComfyUI reuses node instances across executions,
    so the first frames of one job would be dragged toward the last frame of the
    previous one.

    Attributes:
        last_xy: last emitted position per joint, or ``None`` if the joint was
            missing in the previous frame (which resets its history).
        last_v: smoothed per-joint velocity, in pixels per frame.
    """

    last_xy: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    last_v: List[Tuple[float, float]] = field(default_factory=list)

    def __init__(self, joints: int) -> None:
        self.last_xy = [None] * joints
        self.last_v = [(0.0, 0.0)] * joints


def _smooth_body_pose(pose_arr: FlatArr, state: BodyState, *, cfg: SmoothConfig) -> FlatArr:
    """Apply one frame of causal, velocity-damped EMA smoothing to a body pose.

    For each joint the filter predicts where it should be (last position plus
    smoothed velocity), blends that prediction with the new measurement by
    ``cfg.ALPHA_BODY``, and clamps the resulting step to ``cfg.MAX_STEP_BODY`` pixels
    so a mis-detection cannot teleport a limb.  Sub-``cfg.EPS`` motion is treated as
    zero, which stops a stationary subject from shimmering.

    Joints below ``cfg.CONF_GATE_BODY`` are emitted as ``(0, 0, 0)`` and have their
    history cleared, so they re-enter cleanly rather than snapping back from a stale
    position.  ``state`` is mutated in place and resizes itself if the joint count
    changes.  Returns a fresh list; ``None`` in gives ``None`` out.
    """
    if pose_arr is None:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    J = len(kps)
    if len(state.last_xy) != J:
        state.last_xy = [None] * J
        state.last_v = [(0.0, 0.0)] * J

    out: List[Tuple[float, float, float]] = []
    for j in range(J):
        x, y, c = kps[j]
        last = state.last_xy[j]
        vx_last, vy_last = state.last_v[j]
        valid_in = _is_valid_xyc(x, y, c) and (c >= cfg.CONF_GATE_BODY)

        if valid_in:
            if last is None:
                state.last_xy[j] = (x, y)
                state.last_v[j] = (0.0, 0.0)
                out.append((x, y, float(c)))
                continue

            dx_raw, dy_raw = x - last[0], y - last[1]
            if abs(dx_raw) < cfg.EPS:
                dx_raw = 0.0
            if abs(dy_raw) < cfg.EPS:
                dy_raw = 0.0

            vx = cfg.VEL_ALPHA * dx_raw + (1.0 - cfg.VEL_ALPHA) * vx_last
            vy = cfg.VEL_ALPHA * dy_raw + (1.0 - cfg.VEL_ALPHA) * vy_last
            nx = cfg.ALPHA_BODY * x + (1.0 - cfg.ALPHA_BODY) * (last[0] + vx)
            ny = cfg.ALPHA_BODY * y + (1.0 - cfg.ALPHA_BODY) * (last[1] + vy)

            d = math.hypot(nx - last[0], ny - last[1])
            if d > cfg.MAX_STEP_BODY and d > 1e-6:
                scale = cfg.MAX_STEP_BODY / d
                nx = last[0] + (nx - last[0]) * scale
                ny = last[1] + (ny - last[1]) * scale
                vx, vy = nx - last[0], ny - last[1]

            state.last_xy[j], state.last_v[j] = (nx, ny), (vx, vy)
            out.append((nx, ny, float(c)))
        else:
            state.last_xy[j] = None
            state.last_v[j] = (0.0, 0.0)
            out.append((0.0, 0.0, 0.0))

    return _flatten_keypoints_2d(out)


def _zero_lag_ema_pose_seq(pose_seq: List[FlatArr], *, alpha: float, conf_gate: float) -> List[FlatArr]:
    """Smooth a whole sequence with a forward then backward EMA (zero net phase lag).

    ``alpha`` is the weight of the current measurement: 1.0 is a pass-through, small
    values smooth hard.  Confidences are left untouched - only x/y are filtered - and
    a joint dropping below ``conf_gate`` resets its history so it does not drag a
    stale position into its next appearance.
    """
    if not pose_seq:
        return pose_seq
    J = next((len(arr) // 3 for arr in pose_seq if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0), None)
    if J is None:
        return pose_seq
    T = len(pose_seq)

    def is_vis(arr: List[float], j: int) -> bool:
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j + 0]) == 0 and float(arr[3 * j + 1]) == 0)

    fwd: List[FlatArr] = [None] * T
    last: List[Optional[Tuple[float, float]]] = [None] * J
    for t in range(T):
        if not isinstance(pose_seq[t], list) or len(pose_seq[t]) != J * 3:
            fwd[t] = pose_seq[t]
            continue
        out = list(pose_seq[t])
        for j in range(J):
            if is_vis(pose_seq[t], j):
                x, y = float(pose_seq[t][3 * j]), float(pose_seq[t][3 * j + 1])
                sx, sy = (
                    (x, y)
                    if last[j] is None
                    else (alpha * x + (1 - alpha) * last[j][0], alpha * y + (1 - alpha) * last[j][1])
                )
                last[j], out[3 * j], out[3 * j + 1] = (sx, sy), float(sx), float(sy)
            else:
                last[j] = None
        fwd[t] = out

    bwd: List[FlatArr] = [None] * T
    last = [None] * J
    for t in range(T - 1, -1, -1):
        if not isinstance(fwd[t], list) or len(fwd[t]) != J * 3:
            bwd[t] = fwd[t]
            continue
        out = list(fwd[t])
        for j in range(J):
            if is_vis(fwd[t], j):
                x, y = float(fwd[t][3 * j]), float(fwd[t][3 * j + 1])
                sx, sy = (
                    (x, y)
                    if last[j] is None
                    else (alpha * x + (1 - alpha) * last[j][0], alpha * y + (1 - alpha) * last[j][1])
                )
                last[j], out[3 * j], out[3 * j + 1] = (sx, sy), float(sx), float(sy)
            else:
                last[j] = None
        bwd[t] = out
    return bwd


def _smooth_dense_seq_anchored_to_body(
    dense_seq: List[FlatArr],
    body_pose_seq: List[FlatArr],
    *,
    kind: str,
    conf_gate_dense: float,
    conf_gate_body: float,
    median3: bool,
    zero_lag_alpha: float,
) -> List[FlatArr]:
    """Smooth face or hand keypoints in a body-relative frame of reference.

    ``kind`` is ``"face"``, ``"hand_left"`` or ``"hand_right"``.  Each frame's dense
    points are expressed relative to a root and scale taken from the body skeleton
    (head span for the face, forearm for a hand), filtered there, then mapped back.
    Working in that space means a subject walking across frame does not look like
    jitter to the filter.

    KNOWN NO-OP - DO NOT "CLEAN UP" IN ISOLATION.  The normalisation comprehension
    below emits one value per keypoint instead of three, so every row of ``norm_seq``
    has length ``Jd`` rather than ``Jd * 3``.  The un-normalisation loop's length
    check therefore rejects every frame and this function returns an unmodified copy
    of ``dense_seq``.  It has behaved this way for the whole life of the node, and
    since the node calls it with ``keep_face_untouched=False`` on every run,
    repairing it would change every rendered frame for every existing user.  The fix
    belongs behind a new opt-in widget, together with a regression comparison - not
    in a refactor whose contract is bit-identical output.
    """
    if not dense_seq:
        return dense_seq
    Jd = next((len(a) // 3 for a in dense_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if Jd is None:
        return dense_seq
    T = len(dense_seq)
    out: List[FlatArr] = [None if a is None else list(a) for a in dense_seq]
    norm_seq: List[FlatArr] = [None] * T

    for t in range(T):
        arr, body = out[t], body_pose_seq[t] if t < len(body_pose_seq) else None
        if not isinstance(arr, list) or len(arr) != Jd * 3 or not isinstance(body, list):
            norm_seq[t] = arr
            continue
        rs = (
            _body_head_root_scale_from_pose(body, conf_gate=conf_gate_body)
            if kind == "face"
            else _body_wrist_root_scale_from_pose(
                body, side="left" if kind == "hand_left" else "right", conf_gate=conf_gate_body
            )
        )
        if not rs or rs[1] <= 1e-6:
            norm_seq[t] = arr
            continue
        (rx, ry), s = rs
        norm_seq[t] = [
            (x - rx) / s if i % 3 == 0 else (y - ry) / s if i % 3 == 1 else c
            for i, (x, y, c) in enumerate(zip(arr[0::3], arr[1::3], arr[2::3]))
        ]

    if median3:
        norm_seq = _median3_pose_seq(norm_seq, conf_gate=conf_gate_dense)
    norm_seq = _zero_lag_ema_pose_seq(norm_seq, alpha=zero_lag_alpha, conf_gate=conf_gate_dense)

    for t in range(T):
        arrn, body = norm_seq[t], body_pose_seq[t] if t < len(body_pose_seq) else None
        if not isinstance(arrn, list) or len(arrn) != Jd * 3 or not isinstance(body, list):
            continue
        rs = (
            _body_head_root_scale_from_pose(body, conf_gate=conf_gate_body)
            if kind == "face"
            else _body_wrist_root_scale_from_pose(
                body, side="left" if kind == "hand_left" else "right", conf_gate=conf_gate_body
            )
        )
        if not rs or rs[1] <= 1e-6:
            continue
        (rx, ry), s = rs
        for j in range(Jd):
            if (
                out[t][3 * j + 2] >= conf_gate_dense
                and arrn[3 * j + 2] >= conf_gate_dense
                and not (out[t][3 * j] == 0 and out[t][3 * j + 1] == 0)
            ):
                out[t][3 * j : 3 * j + 2] = [rx + arrn[3 * j] * s, ry + arrn[3 * j + 1] * s]
    return out
