"""OpenPose skeleton rendering onto a black canvas.

The only module in :mod:`ts_utils` allowed to import cv2, and deliberately kept out
of the package ``__init__`` so the pure-logic modules stay importable on headless
machines where ``opencv-python`` fails on a missing libGL.

Output is plain ``HxWx3`` uint8 BGR-ordered numpy; converting to a torch tensor is
the node's job, not this module's.

The edge tables here are private to rendering and must NOT be aliased to the
topology tables in :mod:`ts_utils.keypoints`.  Limb colour is the edge's *index*
into :data:`BODY_EDGES`, so a differently-ordered table recolours the whole skeleton,
and the keypoints table additionally contains an ``(8, 11)`` hip bar that OpenPose
does not draw.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .keypoints import _reshape_keypoints_2d

__all__ = [
    "OP_COLORS",
    "BODY_EDGES",
    "BODY_EDGE_COLORS",
    "BODY_JOINT_COLORS",
    "HAND_EDGES",
    "_draw_body",
    "_draw_hand",
    "_draw_face",
    "_draw_pose_frame_full",
    "render_pose_frames",
]

Keypoint = Tuple[float, float, float]

OP_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)

BODY_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

BODY_EDGE_COLORS = OP_COLORS[: len(BODY_EDGES)]
BODY_JOINT_COLORS = OP_COLORS

HAND_EDGES: Tuple[Tuple[int, int], ...] = (
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


def _valid_pt(x: float, y: float, c: float, conf_thresh: float) -> bool:
    """Return ``True`` if a keypoint is confident enough to draw."""
    return (c is not None) and (c >= conf_thresh) and not (x == 0 and y == 0)


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert normalised HSV (each in 0..1) to an 8-bit BGR triple."""
    H = int(np.clip(h, 0.0, 1.0) * 179.0)
    S = int(np.clip(s, 0.0, 1.0) * 255.0)
    V = int(np.clip(v, 0.0, 1.0) * 255.0)
    hsv = np.uint8([[[H, S, V]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _looks_normalized(points: List[Keypoint], conf_thresh: float) -> bool:
    """Guess whether coordinates are 0..1 normalised rather than pixels.

    Some producers emit normalised keypoints.  If at least 70% of the visible points
    fall inside the unit square they are treated as normalised and scaled by the
    canvas size at draw time.
    """
    valid = [(x, y, c) for (x, y, c) in points if _valid_pt(x, y, c, conf_thresh)]
    if not valid:
        return False
    in01 = sum(1 for (x, y, _) in valid if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return (in01 / float(len(valid))) >= 0.7


def _draw_body(
    canvas: np.ndarray, pose: List[Keypoint], conf_thresh: float, xinsr_stick_scaling: bool = False
) -> None:
    """Draw the 18-joint body skeleton onto ``canvas`` in place.

    Limbs are filled ellipses at 60% brightness with joints dotted on top, matching
    the reference OpenPose look.  ``xinsr_stick_scaling`` thickens limbs on large
    canvases for the xinsr ControlNet checkpoints.
    """
    CH, CW = canvas.shape[:2]
    # Match comfyui_controlnet_aux's native DWPose/OpenPose renderer.  The
    # previous thin sticks and tiny joints made this output look like a
    # different pose format even though the data was still POSE_KEYPOINT.
    stickwidth = 4
    valid = [(x, y, c) for (x, y, c) in pose if _valid_pt(x, y, c, conf_thresh)]
    norm = False
    if valid:
        in01 = sum(1 for (x, y, _) in valid if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        norm = (in01 / float(len(valid))) >= 0.7

    def to_px(x: float, y: float) -> Tuple[float, float]:
        if norm:
            return x * CW, y * CH
        return x, y

    max_side = max(CW, CH)
    stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7) if xinsr_stick_scaling else 1

    for idx, (a, b) in enumerate(BODY_EDGES):
        if a >= len(pose) or b >= len(pose):
            continue
        ax, ay, ac = pose[a]
        bx, by, bc = pose[b]
        if not (_valid_pt(ax, ay, ac, conf_thresh) and _valid_pt(bx, by, bc, conf_thresh)):
            continue

        ax, ay = to_px(ax, ay)
        bx, by = to_px(bx, by)
        base = BODY_EDGE_COLORS[idx] if idx < len(BODY_EDGE_COLORS) else (255, 255, 255)

        X = np.array([ay, by], dtype=np.float32)
        Y = np.array([ax, bx], dtype=np.float32)

        mX, mY = float(np.mean(X)), float(np.mean(Y))
        length = float(np.hypot(X[0] - X[1], Y[0] - Y[1]))
        if length < 1.0:
            continue

        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), int(stickwidth * stick_scale)), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, (int(base[0] * 0.6), int(base[1] * 0.6), int(base[2] * 0.6)))

    for j, (x, y, c) in enumerate(pose):
        if not _valid_pt(x, y, c, conf_thresh):
            continue
        x, y = to_px(x, y)
        col = BODY_JOINT_COLORS[j] if j < len(BODY_JOINT_COLORS) else (255, 255, 255)
        cv2.circle(canvas, (int(x), int(y)), 4, col, thickness=-1)


def _draw_hand(canvas: np.ndarray, hand: List[Keypoint], conf_thresh: float) -> None:
    """Draw a 21-point hand onto ``canvas`` in place, one hue per finger bone."""
    if not hand or len(hand) < 21:
        return
    CH, CW = canvas.shape[:2]
    norm = _looks_normalized(hand, conf_thresh)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (x * CW, y * CH) if norm else (x, y)

    n_edges = len(HAND_EDGES)
    for i, (a, b) in enumerate(HAND_EDGES):
        x1, y1, c1 = hand[a]
        x2, y2, c2 = hand[b]
        if _valid_pt(x1, y1, c1, conf_thresh) and _valid_pt(x2, y2, c2, conf_thresh):
            x1, y1 = to_px(x1, y1)
            x2, y2 = to_px(x2, y2)
            cv2.line(
                canvas,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                _hsv_to_bgr(i / float(n_edges), 1.0, 1.0),
                2,
                cv2.LINE_AA,
            )
    for x, y, c in hand:
        if _valid_pt(x, y, c, conf_thresh):
            x, y = to_px(x, y)
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)


def _draw_face(canvas: np.ndarray, face: List[Keypoint], conf_thresh: float) -> None:
    """Draw native DWPose/OpenPose face landmarks as visible white points."""
    if not face:
        return
    CH, CW = canvas.shape[:2]
    norm = _looks_normalized(face, conf_thresh)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (x * CW, y * CH) if norm else (x, y)

    for x, y, c in face:
        if _valid_pt(x, y, c, conf_thresh):
            x, y = to_px(x, y)
            cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_pose_frame_full(
    w: int,
    h: int,
    person: Dict[str, Any],
    conf_thresh_body: float = 0.10,
    conf_thresh_hands: float = 0.10,
    conf_thresh_face: float = 0.10,
) -> np.ndarray:
    """Render one person's body, hands and face onto a fresh ``h x w`` black canvas."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    pose = _reshape_keypoints_2d(person.get("pose_keypoints_2d") or [])
    face = _reshape_keypoints_2d(person.get("face_keypoints_2d") or [])
    hand_l = _reshape_keypoints_2d(person.get("hand_left_keypoints_2d") or [])
    hand_r = _reshape_keypoints_2d(person.get("hand_right_keypoints_2d") or [])

    if pose:
        _draw_body(img, pose, conf_thresh_body)
    if hand_l:
        _draw_hand(img, hand_l, conf_thresh_hands)
    if hand_r:
        _draw_hand(img, hand_r, conf_thresh_hands)
    if face:
        _draw_face(img, face, conf_thresh_face)
    return img


def render_pose_frames(
    people: Sequence[Optional[Dict[str, Any]]],
    w: int,
    h: int,
    *,
    conf_thresh_body: float,
    conf_thresh_hands: float,
    conf_thresh_face: float,
) -> List[np.ndarray]:
    """Render one canvas per entry of ``people``.

    Args:
        people: one person dict per frame; ``None`` produces a blank frame, so
            frames where nobody was detected still occupy a slot and the output stays
            aligned with the input timeline.
        w: canvas width in pixels.
        h: canvas height in pixels.
        conf_thresh_body: minimum confidence to draw a body joint.
        conf_thresh_hands: minimum confidence to draw a hand point.
        conf_thresh_face: minimum confidence to draw a face point.

    Returns:
        A list of ``h x w x 3`` uint8 arrays, the same length as ``people``.
    """
    frames: List[np.ndarray] = []
    for person in people:
        if isinstance(person, dict):
            frames.append(
                _draw_pose_frame_full(
                    w,
                    h,
                    person,
                    conf_thresh_body=conf_thresh_body,
                    conf_thresh_hands=conf_thresh_hands,
                    conf_thresh_face=conf_thresh_face,
                )
            )
        else:
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    return frames
