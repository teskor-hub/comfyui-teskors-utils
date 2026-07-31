"""Conversion between ``pose_data`` objects and OpenPose-style frame dicts.

``pose_data`` is the structure passed around by the Aligned-AI / kijai style pose
nodes: an object (or dict) with a ``pose_metas`` list, each entry holding separate
``kps_*`` coordinate arrays and ``kps_*_p`` confidence arrays as numpy.  The
smoothing pipeline instead works on the flat OpenPose JSON layout.  This module
translates in both directions, and loads ``pose_data`` from a pickle when a path is
handed in.

numpy is the only third-party dependency: no cv2, no torch, no ComfyUI imports.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "_load_pose_data_pkl",
    "_coerce_pose_data_to_obj",
    "_as_attr",
    "_set_attr",
    "_xy_p_to_flat",
    "_flat_to_xy_p",
    "_pose_data_to_kps_frames",
    "_kps_frames_to_pose_data",
    "_extract_canvas_wh",
    "select_subject",
]


class _PoseDummyObj:
    """Stand-in for a class that is not importable while unpickling.

    Pose pickles reference classes from whichever custom node pack produced them.
    Rather than requiring that pack to be installed, unknown classes are rebuilt as
    this permissive placeholder, which just absorbs whatever state it is given so the
    attributes we care about (``pose_metas``, ``kps_*``) remain readable.
    """

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[0], dict):
            self.__dict__.update(state[0])
            if isinstance(state[1], dict):
                self.__dict__.update(state[1])
            else:
                self.__dict__["_slotstate"] = state[1]
        else:
            self.__dict__["_state"] = state


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that tolerates missing classes and numpy's private module moves.

    numpy 2 relocated ``numpy.core`` to ``numpy._core``; pickles written by either
    version are remapped so both load.  Anything still unresolvable degrades to
    :class:`_PoseDummyObj` instead of raising.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module.startswith("numpy._globals"):
            module = module.replace("numpy._globals", "numpy", 1)
        if name in {"AAPoseMeta"}:
            return _PoseDummyObj
        try:
            return super().find_class(module, name)
        except Exception:
            return _PoseDummyObj


def _load_pose_data_pkl(path: str) -> Any:
    """Load a pose_data pickle from ``path`` using the tolerant unpickler."""
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def _coerce_pose_data_to_obj(pd: Any) -> Any:
    """Normalise the several shapes a POSEDATA input can arrive in.

    Accepts a filesystem path to a pickle, a ``{"pose_data": ...}`` wrapper dict, or
    the pose_data object itself, and returns the object.
    """
    if isinstance(pd, str):
        return _load_pose_data_pkl(pd)
    if isinstance(pd, dict) and "pose_data" in pd:
        return pd["pose_data"]
    return pd


def _as_attr(x: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from ``x`` whether it is a dict or a plain object."""
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)


def _set_attr(x: Any, key: str, value: Any) -> None:
    """Write ``key`` on ``x`` whether it is a dict or a plain object."""
    if isinstance(x, dict):
        x[key] = value
    else:
        setattr(x, key, value)


def select_subject(frame: Any, index: Optional[int]) -> Optional[Dict[str, Any]]:
    """Return the person dict to treat as the subject of ``frame``.

    ``index`` comes from :attr:`ts_utils.pipeline.SmoothResult.subject_indices`.
    When it is ``None`` or out of range the first person is used, which is what this
    node has always done and is the only sensible choice for frames the pipeline
    never selected a subject in.  Returns ``None`` for frames with no people.
    """
    if not isinstance(frame, dict):
        return None
    people = frame.get("people")
    if not isinstance(people, list) or not people:
        return None
    if index is None or not (0 <= index < len(people)):
        index = 0
    p = people[index]
    return p if isinstance(p, dict) else None


def _xy_p_to_flat(xy: Optional[np.ndarray], p: Optional[np.ndarray]) -> Optional[List[float]]:
    """Interleave an ``(N, 2)`` coordinate array and an ``(N,)`` confidence array.

    Returns the OpenPose flat layout ``[x, y, c, ...]``, or ``None`` if ``xy`` is
    missing or misshapen.  A missing or mismatched confidence array is treated as
    all-ones.
    """
    if xy is None:
        return None
    arr = np.asarray(xy)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    N = arr.shape[0]
    if p is None:
        pp = np.ones((N,), dtype=np.float32)
    else:
        pp = np.asarray(p).reshape(-1)
        if pp.shape[0] != N:
            pp = np.ones((N,), dtype=np.float32)
    out: List[float] = []
    for i in range(N):
        out.extend([float(arr[i, 0]), float(arr[i, 1]), float(pp[i])])
    return out


def _flat_to_xy_p(flat: Optional[List[float]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Split an OpenPose flat array back into ``(N, 2)`` coords and ``(N,)`` confidences."""
    if not isinstance(flat, list) or len(flat) % 3 != 0:
        return None, None
    N = len(flat) // 3
    xy = np.zeros((N, 2), dtype=np.float32)
    p = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        xy[i, 0] = float(flat[3 * i + 0])
        xy[i, 1] = float(flat[3 * i + 1])
        p[i] = float(flat[3 * i + 2])
    return xy, p


def _pose_data_to_kps_frames(pose_data: Any, *, force_body_18: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert pose_data into a list of single-person OpenPose frame dicts.

    Args:
        pose_data: object or dict exposing ``pose_metas`` (or ``frames``).
        force_body_18: truncate body keypoints to the first 18 joints, for
            consumers that reject the 25-joint BODY_25 layout.

    Returns:
        ``(frames, meta_ref)`` where ``meta_ref`` retains the original metas list so
        the result can be written back even if the output object loses it.

    Raises:
        ValueError: if no ``pose_metas`` list can be found.
    """
    pose_metas = _as_attr(pose_data, "pose_metas", None)
    if pose_metas is None:
        pose_metas = _as_attr(pose_data, "frames", None)
    if pose_metas is None or not isinstance(pose_metas, list):
        raise ValueError("pose_data does not contain 'pose_metas' list.")
    frames: List[Dict[str, Any]] = []
    for meta in pose_metas:
        h = _as_attr(meta, "height", 1280)
        w = _as_attr(meta, "width", 720)
        kps_body = _as_attr(meta, "kps_body", None)
        kps_body_p = _as_attr(meta, "kps_body_p", None)
        kps_face = _as_attr(meta, "kps_face", None)
        kps_face_p = _as_attr(meta, "kps_face_p", None)
        kps_lhand = _as_attr(meta, "kps_lhand", None)
        kps_lhand_p = _as_attr(meta, "kps_lhand_p", None)
        kps_rhand = _as_attr(meta, "kps_rhand", None)
        kps_rhand_p = _as_attr(meta, "kps_rhand_p", None)

        pose_flat = _xy_p_to_flat(kps_body, kps_body_p)
        face_flat = _xy_p_to_flat(kps_face, kps_face_p)
        lh_flat = _xy_p_to_flat(kps_lhand, kps_lhand_p)
        rh_flat = _xy_p_to_flat(kps_rhand, kps_rhand_p)

        if force_body_18 and isinstance(pose_flat, list) and len(pose_flat) >= 18 * 3:
            pose_flat = pose_flat[: 18 * 3]

        person = {
            "pose_keypoints_2d": pose_flat if pose_flat is not None else [],
            "face_keypoints_2d": face_flat if face_flat is not None else [],
            "hand_left_keypoints_2d": lh_flat,
            "hand_right_keypoints_2d": rh_flat,
        }
        frame = {"people": [person], "canvas_height": int(h), "canvas_width": int(w)}
        frames.append(frame)

    meta_ref = {"pose_metas": pose_metas, "len": len(pose_metas)}
    return frames, meta_ref


def _kps_frames_to_pose_data(
    pose_data_in: Any,
    frames_kps: List[Dict[str, Any]],
    meta_ref: Dict[str, Any],
    *,
    force_body_18: bool,
    subject_indices: Optional[Sequence[Optional[int]]] = None,
) -> Any:
    """Write smoothed frame dicts back into a deep copy of the input pose_data.

    Args:
        pose_data_in: the original pose_data; never mutated.
        frames_kps: smoothed frames, aligned with ``pose_metas`` by position.
        meta_ref: the ``meta_ref`` returned by :func:`_pose_data_to_kps_frames`.
        force_body_18: truncate body keypoints to the first 18 joints.
        subject_indices: per-frame index of the smoothed person; see
            :func:`select_subject`.  ``None`` means "first person in every frame".

    Returns:
        A new pose_data object carrying the smoothed keypoints.

    Raises:
        ValueError: if the output object has no ``pose_metas`` list.
    """
    out_pd = copy.deepcopy(pose_data_in)
    pose_metas_out = _as_attr(out_pd, "pose_metas", None)
    if pose_metas_out is None:
        pose_metas_out = meta_ref.get("pose_metas")
    if pose_metas_out is None or not isinstance(pose_metas_out, list):
        raise ValueError("Failed to locate pose_metas in output pose_data.")

    T = min(len(pose_metas_out), len(frames_kps))
    for t in range(T):
        meta = pose_metas_out[t]
        fr = frames_kps[t]
        idx = subject_indices[t] if subject_indices is not None and t < len(subject_indices) else None
        p0 = select_subject(fr, idx)
        if p0 is None:
            continue

        pose_flat = p0.get("pose_keypoints_2d")
        face_flat = p0.get("face_keypoints_2d")
        lh_flat = p0.get("hand_left_keypoints_2d")
        rh_flat = p0.get("hand_right_keypoints_2d")

        if force_body_18 and isinstance(pose_flat, list) and len(pose_flat) >= 18 * 3:
            pose_flat = pose_flat[: 18 * 3]

        body_xy, body_p = _flat_to_xy_p(pose_flat if isinstance(pose_flat, list) else None)
        face_xy, face_p = _flat_to_xy_p(face_flat if isinstance(face_flat, list) else None)
        lh_xy, lh_p = _flat_to_xy_p(lh_flat if isinstance(lh_flat, list) else None)
        rh_xy, rh_p = _flat_to_xy_p(rh_flat if isinstance(rh_flat, list) else None)

        if body_xy is not None and body_p is not None:
            _set_attr(meta, "kps_body", body_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_body_p", body_p.astype(np.float32, copy=False))
        if face_xy is not None and face_p is not None:
            _set_attr(meta, "kps_face", face_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_face_p", face_p.astype(np.float32, copy=False))
        if lh_xy is not None and lh_p is not None:
            _set_attr(meta, "kps_lhand", lh_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_lhand_p", lh_p.astype(np.float32, copy=False))
        if rh_xy is not None and rh_p is not None:
            _set_attr(meta, "kps_rhand", rh_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_rhand_p", rh_p.astype(np.float32, copy=False))

        if isinstance(fr, dict):
            if "canvas_width" in fr:
                _set_attr(meta, "width", int(fr["canvas_width"]))
            if "canvas_height" in fr:
                _set_attr(meta, "height", int(fr["canvas_height"]))

    _set_attr(out_pd, "pose_metas", pose_metas_out)
    return out_pd


def _extract_canvas_wh(data: Any, default_w: int, default_h: int) -> Tuple[int, int]:
    """Return the canvas size declared by the first frame that declares one."""
    w, h = int(default_w), int(default_h)
    if isinstance(data, list):
        for fr in data:
            if isinstance(fr, dict) and "canvas_width" in fr and "canvas_height" in fr:
                try:
                    w = int(fr["canvas_width"])
                    h = int(fr["canvas_height"])
                    break
                except Exception:
                    pass
    return w, h
