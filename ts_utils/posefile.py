"""Pickle-free storage for POSEDATA.

POSEDATA is a dict of two parallel per-frame sequences plus two optional
extras::

    {
      "pose_metas":          [AAPoseMeta, ...],   # objects with ndarray fields
      "pose_metas_original": [{...}, ...],        # plain dicts of ndarrays
      "retarget_image":      None,
      "refer_pose_meta":     None,
    }

Everything in there is an int, a str, ``None`` or a float32 ndarray, so it maps
cleanly onto ``numpy.savez_compressed``.

Why not pickle
--------------
Unpickling executes whatever the file names, and the load node reads files out
of ComfyUI's ``input`` folder — i.e. untrusted input. On top of that, the
pickles produced by the old node embedded the *absolute install path* of
ComfyUI-WanAnimatePreprocess as a module name, so they silently stopped loading
if the pack was moved or reinstalled. The ``.npz`` files written here contain no
code references at all and are portable between machines.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["save_posedata", "load_posedata", "POSE_FILE_EXT"]

POSE_FILE_EXT = ".npz"

#: Attributes carried by one AAPoseMeta entry. Array fields are stacked across
#: frames; scalars are stored as one array per field.
_META_ARRAYS = (
    "kps_body", "kps_lhand", "kps_rhand", "kps_face",
    "kps_body_p", "kps_lhand_p", "kps_rhand_p", "kps_face_p",
)
_META_SCALARS = ("image_id", "height", "width")

_FORMAT_VERSION = 1


def _stack(frames: List[Any], getter) -> Dict[str, np.ndarray]:
    """Stack one field across frames, recording which frames actually had it.

    Frames where the field is missing get a zero-filled slot and a False mask
    entry, so a partially-detected sequence round-trips exactly rather than
    collapsing to a shorter list.
    """
    values = [getter(f) for f in frames]
    present = np.array([v is not None for v in values], dtype=bool)
    real = [np.asarray(v) for v in values if v is not None]
    if not real:
        return {"mask": present, "data": np.zeros((len(values), 0), dtype=np.float32)}

    shape = real[0].shape
    if any(r.shape != shape for r in real):
        raise ValueError("ragged keypoint shapes are not supported")

    out = np.zeros((len(values),) + shape, dtype=real[0].dtype)
    idx = 0
    for i, v in enumerate(values):
        if v is not None:
            out[i] = real[idx]
            idx += 1
    return {"mask": present, "data": out}


def save_posedata(pose_data: Dict[str, Any], path: str) -> None:
    """Write POSEDATA to ``path`` as a compressed npz archive."""
    if not isinstance(pose_data, dict):
        raise TypeError(f"POSEDATA must be a dict, got {type(pose_data).__name__}")

    metas = pose_data.get("pose_metas") or []
    originals = pose_data.get("pose_metas_original") or []

    blob: Dict[str, np.ndarray] = {
        "__version__": np.array([_FORMAT_VERSION]),
        "n_metas": np.array([len(metas)]),
        "n_originals": np.array([len(originals)]),
    }

    for field in _META_ARRAYS:
        packed = _stack(metas, lambda m, f=field: getattr(m, f, None))
        blob[f"meta.{field}.mask"] = packed["mask"]
        blob[f"meta.{field}.data"] = packed["data"]

    for field in _META_SCALARS:
        blob[f"meta.{field}"] = np.array([getattr(m, field, None) for m in metas], dtype=object).astype(str)

    if originals:
        keys = sorted({k for d in originals if isinstance(d, dict) for k in d})
        blob["orig.keys"] = np.array(keys, dtype=str)
        for k in keys:
            packed = _stack(originals, lambda d, kk=k: d.get(kk) if isinstance(d, dict) else None)
            if packed["data"].dtype.kind in "fiub":
                blob[f"orig.{k}.mask"] = packed["mask"]
                blob[f"orig.{k}.data"] = packed["data"]
            else:
                blob[f"orig.{k}.scalar"] = np.array(
                    [d.get(k) if isinstance(d, dict) else None for d in originals], dtype=object
                ).astype(str)
    else:
        blob["orig.keys"] = np.array([], dtype=str)

    np.savez_compressed(path, **blob)


def _meta_class():
    """Locate the real AAPoseMeta class, or ``None`` if the pack is absent.

    ComfyUI registers custom node packages under path-derived module names, so
    the class cannot be imported by a fixed dotted path. Scanning the already
    imported modules finds it without importing anything new — and without ever
    letting a file choose what gets imported, which is the whole problem with
    pickle.
    """
    for mod in list(sys.modules.values()):
        cls = getattr(mod, "AAPoseMeta", None)
        if isinstance(cls, type):
            return cls
    return None


class _PoseMetaShim:
    """Stand-in used when ComfyUI-WanAnimatePreprocess is not installed."""

    def __init__(self) -> None:
        self.image_id = ""
        self.height = 0
        self.width = 0
        for f in _META_ARRAYS:
            setattr(self, f, None)


def _new_meta():
    cls = _meta_class()
    if cls is None:
        return _PoseMetaShim()
    try:
        return cls()
    except Exception:
        return _PoseMetaShim()


def load_posedata(path: str) -> Dict[str, Any]:
    """Read POSEDATA back from an npz archive written by :func:`save_posedata`."""
    with np.load(path, allow_pickle=False) as z:
        n_metas = int(z["n_metas"][0])
        n_orig = int(z["n_originals"][0])

        metas = []
        for i in range(n_metas):
            m = _new_meta()
            for field in _META_ARRAYS:
                mask = z[f"meta.{field}.mask"]
                data = z[f"meta.{field}.data"]
                setattr(m, field, data[i] if (i < len(mask) and mask[i] and data.size) else None)
            ids = z["meta.image_id"]
            m.image_id = str(ids[i]) if i < len(ids) else ""
            for field in ("height", "width"):
                vals = z[f"meta.{field}"]
                try:
                    setattr(m, field, int(float(vals[i])))
                except (ValueError, IndexError):
                    setattr(m, field, 0)
            metas.append(m)

        originals: List[Dict[str, Any]] = [{} for _ in range(n_orig)]
        for k in [str(x) for x in z["orig.keys"]]:
            if f"orig.{k}.data" in z:
                mask, data = z[f"orig.{k}.mask"], z[f"orig.{k}.data"]
                for i in range(n_orig):
                    if i < len(mask) and mask[i] and data.size:
                        originals[i][k] = data[i]
            elif f"orig.{k}.scalar" in z:
                vals = z[f"orig.{k}.scalar"]
                for i in range(n_orig):
                    if i >= len(vals):
                        continue
                    raw = str(vals[i])
                    try:
                        originals[i][k] = int(float(raw))
                    except ValueError:
                        originals[i][k] = raw

    return {
        "pose_metas": metas,
        "pose_metas_original": originals,
        "retarget_image": None,
        "refer_pose_meta": None,
    }
