"""Pose smoothing library backing the TS Pose Data Smoother node.

Layout::

    config     tuning parameters (frozen dataclass), imports nothing
    keypoints  flat-array primitives and skeleton topology
    filters    per-frame and per-sequence repair passes
    smoothing  temporal filters
    tracking   multi-person association and subject selection
    pipeline   the orchestrator
    posedata   pose_data <-> frame-dict conversion  (numpy)
    render     skeleton drawing                     (cv2 + numpy)

Only the first six are re-exported here, and between them they import nothing but
the standard library: importing this package must never pull in numpy, cv2 or torch,
so the smoothing logic stays testable on a headless box with no ComfyUI present.
Import :mod:`ts_utils.posedata` and :mod:`ts_utils.render` explicitly when you need
them.

All intra-package imports are relative (``from .keypoints import ...``) because the
repository directory name contains a hyphen and can therefore never be imported by
absolute name.
"""

from .config import DEFAULT_CONFIG, SmoothConfig
from .pipeline import SmoothResult, smooth_KPS_json_obj, smooth_kps_frames

__all__ = ["SmoothConfig", "DEFAULT_CONFIG", "SmoothResult", "smooth_kps_frames", "smooth_KPS_json_obj"]
