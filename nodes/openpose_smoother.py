"""ComfyUI node: temporal smoothing for OpenPose pose_data.

A thin adapter. Widget values in, :class:`~ts_utils.config.SmoothConfig` out, and the
work happens in :mod:`ts_utils`.  Imports are relative because the repository
directory name contains a hyphen and cannot be imported by absolute name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..ts_utils import posedata, render
from ..ts_utils.config import DEFAULT_CONFIG, SmoothConfig
from ..ts_utils.pipeline import smooth_KPS_json_obj, smooth_kps_frames  # noqa: F401  (re-exported)

__all__ = ["KPSSmoothPoseDataAndRender", "smooth_KPS_json_obj"]

# The face gate is not exposed as a widget; the pipeline default is used as-is.
CONF_THRESH_FACE = DEFAULT_CONFIG.CONF_GATE_FACE

DEFAULT_CANVAS_W = 720
DEFAULT_CANVAS_H = 1280

_TIP_FORCE_18 = (
    "Truncate the body skeleton to the first 18 joints (COCO), for consumers that reject the 25-joint BODY_25 layout."
)
_TIP_SMOOTH_HANDS = (
    "EXPERIMENTAL: also clean and smooth the hand keypoints. "
    "Off by default; leaving it off reproduces previous output exactly."
)


class KPSSmoothPoseDataAndRender:
    """Smooths a pose_data sequence and renders the smoothed subject to an IMAGE batch.

    The widget defaults below are a contract with saved workflows and must not change;
    :data:`ts_utils.config.DEFAULT_CONFIG` mirrors them field for field.  ``run()``
    reads its ``kwargs`` fallbacks from ``DEFAULT_CONFIG`` rather than repeating the
    literals, which is how the two used to drift apart (``min_run_frames`` 3 vs 2,
    ``conf_thresh_body`` 0.35 vs 0.20, ``conf_thresh_hands`` 0.60 vs 0.50) and give
    direct callers different output from the same node.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "filter_extra_people": ("BOOLEAN", {"default": True}),
                "smooth_alpha": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 0.99, "step": 0.01}),
                "gap_frames": ("INT", {"default": 12, "min": 0, "max": 100, "step": 1}),
                "min_run_frames": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1}),
                "conf_thresh_body": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "conf_thresh_hands": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "force_body_18": ("BOOLEAN", {"default": False, "tooltip": _TIP_FORCE_18}),
                "smooth_hands": ("BOOLEAN", {"default": False, "tooltip": _TIP_SMOOTH_HANDS}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA")
    RETURN_NAMES = ("IMAGE", "pose_data")
    FUNCTION = "run"
    CATEGORY = "TS Utils/Pose"

    def run(self, pose_data: Any, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
        """Smooth ``pose_data`` and return ``(rendered frames, smoothed pose_data)``.

        Args:
            pose_data: a POSEDATA object, a ``{"pose_data": ...}`` wrapper, or a path
                to a pose pickle.
            **kwargs: the widget values; see ``INPUT_TYPES``.

        Returns:
            A ``(B, H, W, 3)`` float tensor in 0..1 alongside the new pose_data.  The
            tensor has ``B == 0`` when the input holds no frames.
        """
        filter_extra_people = bool(kwargs.get("filter_extra_people", DEFAULT_CONFIG.FILTER_EXTRA_PEOPLE))

        smooth_alpha = float(kwargs.get("smooth_alpha", DEFAULT_CONFIG.ALPHA_BODY))
        gap_frames = int(kwargs.get("gap_frames", DEFAULT_CONFIG.MAX_GAP_FRAMES))
        min_run_frames = int(kwargs.get("min_run_frames", DEFAULT_CONFIG.MIN_RUN_FRAMES))

        conf_thresh_body = float(kwargs.get("conf_thresh_body", DEFAULT_CONFIG.CONF_GATE_BODY))
        conf_thresh_hands = float(kwargs.get("conf_thresh_hands", DEFAULT_CONFIG.CONF_GATE_HAND))
        conf_thresh_face = CONF_THRESH_FACE

        force_body_18 = bool(kwargs.get("force_body_18", False))
        smooth_hands = bool(kwargs.get("smooth_hands", False))

        cfg = SmoothConfig(
            CONF_GATE_BODY=conf_thresh_body,
            CONF_GATE_HAND=conf_thresh_hands,
            CONF_GATE_FACE=conf_thresh_face,
            ALPHA_BODY=smooth_alpha,
            SUPER_SMOOTH_ALPHA=smooth_alpha,
            MAX_GAP_FRAMES=gap_frames,
            MIN_RUN_FRAMES=min_run_frames,
            DENSE_SUPER_SMOOTH_ALPHA=smooth_alpha,
            DENSE_MAX_GAP_FRAMES=gap_frames,
            DENSE_MIN_RUN_FRAMES=min_run_frames,
            FILTER_EXTRA_PEOPLE=filter_extra_people,
            HANDS_SMOOTH_ENABLED=smooth_hands,
        )

        pose_data = posedata._coerce_pose_data_to_obj(pose_data)
        frames_json_like, meta_ref = posedata._pose_data_to_kps_frames(pose_data, force_body_18=force_body_18)

        result = smooth_kps_frames(
            frames_json_like,
            keep_face_untouched=False,
            keep_hands_untouched=False,
            filter_extra_people=filter_extra_people,
            cfg=cfg,
        )

        out_pose_data = posedata._kps_frames_to_pose_data(
            pose_data,
            result.frames,
            meta_ref,
            force_body_18=force_body_18,
            subject_indices=result.subject_indices,
        )

        w, h = posedata._extract_canvas_wh(result.frames, default_w=DEFAULT_CANVAS_W, default_h=DEFAULT_CANVAS_H)
        # Draw the person the pipeline actually smoothed. With filter_extra_people
        # off the original people order is kept, so the subject is not always first.
        subjects: List[Optional[Dict[str, Any]]] = [
            posedata.select_subject(fr, idx) for fr, idx in zip(result.frames, result.subject_indices)
        ]
        frames_np = render.render_pose_frames(
            subjects,
            w,
            h,
            conf_thresh_body=conf_thresh_body,
            conf_thresh_hands=conf_thresh_hands,
            conf_thresh_face=conf_thresh_face,
        )

        if not frames_np:
            # np.stack raises on an empty list; hand back a valid empty batch instead.
            return (torch.zeros((0, h, w, 3), dtype=torch.float32), out_pose_data)

        frames_t = torch.from_numpy(np.stack(frames_np, axis=0)).float() / 255.0
        return (frames_t, out_pose_data)
