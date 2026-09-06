"""ComfyUI node: temporal smoothing for OpenPose pose_data.

A thin adapter. Widget values in, :class:`~ts_utils.config.SmoothConfig` out, and the
work happens in :mod:`ts_utils`.  Imports are relative because the repository
directory name contains a hyphen and cannot be imported by absolute name.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..ts_utils import posedata, render
from ..ts_utils.config import DEFAULT_CONFIG, SmoothConfig
from ..ts_utils.pipeline import smooth_KPS_json_obj, smooth_kps_frames  # noqa: F401  (re-exported)

__all__ = ["KPSSmoothPoseDataAndRender", "KPSSmoothPoseKeypointAndRender", "smooth_KPS_json_obj"]

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


def _build_smooth_config(kwargs: Dict[str, Any]) -> Tuple[SmoothConfig, float, float, bool, bool]:
    """Build one config for both POSEDATA and POSE_KEYPOINT adapters."""
    filter_extra_people = bool(kwargs.get("filter_extra_people", DEFAULT_CONFIG.FILTER_EXTRA_PEOPLE))
    smooth_alpha = float(kwargs.get("smooth_alpha", DEFAULT_CONFIG.ALPHA_BODY))
    gap_frames = int(kwargs.get("gap_frames", DEFAULT_CONFIG.MAX_GAP_FRAMES))
    min_run_frames = int(kwargs.get("min_run_frames", DEFAULT_CONFIG.MIN_RUN_FRAMES))
    conf_thresh_body = float(kwargs.get("conf_thresh_body", DEFAULT_CONFIG.CONF_GATE_BODY))
    conf_thresh_hands = float(kwargs.get("conf_thresh_hands", DEFAULT_CONFIG.CONF_GATE_HAND))
    smooth_hands = bool(kwargs.get("smooth_hands", False))

    cfg = SmoothConfig(
        CONF_GATE_BODY=conf_thresh_body,
        CONF_GATE_HAND=conf_thresh_hands,
        CONF_GATE_FACE=CONF_THRESH_FACE,
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
    return cfg, conf_thresh_body, conf_thresh_hands, filter_extra_people, smooth_hands


def _render_result(
    result: Any,
    *,
    conf_thresh_body: float,
    conf_thresh_hands: float,
    render_resolution: int = 0,
) -> torch.Tensor:
    """Render the tracked subject from a smoothing result to a ComfyUI IMAGE batch."""
    w, h = posedata._extract_canvas_wh(result.frames, default_w=DEFAULT_CANVAS_W, default_h=DEFAULT_CANVAS_H)
    subjects: List[Optional[Dict[str, Any]]] = [
        posedata.select_subject(fr, idx) for fr, idx in zip(result.frames, result.subject_indices)
    ]
    render_w, render_h = w, h
    if render_resolution > 0 and min(w, h) > 0:
        scale = render_resolution / float(min(w, h))
        render_w = max(1, int(round(w * scale)))
        render_h = max(1, int(round(h * scale)))
        subjects = [_scale_subject_for_render(subject, w, h, render_w, render_h) for subject in subjects]
    frames_np = render.render_pose_frames(
        subjects,
        render_w,
        render_h,
        conf_thresh_body=conf_thresh_body,
        conf_thresh_hands=conf_thresh_hands,
        conf_thresh_face=CONF_THRESH_FACE,
    )
    if not frames_np:
        return torch.zeros((0, render_h, render_w, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(frames_np, axis=0)).float() / 255.0


def _scale_subject_for_render(
    subject: Optional[Dict[str, Any]],
    source_w: int,
    source_h: int,
    render_w: int,
    render_h: int,
) -> Optional[Dict[str, Any]]:
    """Scale pixel-space OpenPose coordinates without changing normalized inputs."""
    if not isinstance(subject, dict):
        return None
    scaled = copy.deepcopy(subject)
    sx = render_w / float(source_w)
    sy = render_h / float(source_h)
    for key in (
        "pose_keypoints_2d",
        "face_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
    ):
        values = scaled.get(key)
        if not isinstance(values, list) or len(values) % 3 != 0:
            continue
        visible = [values[i : i + 3] for i in range(0, len(values), 3) if values[i + 2] > 0]
        is_normalized = bool(visible) and (
            sum(1 for x, y, _ in visible if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) / len(visible) >= 0.7
        )
        if is_normalized:
            continue
        for i in range(0, len(values), 3):
            values[i] *= sx
            values[i + 1] *= sy
    return scaled


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
        cfg, conf_thresh_body, conf_thresh_hands, filter_extra_people, smooth_hands = _build_smooth_config(kwargs)
        force_body_18 = bool(kwargs.get("force_body_18", False))

        pose_data = posedata._coerce_pose_data_to_obj(pose_data)
        frames_json_like, meta_ref = posedata._pose_data_to_kps_frames(pose_data, force_body_18=force_body_18)

        result = smooth_kps_frames(
            frames_json_like,
            keep_face_untouched=False,
            keep_hands_untouched=not smooth_hands,
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

        frames_t = _render_result(
            result,
            conf_thresh_body=conf_thresh_body,
            conf_thresh_hands=conf_thresh_hands,
        )
        return (frames_t, out_pose_data)


class KPSSmoothPoseKeypointAndRender:
    """Smooth standard comfyui_controlnet_aux POSE_KEYPOINT data from DWPose/OpenPose."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "filter_extra_people": ("BOOLEAN", {"default": True}),
                "smooth_alpha": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 0.99, "step": 0.01}),
                "gap_frames": ("INT", {"default": 12, "min": 0, "max": 100, "step": 1}),
                "min_run_frames": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1}),
                "conf_thresh_body": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "conf_thresh_hands": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
                "render_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
            },
            "optional": {
                "smooth_hands": ("BOOLEAN", {"default": False, "tooltip": _TIP_SMOOTH_HANDS}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("IMAGE", "pose_keypoints")
    FUNCTION = "run"
    CATEGORY = "TS Utils/Pose"

    def run(self, pose_keypoints: Any, **kwargs: Any) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Return a rendered pose batch and smoothed OpenPose-format keypoint dictionaries."""
        if not isinstance(pose_keypoints, list):
            raise ValueError("POSE_KEYPOINT input must be a list of OpenPose frame dictionaries.")

        cfg, conf_thresh_body, conf_thresh_hands, filter_extra_people, smooth_hands = _build_smooth_config(kwargs)
        result = smooth_kps_frames(
            pose_keypoints,
            keep_face_untouched=True,
            keep_hands_untouched=not smooth_hands,
            preserve_untouched_dense=True,
            filter_extra_people=filter_extra_people,
            cfg=cfg,
        )
        frames_t = _render_result(
            result,
            conf_thresh_body=conf_thresh_body,
            conf_thresh_hands=conf_thresh_hands,
            render_resolution=int(kwargs.get("render_resolution", 768)),
        )
        return frames_t, result.frames
