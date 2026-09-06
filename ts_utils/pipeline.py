"""End-to-end pose smoothing: pick the subject, repair, then smooth.

This is the orchestrator and the only module that reads :class:`SmoothConfig`
broadly.  It imports nothing that needs numpy, cv2 or torch, so the entire smoothing
algorithm can be exercised from a plain Python test.

Stage order matters and is load-bearing:

1. build tracks over the whole video and choose the subject per frame
2. per-frame spatial outlier suppression
3. gap fill / short-run rejection
4. torso appearance sync
5. isolated-joint removal
6. 3-tap median
7. zero-lag EMA
8. root-scale carry, then forced torso pair
9. de-flicker again (``max_gap=0``)
10. hand block, then face block (both anchored to the *repaired* body sequence)
11. per-frame causal EMA, wrist pinning, elbow IK, people assembly
12. a final presence pass written back into the output frames

In particular the dense (face/hand) block consumes the body sequence after all
sequence filters but before the per-frame EMA; moving it would anchor face and hands
to different body coordinates than they are today.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import SmoothConfig
from .filters import (
    _carry_pose_when_torso_missing,
    _denoise_and_fill_gaps_pose_seq,
    _fix_elbow_using_wrist,
    _force_full_torso_pair,
    _median3_pose_seq,
    _pin_body_wrist_to_hand,
    _remove_short_presence_runs_kps_seq,
    _suppress_isolated_joints_in_pose_arr,
    _suppress_spatial_outliers_in_hand_arr,
    _suppress_spatial_outliers_in_pose_arr,
    _sync_group_appearances,
    _zero_sparse_frames_kps_seq,
)
from .keypoints import _body_center_from_pose
from .smoothing import BodyState, _smooth_body_pose, _smooth_dense_seq_anchored_to_body, _zero_lag_ema_pose_seq
from .tracking import _build_tracks_over_video, _choose_single_person, _pick_main_track

__all__ = ["SmoothResult", "smooth_kps_frames", "smooth_KPS_json_obj"]

FlatArr = Optional[List[float]]


@dataclass(frozen=True)
class SmoothResult:
    """Output of :func:`smooth_kps_frames`.

    Attributes:
        frames: the smoothed frame dicts, one per input frame, in input order.
        subject_indices: for each frame, the position of the smoothed subject inside
            ``frames[i]["people"]``, or ``None`` if no subject was found in that
            frame.  With ``filter_extra_people`` disabled the original people order
            is preserved, so the subject is *not* necessarily at index 0 - anything
            that wants to read or draw "the person we smoothed" must use this.
    """

    frames: List[Any]
    subject_indices: List[Optional[int]]


def _select_subject_per_frame(data: List[Any], *, cfg: SmoothConfig) -> List[Optional[Dict[str, Any]]]:
    """Choose the subject in every frame, returning the caller's own person dicts.

    Prefers whole-video tracking (``cfg.MAIN_PERSON_MODE == "longest_track"``), which
    survives a frame or two of missed detections.  If tracking yields nothing, falls
    back to greedy per-frame scoring seeded by the previous frame's centre.

    The returned dicts are references into ``data``; callers rely on ``is`` identity
    to locate the subject again inside the untouched input frame.
    """
    chosen_people: List[Optional[Dict[str, Any]]] = [None] * len(data)

    # DWPose can briefly emit a duplicate detection even in a genuinely
    # single-person clip.  Whole-video tracking may split a fast or close-up
    # subject into several short tracks when the body centre moves farther than
    # its match radius; choosing only the longest fragment then turns every
    # other frame completely black.  When multi-person frames are only a tiny
    # minority, keep the sole detection in every frame and use the normal
    # per-frame scorer only for those few ambiguous frames.
    populated: List[Tuple[int, List[Dict[str, Any]]]] = []
    ambiguous_frames = 0
    for index, frame in enumerate(data):
        if not isinstance(frame, dict):
            continue
        people = [person for person in (frame.get("people") or []) if isinstance(person, dict)]
        if not people:
            continue
        populated.append((index, people))
        if len(people) > 1:
            ambiguous_frames += 1
    ambiguity_limit = max(2, len(populated) // 50)
    if populated and ambiguous_frames <= ambiguity_limit:
        prev_center: Optional[Tuple[float, float]] = None
        for index, people in populated:
            chosen = people[0] if len(people) == 1 else _choose_single_person(people, prev_center, cfg=cfg)
            chosen_people[index] = chosen
            if chosen:
                center = _body_center_from_pose(chosen.get("pose_keypoints_2d"))
                if center:
                    prev_center = center
        return chosen_people

    main_tr = None
    if cfg.MAIN_PERSON_MODE == "longest_track":
        # Canvas size lets the subject scorer prefer whoever is framed centrally;
        # without it that signal is simply skipped rather than guessed.
        canvas: Optional[Tuple[float, float]] = None
        for fr in data:
            if isinstance(fr, dict) and fr.get("canvas_width") and fr.get("canvas_height"):
                try:
                    canvas = (float(fr["canvas_width"]), float(fr["canvas_height"]))
                except (TypeError, ValueError):
                    canvas = None
                break
        main_tr = _pick_main_track(
            _build_tracks_over_video(data, cfg=cfg),
            cfg=cfg,
            n_frames=len(data),
            canvas=canvas,
        )
    if main_tr:
        for t in range(len(data)):
            if t in main_tr.frames:
                chosen_people[t] = main_tr.frames[t]
        return chosen_people

    prev_center: Optional[Tuple[float, float]] = None
    for i, frame in enumerate(data):
        if not isinstance(frame, dict) or not frame.get("people"):
            continue
        chosen_people[i] = _choose_single_person(frame.get("people", []), prev_center, cfg=cfg)
        if chosen_people[i]:
            c = _body_center_from_pose(chosen_people[i].get("pose_keypoints_2d"))
            if c:
                prev_center = c
    return chosen_people


def smooth_kps_frames(
    data: Any,
    *,
    keep_face_untouched: bool = True,
    keep_hands_untouched: bool = True,
    preserve_untouched_dense: bool = False,
    filter_extra_people: Optional[bool] = None,
    cfg: Optional[SmoothConfig] = None,
) -> SmoothResult:
    """Smooth a list of OpenPose-style frame dicts and report where the subject ended up.

    Args:
        data: list of frames, each ``{"people": [...], ...}``.  Frames that are not
            dicts are passed through untouched.
        keep_face_untouched: skip the face block entirely.
        keep_hands_untouched: skip the hand block entirely.  Note the hand block is
            additionally gated by ``cfg.HANDS_SMOOTH_ENABLED``.
        preserve_untouched_dense: also skip the final face/hand presence cleanup
            for dense keypoint groups whose smoothing is disabled.
        filter_extra_people: drop everyone except the subject.  ``None`` means "use
            ``cfg.FILTER_EXTRA_PEOPLE``".
        cfg: tuning parameters; defaults to a fresh :class:`SmoothConfig`.

    Returns:
        A :class:`SmoothResult` holding the new frames and the per-frame index of
        the smoothed subject.

    Raises:
        ValueError: if ``data`` is not a list.
    """
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of frames.")
    cfg = cfg or SmoothConfig()
    filter_extra_people = bool(cfg.FILTER_EXTRA_PEOPLE) if filter_extra_people is None else filter_extra_people

    chosen_people = _select_subject_per_frame(data, cfg=cfg)
    pose_seq: List[FlatArr] = [p.get("pose_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]

    # --- Sequence-level repair of the body skeleton. -----------------------
    if cfg.SPATIAL_OUTLIER_FIX:
        pose_seq = [
            (
                _suppress_spatial_outliers_in_pose_arr(
                    arr,
                    conf_gate=cfg.CONF_GATE_BODY,
                    torso_radius_factor=cfg.TORSO_RADIUS_FACTOR,
                    bone_max_factor=cfg.BONE_MAX_FACTOR,
                )
                if arr
                else None
            )
            for arr in pose_seq
        ]

    if cfg.GAP_FILL_ENABLED:
        pose_seq = _denoise_and_fill_gaps_pose_seq(
            pose_seq, conf_gate=cfg.CONF_GATE_BODY, min_run=cfg.MIN_RUN_FRAMES, max_gap=cfg.MAX_GAP_FRAMES
        )

    if cfg.TORSO_SYNC_ENABLED:
        pose_seq = _sync_group_appearances(
            pose_seq, group=cfg.TORSO_JOINTS, conf_gate=cfg.CONF_GATE_BODY, lookahead=cfg.TORSO_LOOKAHEAD_FRAMES
        )

    pose_seq = [
        _suppress_isolated_joints_in_pose_arr(arr, conf_gate=cfg.CONF_GATE_BODY, keep=cfg.TORSO_JOINTS) if arr else None
        for arr in pose_seq
    ]

    # --- Sequence-level smoothing of the body skeleton. --------------------
    if cfg.MEDIAN3_ENABLED:
        pose_seq = _median3_pose_seq(pose_seq, conf_gate=cfg.CONF_GATE_BODY)
    if cfg.SUPER_SMOOTH_ENABLED:
        pose_seq = _zero_lag_ema_pose_seq(pose_seq, alpha=cfg.SUPER_SMOOTH_ALPHA, conf_gate=cfg.SUPER_SMOOTH_MIN_CONF)
    if cfg.ROOTSCALE_CARRY_ENABLED:
        pose_seq = _carry_pose_when_torso_missing(
            pose_seq,
            conf_gate=cfg.CARRY_CONF_GATE,
            max_carry=cfg.CARRY_MAX_FRAMES,
            anchor_joints=cfg.CARRY_ANCHOR_JOINTS,
            min_anchors=cfg.CARRY_MIN_ANCHORS,
            allow_disappear_joints=cfg.ALLOW_DISAPPEAR_JOINTS,
        )
    pose_seq = _force_full_torso_pair(
        pose_seq,
        conf_gate=cfg.CARRY_CONF_GATE,
        anchor_joints=cfg.CARRY_ANCHOR_JOINTS,
        min_anchors=cfg.CARRY_MIN_ANCHORS,
        max_lookback=240,
        fill_legs_with_hip=True,
        always_fill_if_one_hip=True,
    )
    pose_seq = _denoise_and_fill_gaps_pose_seq(
        pose_seq, conf_gate=cfg.CONF_GATE_BODY, min_run=cfg.MIN_RUN_FRAMES, max_gap=0
    )

    # --- Dense keypoints, anchored to the repaired body sequence. ----------
    face_seq: List[FlatArr] = [p.get("face_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]
    lh_seq: List[FlatArr] = [p.get("hand_left_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]
    rh_seq: List[FlatArr] = [p.get("hand_right_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]

    smooth_hands = cfg.HANDS_SMOOTH_ENABLED and not keep_hands_untouched
    smooth_face = cfg.FACE_SMOOTH_ENABLED and not keep_face_untouched

    if smooth_hands:
        lh_seq = [
            _suppress_spatial_outliers_in_hand_arr(a, conf_gate=cfg.CONF_GATE_HAND) if a else None for a in lh_seq
        ]
        rh_seq = [
            _suppress_spatial_outliers_in_hand_arr(a, conf_gate=cfg.CONF_GATE_HAND) if a else None for a in rh_seq
        ]
        lh_seq = _remove_short_presence_runs_kps_seq(
            lh_seq,
            conf_gate=cfg.CONF_GATE_HAND,
            min_points_present=cfg.HAND_MIN_POINTS_PRESENT,
            min_run=cfg.MIN_HAND_RUN_FRAMES,
        )
        rh_seq = _remove_short_presence_runs_kps_seq(
            rh_seq,
            conf_gate=cfg.CONF_GATE_HAND,
            min_points_present=cfg.HAND_MIN_POINTS_PRESENT,
            min_run=cfg.MIN_HAND_RUN_FRAMES,
        )
        lh_seq = _zero_sparse_frames_kps_seq(
            lh_seq, conf_gate=cfg.CONF_GATE_HAND, min_points_present=cfg.HAND_MIN_POINTS_PRESENT
        )
        rh_seq = _zero_sparse_frames_kps_seq(
            rh_seq, conf_gate=cfg.CONF_GATE_HAND, min_points_present=cfg.HAND_MIN_POINTS_PRESENT
        )
        if cfg.DENSE_GAP_FILL_ENABLED:
            lh_seq = _denoise_and_fill_gaps_pose_seq(
                lh_seq, conf_gate=cfg.CONF_GATE_HAND, min_run=cfg.DENSE_MIN_RUN_FRAMES, max_gap=cfg.DENSE_MAX_GAP_FRAMES
            )
            rh_seq = _denoise_and_fill_gaps_pose_seq(
                rh_seq, conf_gate=cfg.CONF_GATE_HAND, min_run=cfg.DENSE_MIN_RUN_FRAMES, max_gap=cfg.DENSE_MAX_GAP_FRAMES
            )
        lh_seq = _smooth_dense_seq_anchored_to_body(
            lh_seq,
            pose_seq,
            kind="hand_left",
            conf_gate_dense=cfg.CONF_GATE_HAND,
            conf_gate_body=cfg.CONF_GATE_BODY,
            median3=cfg.DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=cfg.DENSE_SUPER_SMOOTH_ALPHA,
        )
        rh_seq = _smooth_dense_seq_anchored_to_body(
            rh_seq,
            pose_seq,
            kind="hand_right",
            conf_gate_dense=cfg.CONF_GATE_HAND,
            conf_gate_body=cfg.CONF_GATE_BODY,
            median3=cfg.DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=cfg.DENSE_SUPER_SMOOTH_ALPHA,
        )

    if smooth_face:
        if cfg.DENSE_GAP_FILL_ENABLED:
            face_seq = _denoise_and_fill_gaps_pose_seq(
                face_seq,
                conf_gate=cfg.CONF_GATE_FACE,
                min_run=cfg.DENSE_MIN_RUN_FRAMES,
                max_gap=cfg.DENSE_MAX_GAP_FRAMES,
            )
        face_seq = _smooth_dense_seq_anchored_to_body(
            face_seq,
            pose_seq,
            kind="face",
            conf_gate_dense=cfg.CONF_GATE_FACE,
            conf_gate_body=cfg.CONF_GATE_BODY,
            median3=cfg.DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=cfg.DENSE_SUPER_SMOOTH_ALPHA,
        )

    # --- Per-frame pass: causal EMA, hand/arm reconciliation, assembly. ----
    out_frames: List[Any] = []
    subject_indices: List[Optional[int]] = [None] * len(data)
    body_state: Optional[BodyState] = None

    for i, frame in enumerate(data):
        if not isinstance(frame, dict):
            out_frames.append(frame)
            continue

        frame_out = copy.deepcopy(frame)
        chosen = chosen_people[i]

        if chosen is None:
            if filter_extra_people:
                frame_out["people"] = []
            out_frames.append(frame_out)
            body_state = None
            continue

        p_out = copy.deepcopy(chosen)
        # Aliases pose_seq[i] on purpose, and is only safe because _smooth_body_pose
        # returns a FRESH list two lines down. The in-place fixups that follow
        # (_pin_body_wrist_to_hand / _fix_elbow_using_wrist) would otherwise write
        # back into pose_seq and corrupt later frames.
        p_out["pose_keypoints_2d"] = pose_seq[i]

        pose_arr = p_out.get("pose_keypoints_2d")
        joints = (len(pose_arr) // 3) if isinstance(pose_arr, list) else 0
        if body_state is None:
            body_state = BodyState(joints if joints > 0 else 18)

        p_out["pose_keypoints_2d"] = _smooth_body_pose(p_out.get("pose_keypoints_2d"), body_state, cfg=cfg)

        if smooth_face:
            p_out["face_keypoints_2d"] = face_seq[i]
        else:
            p_out["face_keypoints_2d"] = chosen.get("face_keypoints_2d", p_out.get("face_keypoints_2d"))

        if smooth_hands:
            p_out["hand_left_keypoints_2d"], p_out["hand_right_keypoints_2d"] = lh_seq[i], rh_seq[i]
        else:
            p_out["hand_left_keypoints_2d"] = chosen.get("hand_left_keypoints_2d", p_out.get("hand_left_keypoints_2d"))
            p_out["hand_right_keypoints_2d"] = chosen.get(
                "hand_right_keypoints_2d", p_out.get("hand_right_keypoints_2d")
            )

        _pin_body_wrist_to_hand(
            p_out, side="left", conf_gate_body=cfg.CONF_GATE_BODY, conf_gate_hand=cfg.CONF_GATE_HAND, blend=1.0
        )
        _pin_body_wrist_to_hand(
            p_out, side="right", conf_gate_body=cfg.CONF_GATE_BODY, conf_gate_hand=cfg.CONF_GATE_HAND, blend=1.0
        )
        _fix_elbow_using_wrist(p_out, side="left", conf_gate=cfg.CONF_GATE_BODY)
        _fix_elbow_using_wrist(p_out, side="right", conf_gate=cfg.CONF_GATE_BODY)

        if filter_extra_people:
            frame_out["people"] = [p_out]
            subject_idx = 0
        else:
            orig_people = frame.get("people", [])
            if not isinstance(orig_people, list):
                frame_out["people"] = [p_out]
                subject_idx = 0
            else:
                # Keep the original people order; find the subject by identity.
                replaced, subject_idx, new_people = False, 0, []
                for op in orig_people:
                    if not replaced and (op is chosen):
                        subject_idx = len(new_people)
                        new_people.append(p_out)
                        replaced = True
                    else:
                        new_people.append(copy.deepcopy(op))
                if not replaced:
                    new_people = [p_out] + [copy.deepcopy(op) for op in orig_people]
                    subject_idx = 0
                frame_out["people"] = new_people

        subject_indices[i] = subject_idx
        out_frames.append(frame_out)

    _apply_final_presence_pass(
        out_frames,
        subject_indices,
        cfg=cfg,
        clean_face=smooth_face or not preserve_untouched_dense,
        clean_hands=smooth_hands or not preserve_untouched_dense,
    )
    return SmoothResult(frames=out_frames, subject_indices=subject_indices)


def _apply_final_presence_pass(
    out_frames: List[Any],
    subject_indices: List[Optional[int]],
    *,
    cfg: SmoothConfig,
    clean_face: bool,
    clean_hands: bool,
) -> None:
    """Run a last de-flicker over the assembled subject and write it back, in place.

    The per-frame passes can reintroduce one- or two-frame appearances (a wrist
    pinned to a hand that itself only existed for a frame, say), so the body gets
    one more short-run rejection here. Hands and face receive it when requested
    by the caller.

    Reads and writes the subject at ``subject_indices[i]``, not blindly
    ``people[0]``: when extra people are kept the subject can sit at any index, and
    the old index-0 assumption applied this pass to an unrelated, unsmoothed person
    while the real subject never received it at all.
    """
    final_body: List[FlatArr] = []
    final_lh: List[FlatArr] = []
    final_rh: List[FlatArr] = []
    final_face: List[FlatArr] = []

    for i, f in enumerate(out_frames):
        p = _subject_of(f, subject_indices[i])
        final_body.append(p.get("pose_keypoints_2d") if p is not None else None)
        final_lh.append(p.get("hand_left_keypoints_2d") if p is not None else None)
        final_rh.append(p.get("hand_right_keypoints_2d") if p is not None else None)
        final_face.append(p.get("face_keypoints_2d") if p is not None else None)

    eff_min = max(2, cfg.MIN_RUN_FRAMES)

    final_body = _denoise_and_fill_gaps_pose_seq(final_body, conf_gate=cfg.CONF_GATE_BODY, min_run=eff_min, max_gap=0)
    if clean_hands:
        final_lh = _remove_short_presence_runs_kps_seq(
            final_lh, conf_gate=cfg.CONF_GATE_HAND, min_points_present=1, min_run=eff_min
        )
        final_rh = _remove_short_presence_runs_kps_seq(
            final_rh, conf_gate=cfg.CONF_GATE_HAND, min_points_present=1, min_run=eff_min
        )
    if clean_face:
        final_face = _remove_short_presence_runs_kps_seq(
            final_face, conf_gate=cfg.CONF_GATE_FACE, min_points_present=1, min_run=eff_min
        )

    for i, f in enumerate(out_frames):
        p = _subject_of(f, subject_indices[i])
        if p is None:
            continue
        p["pose_keypoints_2d"] = final_body[i]
        if clean_hands:
            p["hand_left_keypoints_2d"] = final_lh[i]
            p["hand_right_keypoints_2d"] = final_rh[i]
        if clean_face:
            p["face_keypoints_2d"] = final_face[i]


def _subject_of(frame: Any, index: Optional[int]) -> Optional[Dict[str, Any]]:
    """Return the person dict at ``index`` in ``frame``, or ``None`` if unavailable."""
    if index is None or not isinstance(frame, dict):
        return None
    people = frame.get("people")
    if not isinstance(people, list) or not (0 <= index < len(people)):
        return None
    p = people[index]
    return p if isinstance(p, dict) else None


def smooth_KPS_json_obj(
    data: Any,
    *,
    keep_face_untouched: bool = True,
    keep_hands_untouched: bool = True,
    filter_extra_people: Optional[bool] = None,
    cfg: Optional[SmoothConfig] = None,
) -> Any:
    """Smooth a list of OpenPose-style frame dicts and return the new frames.

    Thin wrapper around :func:`smooth_kps_frames` for callers that only want the
    frames.  See that function for the meaning of every argument.  Prefer it if you
    also need to know which person in each frame is the smoothed subject.
    """
    return smooth_kps_frames(
        data,
        keep_face_untouched=keep_face_untouched,
        keep_hands_untouched=keep_hands_untouched,
        filter_extra_people=filter_extra_people,
        cfg=cfg,
    ).frames
